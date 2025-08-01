import uuid
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional, Union, List, Dict


from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor


from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError, TransportError
from opensearchpy.helpers import bulk

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.embed import (
    ensure_embeddings,
    get_text_at_path,
)

logger = logging.getLogger(__name__)

_prefix_script_query = """
    def prefix = params.prefix;
    def namespace = params._source.namespace;

    boolean useWildcard = params.containsKey('use_wildcard') ? params.use_wildcard : false;
    boolean useDepth = params.containsKey('max_depth');

    int depth = prefix.size();
    if (useDepth) {
        int maxDepth = params.max_depth;
        depth = (int)Math.min((int)Math.min(namespace.size(), prefix.size()), maxDepth);
    } else {
        if (namespace.size() < prefix.size()) return 0;
    }

    for (int i = 0; i < depth; i++) {
        if (useWildcard && prefix[i] == "*") continue;
        if (namespace[i] != prefix[i]) return 0;
    }
    return 1;
"""

_suffix_script_query = """
    def suffix = params.suffix;
    def namespace = params._source.namespace;

    boolean useWildcard = params.containsKey('use_wildcard') ? params.use_wildcard : false;
    boolean useDepth = params.containsKey('max_depth');

    if (namespace == null || suffix == null) return 0;

    int depth = suffix.size();
    if (useDepth) {
        int maxDepth = params.max_depth;
        depth = (int)Math.min((int)Math.min(namespace.size(), suffix.size()), maxDepth);
    } else {
        if (namespace.size() < suffix.size()) return 0;
    }

    int startIndex = namespace.size() - depth;
    for (int i = 0; i < depth; i++) {
        if (useWildcard && suffix[i] == "*") continue;
        if (namespace[startIndex + i] != suffix[i]) return 0;
    }
    return 1;
"""

_all_script_query = """
    def namespace = params._source.namespace;
    if (namespace == null) return 0;

    boolean useWildcard = params.containsKey('use_wildcard') ? params.use_wildcard : false;
    boolean hasMaxDepth = params.containsKey('max_depth');
    int nsSize = namespace.size();

    // ---------------- Prefix Check ----------------
    if (params.containsKey('prefix')) {
        def prefix = params.prefix;
        int prefixSize = prefix.size();
        int prefixDepth = prefixSize;

        if (hasMaxDepth) {
            int maxDepth = params.max_depth;
            prefixDepth = (int)Math.min(prefixSize, Math.min(nsSize, maxDepth));
        } else if (nsSize < prefixSize) {
            return 0;
        }

        for (int i = 0; i < prefixDepth; i++) {
            if (useWildcard && prefix[i] == "*") continue;
            if (namespace[i] != prefix[i]) return 0;
        }
    }

    // ---------------- Suffix Check ----------------
    if (params.containsKey('suffix')) {
        def suffix = params.suffix;
        int suffixSize = suffix.size();
        int suffixDepth = suffixSize;

        if (hasMaxDepth) {
            int maxDepth = params.max_depth;
            suffixDepth = (int)Math.min(suffixSize, Math.min(nsSize, maxDepth));
        } else if (nsSize < suffixSize) {
            return 0;
        }

        int startIndex = nsSize - suffixDepth;
        for (int i = 0; i < suffixDepth; i++) {
            if (useWildcard && suffix[i] == "*") continue;
            if (namespace[startIndex + i] != suffix[i]) return 0;
        }
    }

    return 1;
"""

class VectorIndexConfig(IndexConfig, total=False):
    """
    Configuration class for a vector index in OpenSearch.
    Attributes:
        engine (str): The vector search engine to use. Defaults to "nmslib".
            - Example: "nmslib", "faiss", etc.
        space_type (str): The distance metric or space type for vector similarity.
            Defaults to "l2" (Euclidean distance).
            - Example: "l2", "cosine", "dot_product", etc.
        ef_search (int): The size of the dynamic search list for nearest neighbor queries.
            Higher values improve recall at the cost of latency. Defaults to 512.
        ef_construction (int): The size of the dynamic search list during index construction.
            Higher values improve index quality at the cost of memory and build time. Defaults to 512.
        m (int): The number of bi-directional links created for each element during index construction.
            Higher values improve recall at the cost of memory. Defaults to 16.
        vector_field (str): The name of the field in the index that stores vector data.
            Defaults to "vector_field".
    """

    engine: str = "nmslib"
    space_type: str = "l2"
    ef_search: int = 512
    ef_construction: int = 512
    m: int = 16
    vector_field: str = "vector_field"


class OpenSearchStore(BaseStore):
    """OpenSearch's persistent key-value stores for long-term memory.

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.

    Supports semantic search capabilities through
    an optional `index` configuration.
    Only a single embedding is permitted per item.

    This implementation leverages OpenSearch for indexing and querying,
    providing support for vector-based semantic search and filtering.
    """

    client: OpenSearch

    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        index_config: Optional[IndexConfig] = None,
        ttl_config: Optional[TTLConfig] = None,
        **kwargs: Any,
    ):
        self.client = client
        self.index_name = index_name
        self.index_config = {} if index_config is None else index_config
        self.ttl_config = {} if ttl_config is None else ttl_config

        _mapping = {
            'mappings': {
                'properties': {
                    "namespace": {"type": "keyword"},
                    "text": {"type": "text"},
                    "key": {"type": "keyword"},
                    "value": {"type": "nested"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }

        if self.index_config:
            self.index_field = self._ensure_index_fields(
                self.index_config["fields"])
            self.embeddings: Embeddings = ensure_embeddings(
                self.index_config.get("embed"),
            )
            _mapping['settings'] = {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": self.index_config.get("ef_search", 512)
                }
            }
            _mapping['mappings']['properties'][self.index_config.get("vector_field", "vector")] = {
                "type": "knn_vector",
                "dimension": self.index_config.get("dims", 1536),
                "method": {
                    "name": "hnsw",
                    "space_type": self.index_config.get("space_type", "l2"),
                    "engine": self.index_config.get("engine", "nmslib"),
                    "parameters": {
                        "ef_construction": self.index_config.get(
                            "ef_construction", 512),
                        "m": self.index_config.get(
                            "m", 16
                        ),
                    }
                }
            }
        # Create indexes if not present
        try:
            client.indices.get(index=self.index_name)
        except NotFoundError:
            client.indices.create(index=self.index_name, body=_mapping)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        client_kwargs: dict[str, Any],
        index_name: str = "persistent_store",
        ttl_config: Optional[TTLConfig] = None,
        index_config: Optional[VectorIndexConfig] = None,
        **kwargs: Any,
    ) -> Iterator["OpenSearchStore"]:

        client: Optional[OpenSearch] = None
        try:
            client = OpenSearch(**client_kwargs)
            yield OpenSearchStore(
                client,
                index_name,
                index_config,
                ttl_config,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def close(self) -> None:
        """Close the resources used by the OpenSearch."""
        self.client.close()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        """Retrieve a single item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
            refresh_ttl: Whether to refresh TTLs for the returned item.
                If None (default), uses the store's default refresh_ttl setting.
                If no TTL is specified, this argument is ignored.

        Returns:
            The retrieved item or None if not found.
        """
        refresh_on_read = False if (
            refresh_ttl is False or (self.ttl_config and not self.ttl_config["refresh_on_read"])
        ) else True

        id, res = self._get(
            index=self.index_name,
            key=key,
            namespace=namespace
        )
        now = datetime.now(tz=timezone.utc)
        if res and refresh_on_read:
            doc = {
                "updated_at": now,
            }
            self.client.update(
                index=self.index_name,
                id=id,
                body={"doc": doc}
            )
            self._try_refresh(self.index_name)
        if res:
            return Item(
                value=res["value"],
                key=res["key"],
                namespace=tuple(res["namespace"]),
                created_at=res["created_at"],
                updated_at=now if refresh_on_read else res["updated_at"]
            )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item.

        Args:
            namespace: Hierarchical path for the item.
            key: Unique identifier within the namespace.
        """
        self._delete_by_query_safe(
            self.client,
            index=self.index_name,
            query={
                "bool": {
                    "must": [
                        {"match": {"key": key}},
                        {"terms": {"namespace": list(namespace)}}
                    ]
                }
            }
        )
        self._try_refresh(self.index_name)

    @staticmethod
    def _match_prefix(prefix: NamespacePath, max_depth: int = None) -> tuple[bool, dict[str, Any]]:
        """Helper for list_namespaces."""
        if not prefix or prefix == "*":
            return False, {
                "match_all": {}
            }
        params = {}
        if prefix is not None and prefix != "*":
            params["prefix"] = list(prefix)
        if "*" not in prefix:
            params["use_wildcard"] = False
        else:
            params["use_wildcard"] = True
        if max_depth is not None:
            params["max_depth"] = max_depth
        return True,{
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"""
                            {_prefix_script_query}
                        """,
                        "params": params
                    }
                }
            }
        }
        
    @staticmethod
    def _match_suffix(suffix: NamespacePath, max_depth: int = None) -> tuple[bool, dict[str, Any]]:
        """Helper for list_namespaces."""
        if not suffix or suffix == "*":
            return False, {
                "match_all": {}
            }
        params = {}
        if suffix is not None and suffix != "*":
            params["suffix"] = list(suffix)
        if "*" not in suffix:
            params["use_wildcard"] = False
        else:
            params["use_wildcard"] = True
        if max_depth is not None:
            params["max_depth"] = max_depth
        # Use script_score to match suffix
        return True,{
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"""
                            {_suffix_script_query}
                        """,
                        "params": params
                    }
                }
            }
        }

    @staticmethod
    def _match_all(prefix: NamespacePath = None, suffix: NamespacePath = None, max_depth: int = None) -> tuple[bool, dict[str, Any]]:
        if not prefix and not suffix:
            return False, {"match_all": {}}
        params = {}
        if prefix:
            params["prefix"] = list(prefix)
        
        if suffix:
            params["suffix"] = list(suffix)
        
        if "*" in prefix or "*" in suffix:
            params["use_wildcard"] = True
        else:
            params["use_wildcard"] = False
        
        if max_depth is not None:
            params["max_depth"] = max_depth
        
        return True, {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"""
                            {_all_script_query}
                        """,
                        "params": params
                    }
                }
            }
        }

    def list_namespaces(
        self,
        *,
        prefix: Optional[NamespacePath] = None,
        suffix: Optional[NamespacePath] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store.

        Args:
            prefix: Filter namespaces that start with this path.
            suffix: Filter namespaces that end with this path.
            max_depth: Return namespaces up to this depth in the hierarchy.
            limit: Maximum number of namespaces to return (default 100).
            offset: Number of namespaces to skip for pagination. [Not implemented.]

        Returns: A list of namespace tuples that match the criteria.
        """
        dsl = {"match_all": {}}
        if prefix and not suffix:
            script_score, q = self._match_prefix(prefix, max_depth)
            dsl = q["query"]
        if suffix and not prefix:
            script_score, q = self._match_suffix(suffix, max_depth)
            dsl = q["query"]
        if not prefix and not suffix:
            dsl = {"match_all": {}}
        if prefix and suffix:
            script_score, q = self._match_all(prefix, suffix, max_depth)
            dsl = q["query"]

        if offset:
            raise NotImplementedError("offset is not implemented")

        results = self._search(
            index=self.index_name,
            query={
                "min_score": 0.01,
                "size": limit,
                "query": dsl
            },
            scroll=False
        )
        data = [tuple(res["_source"]["namespace"]) for res in results] if max_depth is None else [tuple(res["_source"]["namespace"][:max_depth]) for res in results]
        return list(set(data))

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations synchronously in a single batch.

        Get, Search, and List operations are performed on state before batch.
        Put and Delete change state. They are deduplicated and applied in order,
        but only after the read operations have completed.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
            The length of output may not match the input as PutOp returns None.
        """
        results: list[Result] = []
        dedupped_putops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        writes: list[dict] = []

        for op in ops:
            if isinstance(op, PutOp):
                dedupped_putops[(op.namespace, op.key)] = op
                results.append(None)

            elif isinstance(op, GetOp):
                results.append(
                    self.get(
                        namespace=op.namespace,
                        key=op.key,
                        refresh_ttl=op.refresh_ttl,
                    )
                )

            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )

            elif isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                if op.match_conditions:
                    for cond in op.match_conditions:
                        if cond.match_type == "prefix":
                            prefix = cond.path
                        elif cond.match_type == "suffix":
                            suffix = cond.path
                        else:
                            raise ValueError(
                                f"Match type {cond.match_type} must be prefix or suffix."
                            )
                results.append(
                    self.list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
        # Apply puts and deletes in bulk
        # Extract texts to embed for each op
        if self.index_config:
            texts = self._extract_texts(list(dedupped_putops.values()))
            vectors = self.embeddings.embed_documents(texts)
            v = 0
        for op in dedupped_putops.values():
            if op.value is None:
                # mark the item for deletion.
                self._delete_by_query_safe(
                    self.client,
                    index=self.index_name,
                    query={
                        "bool": {
                            "must": [
                                {"match": {"key": op.key}},
                                {"terms": {"namespace": list(op.namespace)}}
                            ]
                        }
                    }
                )
                self._try_refresh(self.index_name)
            else:
                # Check if the document exists to set created_at only on insert
                doc_id = self._generate_uuid(op.namespace, op.key)
                exists = self.client.exists(index=self.index_name, id=doc_id)
                now = datetime.now(tz=timezone.utc)
                to_set = {
                    "key": op.key,
                    "namespace": list(op.namespace),
                    "value": op.value,
                    "updated_at": now,
                }
                if not exists:
                    to_set["created_at"] = now
                if self.index_config:
                    to_set[self.index_config.vector_field] = vectors[v]
                    to_set["namespace_prefix"] = self._denormalize_path(op.namespace)
                    v += 1

                # Add the update operation to the bulk request (correct OpenSearch bulk format)
                writes.append({
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": self._generate_uuid(op.namespace, op.key),
                    "doc": to_set,
                    "doc_as_upsert": True  # Use upsert to create if not exists
                })

        if writes:
            bulk(self.client, writes)
            # Refresh index to make changes visible
            self._try_refresh(self.index_name)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously in a single batch.

        Args:
            ops: An iterable of operations to execute.

        Returns:
            A list of results, where each result corresponds to an operation in the input.
            The order of results matches the order of input operations.
        """
        return await run_in_executor(None, self.batch, ops)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
        **kwargs: Any,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        This method supports both natural language search using vector embeddings
        and filtering based on specific field values. The search is scoped to a
        namespace prefix and retrieves items matching the criteria.

        Args:
            namespace_prefix: Hierarchical path prefix to search within.
            query: Optional query for natural language search using vector embeddings.
            filter: Key-value pairs to filter results. Filters should not include the `value.` prefix.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results. [Not implemented.]
            refresh_ttl: TTL is not supported for search. Use `get` if needed.

        Returns:
            List of items matching the search criteria.

        Raises:
            TypeError: If `namespace_prefix` is not a tuple of strings.
            NotImplementedError: If `offset` is specified.
            ValueError: If filters include the `value.` prefix.

        Example:
            Basic filtering:
            ```python
            results = store.search(
                ("docs",),
                filter={"type": "article", "status": "published"}
            )
            ```

            Natural language search:
            ```python
            results = store.search(
                ("docs",),
                query="Find relevant articles"
            )
            ```
        """
        if not isinstance(namespace_prefix, tuple) and namespace_prefix != "*":
            raise TypeError(
                "namespace_prefix must be a non-empty tuple of strings")
        if offset:
            raise NotImplementedError(
                "offset is not implemented in OpenSearchStore")
        if filter:
            if any(f.startswith("value") for f in filter):
                raise ValueError("filters should be specified without `value`")

        dsl = {
            "bool": {
                "must": []
            }
        }
        _has_sub_query = False
        # Add filter conditions
        if filter:
            _has_sub_query = True
            # Create a nested query for filtering
            # OpenSearch requires nested queries for fields like `value`
            # that are stored as nested objects.
            filter_cond = {
                "nested": {
                    "path": "value",
                    "query": {
                        "bool": {
                            "must": []
                        }
                    }
                }
            }
            for k, v in filter.items():
                if k.startswith("value."):
                    # Remove 'value.' prefix for OpenSearch
                    k = k[6:]
                filter_cond["nested"]["query"]["bool"]["must"].append({
                    "term": {
                        f"value.{k}": v
                    }
                })
            dsl["bool"]["must"].append(filter_cond)
        
        # Add knn search if query is provided
        if query:
            _has_sub_query = True
            query_vector = self.embeddings.embed_query(query)
            dsl["bool"]["must"].append({
                "knn": {
                    self.index_config.vector_field: {
                        "vector": query_vector,
                        "k": limit
                    }
                }
            })

        if namespace_prefix:
            script_score, q = self._match_prefix(namespace_prefix)
            if script_score:
                # If we have a script score, we need to wrap the query
                if _has_sub_query:
                    q["query"]["script_score"]["query"] = dsl
                dsl = q["query"]
            else:
                dsl["bool"]["must"].append(q)
                
        results = self._search(
            index=self.index_name,
            query={
                "min_score": 0.01,  # Set a minimum score to filter out low relevance results
                "size": limit,
                "query": dsl
            },
            scroll=False
        )

        return [
            SearchItem(
                namespace=tuple(res['_source']["namespace"]),
                key=res['_source']["key"],
                value=res['_source']["value"],
                created_at=res['_source']["created_at"],
                updated_at=res['_source']["updated_at"],
                score=res.get("_score"),
            )
            for res in results
        ]

    def _denormalize_path(self, paths: Union[tuple[str, ...], list[str]]) -> list[str]:
        """Create list of path parents, for use in $vectorSearch filter.

        ???+ example "Example"
        ```python
        namespace = ('parent', 'child', 'pet')
        prefixes=store_mdb.denormalize_path(namespace)
        assert prefixes == ['parent', 'parent/child', 'parent/child/pet']
        ```
        """
        return [self.sep.join(paths[:i]) for i in range(1, len(paths) + 1)]

    def _extract_texts(self, put_ops: Optional[list[PutOp]]) -> list[str]:
        """Extract text to embed according to index config."""
        if put_ops and self.index_config and self.embeddings:
            to_embed = []
            for op in put_ops:
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        field = self.index_field
                    else:
                        field = self._ensure_index_fields(list(op.index))
                    texts = get_text_at_path(op.value, field)
                    if texts:
                        if len(texts) > 1:
                            raise ValueError(
                                "Got multiple texts. Report as bug.")

                        else:
                            to_embed.append(texts[0])
            return to_embed
        else:
            return []

    @staticmethod
    def _ensure_index_fields(fields: Optional[list[str]]) -> str:
        """Ensure that requested fields to be indexed result in a single vector.

        We require that one document may only have one embedding vector.
        """
        if fields and (len(fields) > 1 or "*" in fields[0]):
            raise ValueError("Only one field can be indexed for queries.")
        if isinstance(fields, list):
            return fields[0]
        else:
            return fields

    def _get(self,
             index: str,
             key: str,
             namespace: Optional[tuple[str, ...]] = None) -> Optional[Item]:
        """Retrieve a single item from the OpenSearch index.
        Args:
            index: The name of the OpenSearch index to query.
            key: The unique identifier for the item.
            namespace: Optional hierarchical path for the item.
        Returns:
            The retrieved item or None if not found.
        """
        if namespace is None:
            query = {
                "match": {
                    "key": key
                }
            }
        else:
            query = {
                "bool": {
                    "must": [
                        {"match": {"key": key}},
                        {"terms": {"namespace": list(namespace)}}
                    ]
                }
            }
        dsl = {
            "size": 1,
            "query": query,
            "sort": [{"_id": "asc"}]
        }
        hits = self._search(index, dsl)
        if not hits or len(hits) == 0:
            return None, None
        # Extract the first hit
        else:
            hit = hits[0]
            id = hit['_id']
            result = hit['_source'] if '_source' in hit else None
        return id, result

    def _search(self,
                index: str,
                query: Dict,
                scroll: bool = True) -> List[Dict[str, Any]]:
        all_hits = []
        if scroll:
            search_after = None
            while True:
                if search_after:
                    query["search_after"] = search_after

                response = self.client.search(index=index, body=query)
                hits = response["hits"]["hits"]
                if not hits:
                    break

                all_hits.extend(hits)
                search_after = hits[-1]["sort"]
        else:
            response = self.client.search(index=index, body=query)
            all_hits = response["hits"]["hits"]
        return all_hits

    def _delete_by_query_safe(self, client: OpenSearch, index: str, query: dict, scroll: str = "2m", batch_size: int = 500):
        try:
            # Try native delete_by_query (works in standard OpenSearch)
            response = client.delete_by_query(
                index=index, body={"query": query})
            return True

        except Exception as e:
            if not isinstance(e, (RequestError, TransportError)):
                raise e  # Raise if it's not a known error

        # Fallback: manual scroll and delete
        deleted = 0
        scroll_resp = client.search(
            index=index,
            body={"query": query},
            scroll=scroll,
            size=batch_size,
            _source=False  # we only need the _id
        )

        scroll_id = scroll_resp.get("_scroll_id")
        hits = scroll_resp["hits"]["hits"]

        while hits:
            actions = [
                {"_op_type": "delete",
                    "_index": hit["_index"], "_id": hit["_id"]}
                for hit in hits
            ]
            bulk(client, actions)
            deleted += len(actions)

            scroll_resp = client.scroll(scroll_id=scroll_id, scroll=scroll)
            scroll_id = scroll_resp.get("_scroll_id")
            hits = scroll_resp["hits"]["hits"]

        client.clear_scroll(scroll_id=scroll_id)
        return True

    def _try_refresh(self, index: str):
        try:
            self.client.indices.refresh(index=index)
        except (RequestError, TransportError) as e:
            pass  # Ignore errors if the index does not exist or is not available

    def _generate_uuid(self, namespace: tuple[str, ...], key: str) -> str:
        combined = "/".join(namespace) + "/" + key
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))
