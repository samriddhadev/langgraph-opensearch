import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4
from typing import (
    Any,
    Optional,
    Dict,
    List
)

from langchain_core.runnables import RunnableConfig

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError, TransportError
from opensearchpy.helpers import bulk

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

from .utils import dumps_metadata, loads_metadata
from .jsonplus_ext_opensearch import JSONPlusExtOSSerializer

class OpenSearchSaver(BaseCheckpointSaver):
    """    
    OpenSearchSaver is a class designed to manage checkpoints and writes in an OpenSearch database. 
    It provides methods for saving, retrieving, listing, and deleting checkpoints, as well as storing 
    intermediate writes associated with checkpoints. This class is useful for applications that require 
    persistent storage and retrieval of checkpoint data in a distributed environment.
    Attributes:
        client (OpenSearch): The OpenSearch client instance used for database operations.
        checkpoint_index_name (str): The name of the index used for storing checkpoints.
        writes_index_name (str): The name of the index used for storing checkpoint writes.
        ttl (Optional[int]): Time-to-live for checkpoint entries, in seconds.
    Methods:
        __init__(client, checkpoint_index_name, writes_index_name, ttl, **kwargs):
            Initializes the OpenSearchSaver instance and ensures the required indexes exist.
        from_conn_string(connection, checkpoint_index_name, writes_index_name, ttl, **kwargs):
            A context manager for creating an instance of OpenSearchSaver using a connection string.
        close():
            Closes the OpenSearch client.
        get_tuple(config):
        list(config, filter=None, before=None, limit=None):
        put(config, checkpoint, metadata, new_versions):
            Saves a checkpoint to the OpenSearch database.
        put_writes(config, writes, task_id, task_path=""):
            Stores intermediate writes linked to a checkpoint.
        delete_thread(thread_id):
            Deletes all checkpoints and writes associated with a specific thread ID.
        _build_query_dsl(parameters):
            Builds an OpenSearch query DSL from the provided parameters.
        Example 1: Creating an OpenSearchSaver instance using a connection string
        >>> with OpenSearchSaver.from_conn_string(client_kwargs={
        >>>     'hosts': [{'host': os.getenv('OSS_HOST'), 'port': 443}],
        >>>     'http_auth': awsauth,
        >>>     'use_ssl': True,
        >>>     'verify_certs': True,
        >>>     'connection_class': RequestsHttpConnection
        >>> }) as checkpointer:
        >>>     
        >>>     config = {
        >>>         'configurable': {
        >>>             'thread_id': '3'
        >>>         }
        >>>     }
        >>>     graph = graph.compile(checkpointer=checkpointer)
        >>>     # Run the graph with an initial message
        >>>     response = graph.invoke(
        >>>         {
        >>>             "messages": [
        >>>                 HumanMessage(content="What is the capital of France?")
        >>>             ]
        >>>         },
        >>>         config
        >>>     )
        >>>     print(response)
        >>> 
        >>>     response = graph.invoke(
        >>>         {
        >>>             "messages": [
        >>>                 HumanMessage(content="What are its key attractions?")
        >>>             ]
        >>>         },
        >>>         config
        >>>     )
        >>>     print(response)
    """

    client: OpenSearch

    def __init__(
        self,
        client: OpenSearch,
        checkpoint_index_name: str = "checkpoints",
        writes_index_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.serde = JSONPlusExtOSSerializer(**kwargs)
        self.client = client
        self.ttl = ttl
        self.checkpoint_index_name = checkpoint_index_name
        self.writes_index_name = writes_index_name

        _checkpointer_mapping = {
            "mappings": {
                "properties": {
                    "checkpoint_id":    {"type": "keyword"},
                    "checkpoint_ns":    {"type": "keyword"},
                    "thread_id":     {"type": "keyword"},
                    "parent_checkpoint_id": {"type": "keyword"}
                }
            }
        }

        _checkpointer_writes_mapping = {
            "mappings": {
                "properties": {
                    "checkpoint_id":    {"type": "keyword"},
                    "checkpoint_ns":    {"type": "keyword"},
                    "thread_id":     {"type": "keyword"},
                    "task_id": {"type": "keyword"},
                    "task_path": {"type": "keyword"},
                    "idx": {"type": "integer"}
                }
            }
        }

        # Create indexes if not present
        try:
            client.indices.get(index=checkpoint_index_name)
        except NotFoundError:
            client.indices.create(index=checkpoint_index_name, body=_checkpointer_mapping)

        try:
            client.indices.get(index=writes_index_name)
        except NotFoundError:
            client.indices.create(index=writes_index_name, body=_checkpointer_writes_mapping)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        client_kwargs: dict[str, Any],
        checkpoint_index_name: str = "checkpoints",
        writes_index_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator["OpenSearchSaver"]:
        """
        A context manager for creating an instance of `OpenSearchSaver` using a connection string.
        This method initializes an OpenSearch client with the provided connection details
        and yields an `OpenSearchSaver` instance configured with the specified parameters.
        The OpenSearch client is automatically closed when the context manager exits.
        Args:
            client_kwargs (dict[str, Any]): A dictionary containing connection parameters for OpenSearch.
            checkpoint_index_name (str, optional): The name of the index used for storing checkpoints.
                Defaults to "checkpoints".
            writes_index_name (str, optional): The name of the index used for storing checkpoint writes.
                Defaults to "checkpoint_writes".
            ttl (Optional[int], optional): Time-to-live for checkpoint entries, in seconds. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to the `OpenSearchSaver` constructor.
        Yields:
            Iterator[OpenSearchSaver]: An instance of `OpenSearchSaver` configured with the provided parameters.
        Raises:
            Any exceptions raised during the initialization of the OpenSearch client or the `OpenSearchSaver`.
        Example:
            connection = {"hosts": ["localhost:9200"], "http_auth": ("user", "password")}
            with OpenSearchSaver.from_conn_string(connection) as saver:
                # Use the saver instance for checkpoint operations
        """
        client: Optional[OpenSearch] = None
        try:
            client = OpenSearch(**client_kwargs)
            yield OpenSearchSaver(
                client,
                checkpoint_index_name,
                writes_index_name,
                ttl,
                **kwargs,
            )
        finally:
            if client:
                client.close()

    def close(self) -> None:
        """Close the resources used by the OpenSearch."""
        self.client.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieves a checkpoint tuple based on the provided configuration.
        This method queries the OpenSearch index to fetch checkpoint data and 
        associated pending writes. It constructs a `CheckpointTuple` object 
        containing the checkpoint information, metadata, parent checkpoint 
        (if available), and pending writes.
        Args:
            config (RunnableConfig): The configuration object containing 
            details such as `thread_id`, `checkpoint_ns`, and other 
            parameters required to identify the checkpoint.
        Returns:
            Optional[CheckpointTuple]: A tuple containing:
            - Configuration values used for the query.
            - The checkpoint object deserialized from the search result.
            - Metadata associated with the checkpoint.
            - Parent checkpoint configuration (if available).
            - List of pending writes associated with the checkpoint.
              Returns `None` if no checkpoint data is found.
        
        Examples:
            >>> from langgraph.checkpoint.opensearch import OpenSearchSaver
            >>> from opensearchpy import OpenSearch
            >>> connection = {"hosts": [{"host": "localhost", "port": 9200}]}
            >>> with OpenSearchSaver.from_conn_string(connection) as saver:
            ...     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            ...     checkpoint_tuple = saver.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': 'abc123'}}, checkpoint=..., metadata=..., parent_config=..., pending_writes=...)
        
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parameters = {}
        if checkpoint_id := get_checkpoint_id(config):
            parameters = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            parameters = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self._search(
            index=self.checkpoint_index_name, 
            query=self._build_query_dsl(parameters),
            sort=[{"checkpoint_id": "desc"}],  # Sort by index
            size=1,  # Limit to the most recent checkpoint
            scroll=False  # Disable scrolling for single result retrieval
        )
        
        for item in result:
            doc = item["_source"]
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads(doc["checkpoint"])
            response = self._search(
                index=self.writes_index_name, 
                query=self._build_query_dsl(config_values),
                sort=[{"checkpoint_id": "desc"}],  # Sort by index
            )
            serialized_writes = response[0] if response else []
            pending_writes = [
                (
                    doc['_source']["task_id"],
                    doc['_source']["channel"],
                    self.serde.loads(doc['_source']["value"]),
                )
                for doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                loads_metadata(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        Lists checkpoints based on the provided configuration, filters, and other parameters.
        This method queries the checkpoint index and retrieves checkpoint documents that match
        the specified parameters. It also fetches associated pending writes for each checkpoint
        and yields them as `CheckpointTuple` objects.
        Args:
            config (Optional[RunnableConfig]): Configuration object containing checkpoint-related
                parameters such as `thread_id` and `checkpoint_ns`.
            filter (Optional[dict[str, Any]]): Metadata filter to narrow down the search results.
                Keys represent metadata fields, and values are the expected values for those fields.
            before (Optional[RunnableConfig]): Configuration object specifying a checkpoint ID
                to limit results to those created before this checkpoint.
            limit (Optional[int]): Maximum number of checkpoints to retrieve.
        Returns:
            Iterator[CheckpointTuple]: An iterator over `CheckpointTuple` objects, each containing
            checkpoint data, metadata, parent configuration (if applicable), and pending writes.
        Example:
            ```python
            config = {
                    "thread_id": "12345",
                    "checkpoint_ns": "namespace1"
            filter = {"status": "active"}
            checkpoints = saver.list(config=config, filter=filter)
            for checkpoint in checkpoints:
                print(checkpoint.config)
                print(checkpoint.metadata)
                print(checkpoint.pending_writes)
            ```
        """
        parameters = {}
        range = {}
        if config is not None:
            if "thread_id" in config["configurable"]:
                parameters["thread_id"] = config["configurable"]["thread_id"]
            if "checkpoint_ns" in config["configurable"]:
                parameters["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]

        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        parameters[f"metadata.{key}.{sub_key}"] = dumps_metadata(sub_value)
                else:
                    parameters[f"metadata.{key}"] = dumps_metadata(value)

        if before is not None:
            range["checkpoint_id"] = {
                "lt": before["configurable"]["checkpoint_id"]}
        result  = self._search(
            index=self.checkpoint_index_name,
            query=self._build_query_dsl(parameters, range),
            sort=[{"checkpoint_id": "desc"}],  # Sort by index
            size=limit if limit is not None else 1,  # Limit results to specified size
        )
        for item in result:
            doc = item["_source"]
            config_values = {
                "thread_id": doc["thread_id"],
                "checkpoint_ns": doc["checkpoint_ns"],
                "checkpoint_id": doc["checkpoint_id"],
            }
            serialized_writes = self._search(
                index=self.writes_index_name,
                query=self._build_query_dsl(config_values)
            )
            pending_writes = [
                (
                    wrt['_source']["task_id"],
                    wrt['_source']["channel"],
                    self.serde.loads(wrt['_source']["value"]),
                )
                for wrt in serialized_writes
            ]

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint=self.serde.loads(doc["checkpoint"]),
                metadata=loads_metadata(doc["metadata"]),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database.
        This method saves a checkpoint along with its metadata and version information 
        into the database. If a checkpoint with the same thread ID, namespace, and ID 
        already exists, it updates the existing record. Otherwise, it creates a new record.
        Args:
            config (RunnableConfig): Configuration object containing details such as 
                thread ID and checkpoint namespace.
            checkpoint (Checkpoint): The checkpoint data to be saved.
            metadata (CheckpointMetadata): Metadata associated with the checkpoint.
            new_versions (ChannelVersions): Version information for the checkpoint.
        Returns:
            RunnableConfig: Updated configuration object containing thread ID, 
            checkpoint namespace, and checkpoint ID.
        Raises:
            Any exceptions raised by the underlying database client during search, 
            update, or index operations.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]

        parameters = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
        }

        response = self._search(
            index=self.checkpoint_index_name, 
            query=self._build_query_dsl(parameters),
            size=1,  # Limit to one result
        )
        result = response[0]["_source"] if response else None
        serialized_checkpoint = self.serde.dumps(checkpoint)
        type_ = 'json' if isinstance(serialized_checkpoint, str) else 'bytes'
        if result:
            doc = {
                "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                "type": type_,
                "checkpoint": serialized_checkpoint,
                "metadata": dumps_metadata(metadata),
                "created_at": datetime.now(),
            }
            self.client.update(
                index=self.checkpoint_index_name,
                id=result["id"],
                body={"doc": doc}
            )
            self._try_refresh(self.checkpoint_index_name)
        else:
            doc = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                "type": type_,
                "checkpoint": serialized_checkpoint,
                "metadata": dumps_metadata(metadata),
            }
            self.client.index(
                index=self.checkpoint_index_name,
                body=doc
            )
            self._try_refresh(self.checkpoint_index_name)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the MongoDB database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        set_method = (  # Allow replacement on existing writes only if there were errors.
            "update" if all(
                w[0] in WRITES_IDX_MAP for w in writes) else "index"
        )
        operations = []
        type_ = 'json'  # Default type for serialized values
        for idx, (channel, value) in enumerate(writes):
            serialized_value = self.serde.dumps(value)
            if set_method == "update":
                # If using update, we need to ensure the document exists
                parameters = {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                }
                response = self._search(
                    index=self.writes_index_name,
                    query=self._build_query_dsl(parameters),
                    size=1,  # Limit to one result 
                )
                op = {
                    "_op_type": set_method,
                    "_index": self.writes_index_name,
                    "_id": response[0]["_id"] if response else str(uuid4()),
                    "doc": {
                        "channel": channel,
                        "type": type_,
                        "value": serialized_value,
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "idx": WRITES_IDX_MAP.get(channel, idx),
                        "created_at": datetime.now(),
                    }
                }
            else:
                op = {
                    "_op_type": set_method,
                    "_index": self.writes_index_name,
                    "_id": str(uuid4()),
                    "_source": {
                        "channel": channel,
                        "type": type_,
                        "value": serialized_value,
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "idx": WRITES_IDX_MAP.get(channel, idx),
                        "created_at": datetime.now(),
                    }
                }
        operations.append(op)
        bulk(self.client, operations)
        self._try_refresh(self.writes_index_name)
        
    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id (str): The thread ID whose checkpoints should be deleted.
        """
        # Delete all checkpoints associated with the thread ID
        self._delete_by_query_safe(
            self.client,
            index=self.checkpoint_index_name,
            query={
                "match": {
                    "thread_id": thread_id
                }
            }
        )
        self._try_refresh(self.checkpoint_index_name)
        # Delete all writes associated with the thread ID
        self._delete_by_query_safe(
            self.client,
            index=self.writes_index_name,
            query={
                "match": {
                    "thread_id": thread_id
                }
            }
        )
        self._try_refresh(self.writes_index_name)

    def _search(self, 
                index: str, 
                query: Dict, 
                sort: Optional[List[Dict[str, str]]] = None, 
                size: int = 1000,
                scroll: bool = True) -> List[Dict[str, Any]]:
        query = {
            "size": size,
            "query": query,
            "sort": sort if sort else [{"_id": "asc"}]  # You must sort on a unique & indexed field
        }
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

    def _build_query_dsl(self, match: Dict[str, Any], range:Dict[str, Any] = None) -> dict[str, Any]:
        """Build OpenSearch query DSL from parameters."""
        q = {}
        if len(match) == 1 and not range:
            key = list(match.keys())[0]
            # If only single parameter is provided, return a simple match query
            q = {"match": {key: match[key]}}
        else:
            # Build a more complex query with multiple conditions
            compound_q = {
                "bool": {
                    "must": []
                }
            }
            for key in match.keys():
                compound_q["bool"]["must"].append({"match": {key: match[key]}})
            if range:
                compound_q["bool"]["must"].append({"range": range})
            q = compound_q
        print("OpenSearch query DSL:", json.dumps(q, indent=2))  # Debugging output
        return q

    def _delete_by_query_safe(self, client: OpenSearch, index: str, query: dict, scroll: str = "2m", batch_size: int = 500):
        try:
            # Try native delete_by_query (works in standard OpenSearch)
            response = client.delete_by_query(index=index, body={"query": query})
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
                {"_op_type": "delete", "_index": hit["_index"], "_id": hit["_id"]}
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