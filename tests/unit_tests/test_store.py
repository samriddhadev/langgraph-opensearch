import os
import pytest
from datetime import datetime
from collections.abc import Generator
from typing import Any, Dict, cast
from langchain.embeddings import init_embeddings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from opensearchpy import OpenSearch

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    TTLConfig,
)
from langgraph.store.opensearch import OpenSearchStore, VectorIndexConfig

# Setup:
INDEX_NAME = "long_term_memory"
KNN_INDEX_NAME = "knn_long_term_memory"

@pytest.fixture
def store(client_kwargs: Dict[str, Any]) -> Generator:
    """Create a simple store following that in base's test_list_namespaces_basic"""
    client: OpenSearch = OpenSearch(**client_kwargs)
    try:
        client.indices.delete(index=INDEX_NAME)
    except Exception:
        pass
    osstore = OpenSearchStore(
        client=client,
        index_name=INDEX_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    )
    namespaces = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
        ("users", "123"),
        ("users", "456", "settings"),
        ("admin", "users", "789"),
    ]
    for i, ns in enumerate(namespaces):
        osstore.put(namespace=ns, key=f"id_{i}", value={"data": f"value_{i:02d}"})

    yield osstore

    if client:
        client.close()

def test_list_namespaces(store: OpenSearchStore) -> None:
    result = store.list_namespaces(prefix=("a", "b"))
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(suffix=("f",))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("a",), suffix=("f",))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(
        prefix=("a",),
        suffix=(
            "b",
            "f",
        ),
    )
    expected = [("a", "b", "f")]
    assert sorted(result) == sorted(expected)

    # Test max_depth and deduplication
    result = store.list_namespaces(prefix=("a", "b"), max_depth=3)
    expected = [
        ("a", "b", "c"),
        ("a", "b", "d"),
        ("a", "b", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("a", "*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("*", "*", "f"))
    expected = [("a", "c", "f"), ("b", "a", "f"), ("a", "b", "f")]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(suffix=("*", "f"))
    expected = [
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
    ]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("a", "b"), suffix=("d", "i"))
    expected = [("a", "b", "d", "i")]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("a", "b"), suffix=("i",))
    expected = [("a", "b", "d", "i")]
    assert sorted(result) == sorted(expected)

    result = store.list_namespaces(prefix=("nonexistent",))
    assert result == []

    result = store.list_namespaces()
    assert len(result) == store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"]

def test_get(store: OpenSearchStore) -> None:
    result = store.get(namespace=("a", "b", "d", "i"), key="id_2")
    assert isinstance(result, Item)
    assert result.updated_at > result.created_at
    assert result.value == {"data": f"value_{2:02d}"}

    result = store.get(namespace=("a", "b", "d", "i"), key="id-2")
    assert result is None

    result = store.get(namespace=tuple(), key="id_2")
    assert result is None

    result = store.get(namespace=("a", "b", "d", "i"), key="")
    assert result is None

    # Test case: refresh_ttl is False
    hits =store.client.search(
        index=store.index_name,
        body={
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"namespace": ["a", "b", "d", "i"]}},
                        {"match": {"key": "id_2"}}
                    ]
                }
            }
        }
    )
    result = hits["hits"]["hits"][0]["_source"]
    assert result is not None
    expected_updated_at = datetime.fromisoformat(cast(str, result["updated_at"]))

    result = store.get(namespace=("a", "b", "d", "i"), key="id_2", refresh_ttl=False)
    assert result is not None
    assert result.updated_at == expected_updated_at

def test_put(store: OpenSearchStore) -> None:
    n = store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"]
    store.put(namespace=("a",), key=f"id_{n}", value={"data": f"value_{n:02d}"})
    assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == n + 1
    store.put(namespace=("a",), key=f"id_{n}", value={"data": f"value_{n:02d}"})
    assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == n + 1
    store.put(namespace=("a",), key="idx", value={"data": "val"}, index=["data"])

def test_delete(store: OpenSearchStore) -> None:
    n_items = store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"]
    store.delete(namespace=("a", "b", "c"), key="id_0")
    assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == n_items - 1
    store.delete(namespace=("a", "b", "c"), key="id_0")
    assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == n_items - 1

def test_search_basic(store: OpenSearchStore) -> None:
    result = store.search(("a", "b"))
    assert len(result) == 4
    assert all(isinstance(res, Item) for res in result)

    namespace = ("a", "b", "c")
    store.put(namespace=namespace, key="id_foo", value={"data": "value_foo"})
    result = store.search(namespace, filter={"data": "value_foo"})
    assert len(result) == 1

    result = store.search("*")
    assert len(result) == 10

    result = store.search(("a", "*", "c"))
    assert len(result) == 2

def test_batch(client_kwargs: Dict[str, Any]) -> None:
    """Test batch operations in OpenSearchStore."""
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}

    op_put = PutOp(namespace=namespace, key=key, value=value)
    op_del = PutOp(namespace=namespace, key=key, value=None)
    op_get = GetOp(namespace=namespace, key=key)
    cond_pre = MatchCondition(match_type="prefix", path=("a", "b"))
    cond_suf = MatchCondition(match_type="suffix", path=("d", "e"))
    op_list = ListNamespacesOp(match_conditions=(cond_pre, cond_suf))

    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=INDEX_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        # 1. Put 1, read it, list namespaces, and delete one item.
        #   => not any(results)
        store.client.delete_by_query(index=store.index_name, body={"query": {"match_all": {}}})
        store.client.indices.refresh(index=store.index_name)
        n_ops = 4
        results = store.batch([op_put, op_get, op_list, op_del])
        assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == 0
        assert len(results) == n_ops
        assert not any(results)

        # 2. delete, put, get
        # => not any(results)
        n_ops = 3
        results = store.batch([op_get, op_del, op_put])
        assert store.client.count(index=store.index_name, body={"query": {"match_all": {}}})["count"] == 1
        assert len(results) == n_ops
        assert not any(results)

        # 3. delete, put, get
        # => get sees item from put in previous batch
        n_ops = 2
        results = store.batch([op_del, op_get, op_list])
        assert results[0] is None
        assert isinstance(results[1], Item)
        assert isinstance(results[2], list) and isinstance(results[2][0], tuple)

def test_ttl(client_kwargs: Dict[str, Any]) -> None:
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}

    # refresh_on_read is True
    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=INDEX_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        store.client.delete_by_query(index=store.index_name, body={"query": {"match_all": {}}})
        store.client.indices.refresh(index=store.index_name)
        store.put(namespace=namespace, key=key, value=value)
        res = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0] 
        assert res is not None
        orig_updated_at = datetime.fromisoformat(cast(str, res["_source"]["updated_at"]))
        res = store.get(namespace=namespace, key=key)
        assert res is not None
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0] 
        assert found is not None
        new_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is False
    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=INDEX_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=False),
    ) as store:
        store.client.delete_by_query(index=store.index_name, body={"query": {"match_all": {}}})
        store.client.indices.refresh(index=store.index_name)
        store.put(namespace=namespace, key=key, value=value)
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        orig_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        res = store.get(namespace=namespace, key=key)
        assert res is not None
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        new_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at

    # ttl_config is None
    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=INDEX_NAME,
        ttl_config=None,
    ) as store:
        store.client.delete_by_query(index=store.index_name, body={"query": {"match_all": {}}})
        store.client.indices.refresh(index=store.index_name)
        store.put(namespace=namespace, key=key, value=value)
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        orig_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        res = store.get(namespace=namespace, key=key)
        assert res is not None
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        new_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        assert new_updated_at > orig_updated_at
        assert res.updated_at == new_updated_at

    # refresh_on_read is True but refresh_ttl=False in get()
    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=INDEX_NAME,
        ttl_config=TTLConfig(default_ttl=3600, refresh_on_read=True),
    ) as store:
        store.client.delete_by_query(index=store.index_name, body={"query": {"match_all": {}}})
        store.client.indices.refresh(index=store.index_name)
        store.put(namespace=namespace, key=key, value=value)
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        orig_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        res = store.get(refresh_ttl=False, namespace=namespace, key=key)
        assert res is not None
        found = store.client.search(
            index=store.index_name,
            body={
                "query": {
                    "match": {
                        "key": key
                    }
                }
            }
        )['hits']['hits'][0]
        assert found is not None
        new_updated_at = datetime.fromisoformat(cast(str, found["_source"]["updated_at"]))
        assert new_updated_at == orig_updated_at
        assert res.updated_at == new_updated_at

@pytest.mark.skip(reason="Skipping this test temporarily")
def test_knn_search(client_kwargs: Dict[str, Any]) -> None:
    namespace = ("a", "b", "c", "d", "e")
    key = "thread"
    value = {"human": "What is the weather in SF?", "ai": "It's always sunny in SF."}
    
    client: OpenSearch = OpenSearch(**client_kwargs)
    try:
        client.indices.delete(index=KNN_INDEX_NAME)
    except Exception:
        pass
    
    # refresh_on_read is True
    with OpenSearchStore.from_conn_string(
        client_kwargs=client_kwargs,
        index_name=KNN_INDEX_NAME,
        ttl_config=TTLConfig(refresh_on_read=True),
        index_config=VectorIndexConfig(
            vector_field="vector", 
            fields=["ai"],
            embed=init_embeddings("openai:text-embedding-3-small")
        )
    ) as store:
        store.put(namespace=namespace, key=key, value=value)
        res = store.search(
            namespace,
            query="weather in SF"
        )
        assert len(res) == 1
        assert isinstance(res[0], Item)
        assert res[0].value["ai"] == "It's always sunny in SF."
        assert res[0].value["human"] == "What is the weather in SF?"