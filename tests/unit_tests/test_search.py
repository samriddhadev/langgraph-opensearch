import os
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from opensearchpy import OpenSearch

from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.opensearch import OpenSearchSaver

# Setup:
# docker run --name mongodb -d -p 27017:27017 mongodb/mongodb-community-server
OSS_HOST = os.environ.get("OSS_HOST", "localhost")
OSS_PORT = int(os.environ.get("OSS_PORT", "9200"))
CHECKPOINT_INDEX_NAME = "test_checkpoints"
CHECKPOINT_WRITER_INDEX_NAME = "test_writes_checkpoints"

def test_search(input_data: dict[str, Any], client_kwargs: Dict[str, Any]) -> None:
    """
    Test the functionality of the `list` and `put` methods in the `OpenSearchSaver` class.

    This test performs the following steps:
    1. Clears existing indices in the OpenSearch client.
    2. Saves multiple checkpoints with associated configurations and metadata using the `put` method.
    3. Executes various search queries using the `list` method to validate:
        - Searching by a single key.
        - Searching by multiple keys.
        - Searching with no keys (returning all checkpoints).
        - Searching with a query that matches no checkpoints.
        - Searching by configuration across namespaces.

    Assertions:
    - Validates the number of search results returned for each query.
    - Ensures the metadata and configurations of the returned checkpoints match the expected values.

    Args:
         input_data (dict[str, Any]): A dictionary containing the configurations, checkpoints, and metadata
         required for testing. Expected keys include:
              - "config_1", "config_2", "config_3": Configuration data for the checkpoints.
              - "chkpnt_1", "chkpnt_2", "chkpnt_3": Checkpoint data to be saved.
              - "metadata_1", "metadata_2", "metadata_3": Metadata associated with the checkpoints.

    Raises:
         AssertionError: If any of the assertions fail during the test.
    """
    # Clear collections if they exist
    client = OpenSearch(**client_kwargs)

    try:
        client.indices.delete(index=CHECKPOINT_INDEX_NAME)
        client.indices.delete(index=CHECKPOINT_WRITER_INDEX_NAME)
    except:
        pass

    with OpenSearchSaver.from_conn_string(
            client_kwargs=client_kwargs,
            checkpoint_index_name=CHECKPOINT_INDEX_NAME,
            writes_index_name=CHECKPOINT_WRITER_INDEX_NAME) as saver:
        # save checkpoints
        saver.put(
            input_data["config_1"],
            input_data["chkpnt_1"],
            input_data["metadata_1"],
            {},
        )
        saver.put(
            input_data["config_2"],
            input_data["chkpnt_2"],
            input_data["metadata_2"],
            {},
        )
        saver.put(
            input_data["config_3"],
            input_data["chkpnt_3"],
            input_data["metadata_3"],
            {},
        )

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        # search by no keys, return all checkpoints
        query_3: dict[str, Any] = {}
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == input_data["metadata_1"]

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == input_data["metadata_2"]

        search_results_3 = list(saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(saver.list(
            {"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


def test_nested_filter(client_kwargs: Dict[str, Any]) -> None:
    """
    Test the functionality of saving, retrieving, and validating nested filter data 
    in an OpenSearch index using the OpenSearchSaver class.
    This test performs the following steps:
    1. Creates a checkpoint with metadata and a nested filter structure.
    2. Saves the checkpoint to an OpenSearch index using the OpenSearchSaver.
    3. Retrieves the saved checkpoint using a nested filter query and validates 
       that the retrieved data matches the input.
    4. Confirms the serialization structure of the data stored in the index.
    5. Validates the checkpoint values both from the checkpointer and the database.
    6. Cleans up by deleting the created OpenSearch indices.
    Assertions:
    - The retrieved checkpoint metadata matches the input message.
    - The serialized structure of the data in the index is as expected.
    - The checkpoint values match the input message both in memory and in the database.
    Raises:
        AssertionError: If any of the assertions fail during the test.
    """
    input_message = HumanMessage(content="This is Test")
    thread_id = "thread-3"

    config = RunnableConfig(
        configurable=dict(thread_id=thread_id,
                          checkpoint_id="1", checkpoint_ns="")
    )
    chkpt = empty_checkpoint()
    chkpt["channel_values"] = input_message

    metadata = CheckpointMetadata(
        source="loop", step=1, writes={"message": input_message}
    )

    with OpenSearchSaver.from_conn_string(
            client_kwargs=client_kwargs,
            checkpoint_index_name=CHECKPOINT_INDEX_NAME,
            writes_index_name=CHECKPOINT_WRITER_INDEX_NAME) as saver:
        saver.put(config, chkpt, metadata, {})

        results = list(saver.list(
            None, filter={"writes.message": input_message}))
        for cptpl in results:
            assert cptpl.metadata["writes"]["message"] == input_message
            break

        # Confirm serialization structure of data in index
        data = saver.client.search(body={
            "query": {
                "match": {
                    "thread_id": thread_id
                }
            }},
            index=CHECKPOINT_INDEX_NAME
        )
        doc = data["hits"]["hits"][0]["_source"]
        assert isinstance(doc["checkpoint"], str)
        assert (
            isinstance(doc["metadata"], dict)
            and isinstance(doc["metadata"]["writes"], dict)
            and isinstance(doc["metadata"]["writes"]["message"], str)
        )

        # In database
        chkpt_db = saver.serde.loads(doc["checkpoint"])
        assert chkpt_db["channel_values"] == input_message

        # Drop indexes
        saver.client.indices.delete(index=CHECKPOINT_INDEX_NAME)
        saver.client.indices.delete(index=CHECKPOINT_WRITER_INDEX_NAME)
