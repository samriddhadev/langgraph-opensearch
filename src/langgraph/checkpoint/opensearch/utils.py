"""
Utilities for handling checkpoint metadata serialization and deserialization.

This module provides functions to serialize and deserialize metadata objects 
for efficient storage and retrieval, particularly in OpenSearch. It includes 
support for recursive processing of nested dictionaries and uses a custom 
serialization protocol for handling complex data types.

Functions:
- loads_metadata: Converts a dictionary representation of metadata into a 
    CheckpointMetadata object, recursively deserializing nested structures.
- dumps_metadata: Serializes metadata into a format suitable for storage or 
    transmission, supporting both dictionary and non-dictionary inputs.

Dependencies:
- SerializerProtocol: Defines the serialization interface.
- JsonPlusSerializer: Implements the serialization protocol for JSON-like data.
"""

from typing import Any, Union

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.serde.base import SerializerProtocol

from .jsonplus_ext_opensearch import JSONPlusExtOSSerializer

serde: SerializerProtocol = JSONPlusExtOSSerializer()


def loads_metadata(metadata: dict[str, Any]) -> CheckpointMetadata:
    """
    Deserialize a metadata document stored in OpenSearch.

    This function converts a dictionary representation of metadata into a 
    CheckpointMetadata object. Since the CheckpointMetadata class cannot be 
    directly stored in OpenSearch, the metadata is stored as a dictionary 
    with string keys and serialized values for efficient filtering.

    Args:
        metadata (dict[str, Any]): A dictionary containing metadata with 
        string keys and serialized values, retrieved from an OpenSearch collection.

    Returns:
        CheckpointMetadata: A deserialized CheckpointMetadata object.

    Notes:
        - The function recursively deserializes nested dictionaries.
        - Serialized values are deserialized using the `serde.loads` method.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = loads_metadata(value)
        return output
    else:
        return serde.loads(metadata)


def dumps_metadata(
    metadata: Union[CheckpointMetadata, Any],
) -> Union[bytes, dict[str, Any]]:
    """
    Serializes metadata into a format suitable for storage or transmission.
    If the input `metadata` is a dictionary, it recursively serializes each key-value pair
    and returns a dictionary with serialized values. Otherwise, it uses `serde.dumps` to
    serialize the input directly.
    Args:
        metadata (Union[CheckpointMetadata, Any]): The metadata to be serialized. 
            It can be a dictionary or any other object compatible with `serde.dumps`.
    Returns:
        Union[bytes, dict[str, Any]]: Serialized metadata. If the input is a dictionary, 
        the output is a dictionary with serialized values. Otherwise, the output is a 
        serialized byte representation of the input.
    """
    if isinstance(metadata, dict):
        output = dict()
        for key, value in metadata.items():
            output[key] = dumps_metadata(value)
        return output
    else:
        return serde.dumps(metadata)
