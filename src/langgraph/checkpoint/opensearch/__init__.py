# This module initializes the OpenSearch checkpoint functionality for LangGraph.
# It provides the necessary imports and exposes the public API for external use.
from .saver import OpenSearchSaver

__all__ = [
    'OpenSearchSaver'  # Expose OpenSearchSaver as part of the public API
]
