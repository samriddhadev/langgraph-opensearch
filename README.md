# LangGraph OpenSearch

This repository contains OpenSearch implementations for LangGraph, providing both Checkpoint Savers and Stores functionality.

## Overview

The project consists of two main components:

1. **OpenSearch Checkpoint Savers**: Implementations for storing and managing checkpoints using OpenSearch

## Dependencies

### Python Dependencies

The project requires the following main Python dependencies:

- `opensearch-py>=2.0.0`
- `langgraph-checkpoint>=2.0.24`

### OpenSearch Requirements

Ensure your OpenSearch instance is properly configured and accessible.

## Installation

Install the library using pip:

```bash
pip install langgraph-checkpoint-opensearch
```

## OpenSearch Checkpoint Savers

### Important Notes

> [!IMPORTANT]
> When using OpenSearch checkpointers for the first time, make sure to call `.setup()` method on them to create required indices. See examples below.

### Standard Implementation

```python
from langgraph.checkpoint.opensearch import OpenSearchSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

with OpenSearchSaver.from_conn_string(connection={'host': 'localhost', 'port': 9200}) as checkpointer:
    # Call setup to initialize indices
    checkpointer.setup()
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # Store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # Retrieve checkpoint
    loaded_checkpoint = checkpointer.get(read_config)

    # List all checkpoints
    checkpoints = list(checkpointer.list(read_config))
```
## Examples

The `examples` directory contains Jupyter notebooks demonstrating the usage of OpenSearch with LangGraph:

- `persistence_opensearch.ipynb`: Demonstrates the usage of OpenSearch checkpoint savers with LangGraph
- `create-react-agent-memory.ipynb`: Shows how to create an agent with persistent memory using OpenSearch
- `cross-thread-persistence.ipynb`: Demonstrates cross-thread persistence capabilities
- `persistence-functional.ipynb`: Shows functional persistence patterns with OpenSearch

### Running Example Notebooks

To run the example notebooks with Docker:

1. Navigate to the examples directory:

   ```bash
   cd examples
   ```

2. Start the Docker containers:

   ```bash
   docker compose up
   ```

3. Open the URL shown in the console (typically <http://127.0.0.1:8888/tree>) in your browser to access Jupyter.

4. When finished, stop the containers:

   ```bash
   docker compose down
   ```

## Implementation Details

### OpenSearch Indexing

The OpenSearch implementation creates these main indices:

1. **Checkpoints Index**: Stores checkpoint metadata and versioning
2. **Channel Values Index**: Stores channel-specific data
3. **Writes Index**: Tracks pending writes and intermediate states

For OpenSearch Stores with vector search:

1. **Store Index**: Main key-value store
2. **Vector Index**: Optional vector embeddings for similarity search

## Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/opensearch-developer/langgraph-opensearch
   cd langgraph-opensearch
   ```

2. Install dependencies:

   ```bash
   `poetry install --all-extras`
   ```

### Available Commands

The project includes several make commands for development:

- **Testing**:

  ```bash
  make test           # Run all tests
  make test-all       # Run all tests including API tests
  ```

- **Linting and Formatting**:

  ```bash
  make format        # Format all files with Black and isort
  make lint          # Run formatting, type checking, and other linters
  make check-types   # Run mypy type checking
  ```

- **OpenSearch for Development/Testing**:

  ```bash
  make opensearch-start   # Start OpenSearch in Docker
  make opensearch-stop    # Stop OpenSearch container
  ```

### Contribution Guidelines

1. Create a new branch for your changes
2. Write tests for new functionality
3. Ensure all tests pass: `make test`
4. Format your code: `make format`
5. Run linting checks: `make lint`
6. Submit a pull request with a clear description of your changes
7. Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages

## License

This project is licensed under the MIT License.
