# LangGraph OpenSearch

This repository contains OpenSearch implementations for LangGraph, providing Checkpoint Savers functionality.

## Overview

The project consists of:

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
pip install langgraph-opensearch
```

## OpenSearch Checkpoint Savers

### Standard Implementation

```python

# Initialize OpenSearchSaver with the graph
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.opensearch import OpenSearchSaver

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

region = "us-east-1"

# Get AWS credentials (from environment, IAM role, or ~/.aws/credentials)
session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=1000,
    streaming=True,
)

graph = StateGraph(MessagesState)

def ask(state: MessagesState) -> MessagesState:
    """
    Ask a question to the LLM and return the response.
    """
    question = state['messages'][-1].content
    response = llm.invoke(
        [*state['messages']]
    )
    return { 'messages': [ AIMessage(content=response.content) ] }

graph.add_node('ask', ask)
graph.add_edge(START, 'ask')
graph.add_edge('ask', END)

with OpenSearchSaver.from_conn_string(client_kwargs={
    'hosts': [{'host': os.getenv('OSS_HOST'), 'port': 443}],
    'http_auth': awsauth,
    'use_ssl': True,
    'verify_certs': True,
    'connection_class': RequestsHttpConnection
}) as checkpointer:
    
    config = {
        'configurable': {
            'thread_id': '3'
        }
    }
    graph = graph.compile(checkpointer=checkpointer)
    # Run the graph with an initial message
    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content="What is the capital of France?")
            ]
        },
        config
    )
    print(response)

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content="What are its key attractions?")
            ]
        },
        config
    )
    print(response)
```
## Examples

The `examples` directory contains Jupyter notebooks demonstrating the usage of OpenSearch with LangGraph:

- `create-basic-checkpoint.ipynb`: Demonstrates the usage of OpenSearch checkpoint savers with LangGraph

## Implementation Details

### OpenSearch Indexing

The OpenSearch implementation creates these main indices:

1. **Checkpoints Index**: Stores checkpoint metadata and versioning
3. **Writes Index**: Tracks pending writes and intermediate states

## Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/samriddhadev/langgraph-opensearch.git
   cd langgraph-opensearch
   ```

2. Install dependencies:

    ```bash
    hatch env create
    ```
    or
    ```bash
    hatch shell
    ```


### Contribution Guidelines

1. Create a new branch for your changes
2. Write tests for new functionality
3. Submit a pull request with a clear description of your changes
4. Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages

## License

This project is licensed under the MIT License.
