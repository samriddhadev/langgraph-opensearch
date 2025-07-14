import os
import time
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

@pytest.fixture(scope="session", autouse=True)
def wait_for_opensearch():
    """
    Wait for OpenSearch to be reachable before running tests.
    Only enforced in CI environment (CI=true).
    Uses exponential backoff up to ~4 minutes.
    """
    is_ci = os.environ.get("CI", "false").lower() == "true"

    if not is_ci:
        print("🧪 Skipping OpenSearch availability check (not in CI).")
        return

    host = os.environ.get("OSS_HOST", "localhost")
    port = os.environ.get("OSS_PORT", "9200")
    user = os.environ.get("OSS_USER", "admin")
    password = os.environ.get("OSS_PASS", "admin")

    scheme = "https"
    base_url = f"{scheme}://{host}:{port}"
    health_url = urljoin(base_url, "/")

    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                health_url,
                auth=HTTPBasicAuth(user, password),
                timeout=5,
                verify=False,
            )
            if response.ok:
                print(f"OpenSearch is available after {attempt} attempt(s)")
                return
            else:
                print(f"OpenSearch responded with {response.status_code}, retrying...")
        except Exception as e:
            print(f"Attempt {attempt}/{max_attempts} failed: {e}")

        sleep_time = min(2 ** attempt, 60)
        print(f"Waiting {sleep_time} seconds before retrying...")
        time.sleep(sleep_time)

    pytest.exit("OpenSearch did not respond after maximum retry attempts.")

@pytest.fixture(scope="session")
def client_kwargs():
    """
    Returns OpenSearch client kwargs based on environment.
    """
    host = os.environ.get("OSS_HOST", "localhost")
    port = int(os.environ.get("OSS_PORT", "9200"))

    if os.environ.get("CI", "false").lower() == "true":
        # CI or build system
        return {
            "hosts": [{"host": host, "port": port}],
            "use_ssl": True,
            "http_auth": (os.environ.get("OSS_USER", "admin"), os.environ.get("OSS_PASS", "admin")),
            "verify_certs": False,
            "timeout": 30,
            "max_retries": 3,
        }
    else:
        # Local development
        from opensearchpy import RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        import boto3
        region = os.getenv('AWS_REGION', 'us-east-1')   # Default region if not set
        session = boto3.Session()
        credentials = session.get_credentials()
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
        return {
            'hosts': [{'host': os.getenv('OSS_HOST'), 'port': 443}],
            'http_auth': awsauth,
            'use_ssl': True,
            'verify_certs': True,
            'connection_class': RequestsHttpConnection
        }


@pytest.fixture(scope="session")
def input_data() -> dict:
    """Setup and store conveniently in a single dictionary."""
    inputs: dict[str, Any] = {}

    inputs["config_1"] = RunnableConfig(
        configurable=dict(thread_id="thread-1",
                          thread_ts="1", checkpoint_ns="")
    )  # config_1 tests deprecated thread_ts

    inputs["config_2"] = RunnableConfig(
        configurable=dict(thread_id="thread-2",
                          checkpoint_id="2", checkpoint_ns="")
    )

    inputs["config_3"] = RunnableConfig(
        configurable=dict(
            thread_id="thread-2", checkpoint_id="2-inner", checkpoint_ns="inner"
        )
    )

    inputs["chkpnt_1"] = empty_checkpoint()
    inputs["chkpnt_2"] = create_checkpoint(inputs["chkpnt_1"], {}, 1)
    inputs["chkpnt_3"] = empty_checkpoint()

    inputs["metadata_1"] = CheckpointMetadata(
        source="input", step=2, writes={}, score=1
    )
    inputs["metadata_2"] = CheckpointMetadata(
        source="loop", step=1, writes={"foo": "bar"}, score=None
    )
    inputs["metadata_3"] = CheckpointMetadata()

    return inputs
