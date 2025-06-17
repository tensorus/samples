import asyncio
import json
from typing import Any, List

import streamlit as st
from fastmcp.client import SSETransport
from tensorus.mcp_client import TensorusMCPClient


# Utility to run a single MCP client call using SSE transport
async def _call(server_url: str, method: str, *args: Any, **kwargs: Any) -> Any:
    transport = SSETransport(f"{server_url.rstrip('/')}/mcp")
    async with TensorusMCPClient(transport) as client:
        func = getattr(client, method)
        return await func(*args, **kwargs)


def run(server_url: str, method: str, *args: Any, **kwargs: Any) -> Any:
    try:
        return asyncio.run(_call(server_url, method, *args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


st.title("Tensorus MCP Endpoint Demo")

st.markdown(
    "This demo uses the `TensorusMCPClient` to communicate with a running MCP "
    "server. Start the server with `python -m tensorus.mcp_server --transport sse`"
    " and specify its base URL below. Use the forms to invoke common endpoints "
    "and inspect the JSON responses."
)

server_url = st.sidebar.text_input("MCP server URL", "http://127.0.0.1:8000")

operation = st.sidebar.selectbox(
    "Choose an operation",
    (
        "list_datasets",
        "create_dataset",
        "delete_dataset",
        "ingest_tensor",
        "get_tensor_details",
        "update_tensor_metadata",
        "delete_tensor",
    ),
)

result: Any = None

if operation == "list_datasets":
    if st.button("List datasets"):
        with st.spinner("Calling list_datasets..."):
            result = run(server_url, "list_datasets")

elif operation == "create_dataset":
    name = st.text_input("Dataset name")
    if st.button("Create") and name:
        with st.spinner("Creating dataset..."):
            result = run(server_url, "create_dataset", name)

elif operation == "delete_dataset":
    name = st.text_input("Dataset name")
    if st.button("Delete") and name:
        with st.spinner("Deleting dataset..."):
            result = run(server_url, "delete_dataset", name)

elif operation == "ingest_tensor":
    dataset = st.text_input("Dataset name")
    data_str = st.text_area(
        "Tensor data (comma separated numbers)",
        "0.0, 1.0, 2.0",
    )
    metadata_str = st.text_area("Metadata JSON", "{}")
    if st.button("Ingest") and dataset and data_str:
        numbers: List[float] = [float(x) for x in data_str.split(",") if x.strip()]
        metadata = json.loads(metadata_str or "{}")
        with st.spinner("Ingesting tensor..."):
            result = run(
                server_url,
                "ingest_tensor",
                dataset,
                [len(numbers)],
                "float32",
                numbers,
                metadata,
            )

elif operation == "get_tensor_details":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    if st.button("Fetch") and dataset and record_id:
        with st.spinner("Fetching tensor..."):
            result = run(server_url, "get_tensor_details", dataset, record_id)

elif operation == "update_tensor_metadata":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    metadata_str = st.text_area("New metadata JSON", "{}")
    if st.button("Update") and dataset and record_id:
        metadata = json.loads(metadata_str or "{}")
        with st.spinner("Updating metadata..."):
            result = run(
                server_url, "update_tensor_metadata", dataset, record_id, metadata
            )

elif operation == "delete_tensor":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    if st.button("Delete") and dataset and record_id:
        with st.spinner("Deleting tensor..."):
            result = run(server_url, "delete_tensor", dataset, record_id)

if result is not None:
    st.subheader("Result")
    st.json(result)
