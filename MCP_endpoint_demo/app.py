import asyncio
import json
from typing import Any, Dict, List

import streamlit as st
from fastmcp.client import Client, StdioTransport


# Utility to run a single MCP tool call
async def _call(tool: str, arguments: Dict[str, Any] | None = None) -> Any:
    transport = StdioTransport("python", ["-m", "tensorus.mcp_server"])
    async with Client(transport) as client:
        result = await client.call_tool(tool, arguments or {})
        if not result:
            return None
        return json.loads(result[0].text)


def run(tool: str, args: Dict[str, Any] | None = None) -> Any:
    return asyncio.run(_call(tool, args))


st.title("Tensorus MCP Endpoint Demo")

st.markdown(
    "This demo spawns the Tensorus MCP server on demand and communicates via"
    " the `fastmcp` client. Use the forms below to invoke common endpoints and"
    " inspect their JSON responses."
)

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
        result = run("tensorus_list_datasets")

elif operation == "create_dataset":
    name = st.text_input("Dataset name")
    if st.button("Create") and name:
        result = run("tensorus_create_dataset", {"dataset_name": name})

elif operation == "delete_dataset":
    name = st.text_input("Dataset name")
    if st.button("Delete") and name:
        result = run("tensorus_delete_dataset", {"dataset_name": name})

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
        result = run(
            "tensorus_ingest_tensor",
            {
                "dataset_name": dataset,
                "tensor_shape": [len(numbers)],
                "tensor_dtype": "float32",
                "tensor_data": numbers,
                "metadata": metadata,
            },
        )

elif operation == "get_tensor_details":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    if st.button("Fetch") and dataset and record_id:
        result = run(
            "tensorus_get_tensor_details",
            {"dataset_name": dataset, "record_id": record_id},
        )

elif operation == "update_tensor_metadata":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    metadata_str = st.text_area("New metadata JSON", "{}")
    if st.button("Update") and dataset and record_id:
        metadata = json.loads(metadata_str or "{}")
        result = run(
            "tensorus_update_tensor_metadata",
            {
                "dataset_name": dataset,
                "record_id": record_id,
                "new_metadata": metadata,
            },
        )

elif operation == "delete_tensor":
    dataset = st.text_input("Dataset name")
    record_id = st.text_input("Record ID")
    if st.button("Delete") and dataset and record_id:
        result = run(
            "tensorus_delete_tensor",
            {"dataset_name": dataset, "record_id": record_id},
        )

if result is not None:
    st.subheader("Result")
    st.json(result)
