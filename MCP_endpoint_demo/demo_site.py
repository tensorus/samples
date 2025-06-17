import asyncio
import json
import streamlit as st
from fastmcp.client import StdioTransport
from tensorus.mcp_client import TensorusMCPClient

st.set_page_config(page_title="Tensorus MCP Endpoint Demo")

# Initialize MCP client and server once per session
if 'mcp_client' not in st.session_state:
    transport = StdioTransport("python", ["-m", "tensorus.mcp_server"])
    client = TensorusMCPClient(transport)
    asyncio.run(client.__aenter__())
    st.session_state['mcp_client'] = client

client: TensorusMCPClient = st.session_state['mcp_client']

st.title("Tensorus MCP Endpoint Demo")
st.write(
    "This simple app demonstrates how to call the Model Context Protocol "
    "(MCP) server using the `TensorusMCPClient`. The server proxies to the "
    "public Tensorus API at https://tensorus-core.hf.space." )

# Utility to run async calls

def run_async(coro):
    return asyncio.run(coro)

def show_json(label, data):
    st.subheader(label)
    st.json(data)

st.header("Dataset Operations")
with st.form("create_dataset"):
    dataset_name = st.text_input("Dataset name", "demo_ds")
    submitted = st.form_submit_button("Create dataset")
    if submitted:
        res = run_async(client.create_dataset(dataset_name))
        show_json("create_dataset", res)
        st.session_state['current_dataset'] = dataset_name

if st.button("List datasets"):
    res = run_async(client.list_datasets())
    show_json("list_datasets", res)

if st.button("Delete current dataset"):
    ds = st.session_state.get('current_dataset')
    if ds:
        res = run_async(client.delete_dataset(ds))
        show_json("delete_dataset", res)
    else:
        st.write("No dataset created yet")

st.header("Tensor Operations")
current_ds = st.session_state.get('current_dataset')
if current_ds:
    if st.button("Ingest sample tensor"):
        data = [1.0, 2.0, 3.0]
        res = run_async(
            client.ingest_tensor(
                current_ds,
                tensor_shape=[3],
                tensor_dtype="float32",
                tensor_data=data,
                metadata={"demo": True},
            )
        )
        show_json("ingest_tensor", res)
        st.session_state['record_id'] = res.get('record_id')

    record_id = st.session_state.get('record_id')
    if record_id and st.button("Get tensor details"):
        res = run_async(client.get_tensor_details(current_ds, record_id))
        show_json("get_tensor_details", res)

    if record_id and st.button("Update tensor metadata"):
        res = run_async(
            client.update_tensor_metadata(
                current_ds, record_id, {"demo": True, "updated": True}
            )
        )
        show_json("update_tensor_metadata", res)

    if record_id and st.button("Delete tensor"):
        res = run_async(client.delete_tensor(current_ds, record_id))
        show_json("delete_tensor", res)
        st.session_state.pop('record_id', None)
else:
    st.info("Create a dataset first to enable tensor actions.")

st.header("Service Utilities")
if st.button("Health check"):
    res = run_async(client.management_health_check())
    show_json("health", res)

