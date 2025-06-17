# Tensorus MCP Endpoint Demo

This demo provides a minimal Streamlit application for exploring the Tensorus **Model Context Protocol** (MCP) server. It illustrates how a client can interact with the server's dataset and tensor management endpoints.

## Prerequisites

- Python 3.8+
- Install the repository requirements:
  ```bash
  pip install -r requirements.txt
  ```

## Running the Demo

1. **Start the MCP server** in a separate terminal. The server proxies requests to the public Tensorus API:
   ```bash
   python -m tensorus.mcp_server
   ```

2. **Launch the Streamlit UI** for this demo:
   ```bash
   streamlit run MCP_endpoint_demo/app.py
   ```

The UI offers simple forms to call several MCP endpoints and shows the JSON responses returned by the server.

## Supported Operations

The app demonstrates these MCP client methods:

- `list_datasets`
- `create_dataset`
- `delete_dataset`
- `ingest_tensor`
- `get_tensor_details`
- `update_tensor_metadata`
- `delete_tensor`

Refer to the [Tensorus API documentation](https://tensorus-core.hf.space/docs) for full details on each endpoint.
