# Tensorus MCP Endpoint Demo

This demo provides a very small Streamlit application that showcases how the
`TensorusMCPClient` communicates with the official Tensorus MCP server.
The server component is started automatically via `StdioTransport` and proxies
requests to the public API at [tensorus-core.hf.space](https://tensorus-core.hf.space).

The demo lets you:

* Create and list datasets
* Ingest a small example tensor
* Retrieve and update tensor metadata
* Delete tensors or entire datasets
* Perform a simple health check on the service

## Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Demo

Launch the Streamlit app (it automatically starts a local MCP server):

```bash
streamlit run MCP_endpoint_demo/app.py
```

The page will display controls for the most common MCP endpoints. Each action
uses the `TensorusMCPClient` under the hood to call the corresponding tool
exposed by the MCP server.

For full API documentation see
[https://tensorus-core.hf.space/docs](https://tensorus-core.hf.space/docs)
and [https://tensorus-core.hf.space/redoc](https://tensorus-core.hf.space/redoc).
