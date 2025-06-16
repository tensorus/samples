# Tensorus MCP Time Series Demo

This small example demonstrates how to use the Model Context Protocol (MCP) server and client from [Tensorus](https://github.com/tensorus/tensorus) to store and retrieve simple time series data.

The demo uses a minimal FastAPI backend with an in-memory tensor store. The MCP server exposes this backend as standard MCP tools and the client communicates with it over a stdio transport.

## Prerequisites

* Python 3.8+
* Install dependencies:
  ```bash
  pip install -r requirements.txt fastmcp uvicorn numpy tensorus
  ```

## Running the Demo

1. **Start the server** (this runs both the API and MCP server):
   ```bash
   python mcp_time_series_server.py
   ```
   Leave this running in a terminal.

2. **Run the client demo** in another terminal:
   ```bash
   python mcp_time_series_demo.py
   ```

The client creates a dataset, inserts a synthetic sine wave, updates its metadata and finally cleans up the stored tensor and dataset. This illustrates how Tensorus capabilities can be accessed through the standardized MCP interface.
