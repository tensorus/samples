import asyncio
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from fastmcp.client import Client as FastMCPClient, StdioTransport
try:
    from fastmcp.tools import TextContent
except Exception:  # pragma: no cover - minimal fallback
    @dataclass
    class TextContent:  # type: ignore
        type: str
        text: str


class TensorusMCPClient:
    """High level client for the Tensorus MCP server."""

    def __init__(self, transport: Any) -> None:
        self._client = FastMCPClient(transport)

    async def __aenter__(self) -> "TensorusMCPClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    async def _call_json(self, name: str, arguments: Optional[dict] = None) -> Any:
        result = await self._client.call_tool(name, arguments or {})
        if not result:
            return None
        content = result[0]
        if isinstance(content, TextContent):
            return json.loads(content.text)
        raise TypeError("Unexpected content type")

    async def list_datasets(self) -> Any:
        return await self._call_json("tensorus_list_datasets")

    async def create_dataset(self, dataset_name: str) -> Any:
        return await self._call_json("tensorus_create_dataset", {"dataset_name": dataset_name})

    async def ingest_tensor(
        self,
        dataset_name: str,
        tensor_shape: Sequence[int],
        tensor_dtype: str,
        tensor_data: Any,
        metadata: Optional[dict] = None,
    ) -> Any:
        payload = {
            "dataset_name": dataset_name,
            "tensor_shape": list(tensor_shape),
            "tensor_dtype": tensor_dtype,
            "tensor_data": tensor_data,
            "metadata": metadata,
        }
        return await self._call_json("tensorus_ingest_tensor", payload)

    async def get_tensor_details(self, dataset_name: str, record_id: str) -> Any:
        return await self._call_json(
            "tensorus_get_tensor_details",
            {"dataset_name": dataset_name, "record_id": record_id},
        )


async def main() -> None:
    server_path = os.path.join(os.path.dirname(__file__), "mcp_time_series_server.py")
    transport = StdioTransport("python", [server_path])
    async with TensorusMCPClient(transport) as client:
        await client.create_dataset("timeseries_demo")
        x = np.linspace(0, 2 * math.pi, 50)
        data = np.sin(x).tolist()
        result = await client.ingest_tensor(
            "timeseries_demo",
            tensor_shape=[50],
            tensor_dtype="float32",
            tensor_data=data,
            metadata={"source": "synthetic_sine"},
        )
        record_id = result.get("record_id")
        print("Inserted record", record_id)
        details = await client.get_tensor_details("timeseries_demo", record_id)
        print("Fetched tensor:", details)
        datasets = await client.list_datasets()
        print("Available datasets:", datasets)


if __name__ == "__main__":
    asyncio.run(main())
