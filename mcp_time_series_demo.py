import asyncio
import math
import os

import numpy as np
from fastmcp.client import StdioTransport
from tensorus.mcp_client import TensorusMCPClient


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
        await client.update_tensor_metadata(
            "timeseries_demo",
            record_id,
            {"source": "synthetic_sine", "processed": True},
        )
        updated = await client.get_tensor_details("timeseries_demo", record_id)
        print("Updated metadata:", updated)
        datasets = await client.list_datasets()
        print("Available datasets:", datasets)
        await client.delete_tensor("timeseries_demo", record_id)
        await client.delete_dataset("timeseries_demo")
        print("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
