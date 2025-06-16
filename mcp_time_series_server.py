import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import httpx
from fastapi import FastAPI, HTTPException
import uvicorn
from fastmcp import FastMCP
try:
    from fastmcp.tools import TextContent
except ImportError:  # pragma: no cover - support older fastmcp versions
    @dataclass
    class TextContent:
        type: str
        text: str

# --- Minimal In-Memory Tensor Storage ---
class EmbeddedTensorStorage:
    def __init__(self) -> None:
        self.datasets: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def create_dataset(self, name: str) -> None:
        if name in self.datasets:
            raise ValueError(f"Dataset {name} already exists")
        self.datasets[name] = {}

    def insert(
        self,
        name: str,
        tensor: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} does not exist")
        record_id = str(uuid.uuid4())
        self.datasets[name][record_id] = {
            "tensor": list(tensor),
            "metadata": metadata or {},
        }
        return record_id

    def get(self, name: str, record_id: str) -> Dict[str, Any]:
        if name not in self.datasets or record_id not in self.datasets[name]:
            raise KeyError("record not found")
        return self.datasets[name][record_id]

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def delete_dataset(self, name: str) -> None:
        if name not in self.datasets:
            raise KeyError("dataset not found")
        del self.datasets[name]

    def delete_tensor(self, name: str, record_id: str) -> None:
        if name not in self.datasets or record_id not in self.datasets[name]:
            raise KeyError("record not found")
        del self.datasets[name][record_id]

    def update_tensor_metadata(
        self, name: str, record_id: str, new_metadata: Dict[str, Any]
    ) -> None:
        if name not in self.datasets or record_id not in self.datasets[name]:
            raise KeyError("record not found")
        self.datasets[name][record_id]["metadata"] = new_metadata


storage = EmbeddedTensorStorage()
app = FastAPI(title="Tensorus Demo API")


@app.post("/datasets/create")
async def create_dataset(payload: Dict[str, str]):
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    try:
        storage.create_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "created"}


@app.get("/datasets")
async def list_datasets():
    return {"data": storage.list_datasets()}


@app.post("/datasets/{dataset_name}/ingest")
async def ingest_tensor(dataset_name: str, payload: Dict[str, Any]):
    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="data must be list")
    record_id = storage.insert(dataset_name, data, payload.get("metadata"))
    return {"record_id": record_id}


@app.get("/datasets/{dataset_name}/tensors/{record_id}")
async def get_tensor(dataset_name: str, record_id: str):
    try:
        return storage.get(dataset_name, record_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="not found")


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    try:
        storage.delete_dataset(dataset_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="not found")
    return {"status": "deleted"}


@app.delete("/datasets/{dataset_name}/tensors/{record_id}")
async def delete_tensor(dataset_name: str, record_id: str):
    try:
        storage.delete_tensor(dataset_name, record_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="not found")
    return {"status": "deleted"}


@app.put("/datasets/{dataset_name}/tensors/{record_id}/metadata")
async def update_tensor_metadata(dataset_name: str, record_id: str, payload: Dict[str, Any]):
    try:
        storage.update_tensor_metadata(dataset_name, record_id, payload.get("new_metadata", {}))
    except KeyError:
        raise HTTPException(status_code=404, detail="not found")
    return {"status": "updated"}


# --- MCP Server (adapted from tensorus.mcp_server) ---
API_BASE_URL = "http://127.0.0.1:8000"
server = FastMCP(name="Tensorus FastMCP")


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


async def _get(path: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _delete(path: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{API_BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _put(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{API_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


@server.tool(name="tensorus_ingest_tensor")
async def tensorus_ingest_tensor(
    dataset_name: str,
    tensor_shape: Sequence[int],
    tensor_dtype: str,
    tensor_data: Any,
    metadata: Optional[dict] = None,
) -> TextContent:
    payload = {
        "shape": list(tensor_shape),
        "dtype": tensor_dtype,
        "data": tensor_data,
        "metadata": metadata,
    }
    result = await _post(f"/datasets/{dataset_name}/ingest", payload)
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def get_tensor(dataset_name: str, record_id: str) -> TextContent:
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool()
async def execute_nql_query(query: str) -> TextContent:
    result = await _post("/query", {"query": query})
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_list_datasets")
async def tensorus_list_datasets() -> TextContent:
    result = await _get("/datasets")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_create_dataset")
async def tensorus_create_dataset(dataset_name: str) -> TextContent:
    result = await _post("/datasets/create", {"name": dataset_name})
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_get_tensor_details")
async def tensorus_get_tensor_details(dataset_name: str, record_id: str) -> TextContent:
    result = await _get(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_delete_dataset")
async def tensorus_delete_dataset(dataset_name: str) -> TextContent:
    result = await _delete(f"/datasets/{dataset_name}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_delete_tensor")
async def tensorus_delete_tensor(dataset_name: str, record_id: str) -> TextContent:
    result = await _delete(f"/datasets/{dataset_name}/tensors/{record_id}")
    return TextContent(type="text", text=json.dumps(result))


@server.tool(name="tensorus_update_tensor_metadata")
async def tensorus_update_tensor_metadata(
    dataset_name: str, record_id: str, new_metadata: dict
) -> TextContent:
    payload = {"new_metadata": new_metadata}
    result = await _put(
        f"/datasets/{dataset_name}/tensors/{record_id}/metadata",
        payload,
    )
    return TextContent(type="text", text=json.dumps(result))


async def run_servers() -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    api_server = uvicorn.Server(config)
    api_task = asyncio.create_task(api_server.serve())
    mcp_task = asyncio.create_task(server.run_stdio_async())
    await asyncio.gather(api_task, mcp_task)


def main() -> None:
    asyncio.run(run_servers())


if __name__ == "__main__":
    main()
