import json

class MCPClient:
    def __init__(self, server_address):
        self.server_address = server_address
        print(f"Mock MCPClient initialized for server: {server_address}")

    def _mock_response(self, success=True, data=None, error_message=None):
        if success:
            return {"status": "success", "data": data}
        else:
            return {"status": "error", "message": error_message}

    def get_tensor(self, tensor_id):
        print(f"Mock MCPClient: get_tensor called for ID: {tensor_id}")
        if tensor_id == "existing_tensor":
            return self._mock_response(data={"id": tensor_id, "name": "Sample Tensor", "shape": "[1, 10]", "dtype": "float32", "data": "[0.1, ..., 0.9]"})
        else:
            return self._mock_response(success=False, error_message=f"Tensor with ID '{tensor_id}' not found.")

    def create_tensor(self, tensor_data):
        print(f"Mock MCPClient: create_tensor called with data: {tensor_data}")
        # Basic validation
        if not all(k in tensor_data for k in ['name', 'shape', 'dtype', 'data']):
            return self._mock_response(success=False, error_message="Missing required tensor fields: name, shape, dtype, data")
        new_id = "new_tensor_" + tensor_data.get("name", "unnamed").lower().replace(" ", "_")
        return self._mock_response(data={"id": new_id, **tensor_data, "message": "Tensor created successfully (mock)." })

    def update_tensor(self, tensor_id, update_data):
        print(f"Mock MCPClient: update_tensor called for ID '{tensor_id}' with data: {update_data}")
        if tensor_id == "existing_tensor" or tensor_id.startswith("new_tensor_"):
            return self._mock_response(data={"id": tensor_id, **update_data, "message": "Tensor updated successfully (mock)."})
        else:
            return self._mock_response(success=False, error_message=f"Tensor with ID '{tensor_id}' not found for update.")

    def delete_tensor(self, tensor_id):
        print(f"Mock MCPClient: delete_tensor called for ID: {tensor_id}")
        if tensor_id == "existing_tensor" or tensor_id.startswith("new_tensor_"):
            return self._mock_response(data={"id": tensor_id, "message": "Tensor deleted successfully (mock)."})
        else:
            return self._mock_response(success=False, error_message=f"Tensor with ID '{tensor_id}' not found for deletion.")

    def get_model_info(self, model_id):
        print(f"Mock MCPClient: get_model_info called for ID: {model_id}")
        if model_id == "sample_model":
            return self._mock_response(data={"id": model_id, "name": "Sample Model", "version": "1.0", "input_schema": {}, "output_schema": {}})
        else:
            return self._mock_response(success=False, error_message=f"Model with ID '{model_id}' not found.")

    def predict(self, model_id, input_data):
        print(f"Mock MCPClient: predict called for model ID '{model_id}' with input: {input_data}")
        if model_id == "sample_model":
            # Simulate prediction based on input_data structure (very basic)
            prediction = {"mock_prediction": "some_value"}
            if isinstance(input_data, dict) and "features" in input_data:
                prediction["received_features_count"] = len(input_data["features"])
            return self._mock_response(data={"model_id": model_id, "prediction": prediction})
        else:
            return self._mock_response(success=False, error_message=f"Model with ID '{model_id}' not found for prediction.")

    def get_server_status(self):
        print(f"Mock MCPClient: get_server_status called")
        return self._mock_response(data={"server_version": "0.1.0-mock", "status": "OPERATIONAL", "active_models": ["sample_model"]})

    def list_tensors(self, limit=10, offset=0):
        print(f"Mock MCPClient: list_tensors called with limit={limit}, offset={offset}")
        mock_tensor_list = [
            {"id": "tensor_abc", "name": "Tensor Alpha", "shape": "[10, 20]", "dtype": "float32"},
            {"id": "tensor_def", "name": "Tensor Beta", "shape": "[5, 5, 5]", "dtype": "int16"},
            {"id": "existing_tensor", "name": "Sample Tensor", "shape": "[1, 10]", "dtype": "float32"}
        ]
        return self._mock_response(data={"tensors": mock_tensor_list[offset:offset+limit], "total_count": len(mock_tensor_list)})

    def list_models(self, limit=10, offset=0):
        print(f"Mock MCPClient: list_models called with limit={limit}, offset={offset}")
        mock_model_list = [
            {"id": "sample_model", "name": "Sample Model", "version": "1.0"},
            {"id": "another_model", "name": "Another Great Model", "version": "2.1"}
        ]
        return self._mock_response(data={"models": mock_model_list[offset:offset+limit], "total_count": len(mock_model_list)})

if __name__ == '__main__':
    # Example Usage (for testing the mock client itself)
    client = MCPClient(server_address="http://localhost:8080") # Address doesn't matter for mock

    print("\n--- Testing Tensor Operations ---")
    print("Get Tensor (existing):", json.dumps(client.get_tensor("existing_tensor"), indent=2))
    print("Get Tensor (non-existing):", json.dumps(client.get_tensor("non_existent_tensor"), indent=2))
    print("Create Tensor:", json.dumps(client.create_tensor({"name": "My New Tensor", "shape": "[1,2,3]", "dtype": "int8", "data": "[1,2,3]"}), indent=2))
    print("Update Tensor:", json.dumps(client.update_tensor("existing_tensor", {"metadata": {"source": "updated"}}), indent=2))
    print("Delete Tensor:", json.dumps(client.delete_tensor("existing_tensor"), indent=2))
    print("List Tensors:", json.dumps(client.list_tensors(), indent=2))


    print("\n--- Testing Model Operations ---")
    print("Get Model Info (existing):", json.dumps(client.get_model_info("sample_model"), indent=2))
    print("Get Model Info (non-existing):", json.dumps(client.get_model_info("unknown_model"), indent=2))
    print("Predict (existing model):", json.dumps(client.predict("sample_model", {"features": [0.1, 0.2, 0.3]}), indent=2))
    print("List Models:", json.dumps(client.list_models(), indent=2))

    print("\n--- Testing Server Operations ---")
    print("Get Server Status:", json.dumps(client.get_server_status(), indent=2))
