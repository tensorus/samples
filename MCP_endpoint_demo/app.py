from flask import Flask, render_template, request, jsonify
from mcp_client import MCPClient # Assuming mcp_client.py is in the same directory
import json # For pretty printing JSON in template

app = Flask(__name__)

# Initialize the MCP Client
# For a real scenario, this address would come from config or environment variable
mcp_server_address = "http://mock-mcp-server:8080" # Actual address doesn't matter for the mock client
client = MCPClient(server_address=mcp_server_address)

@app.route('/')
def index():
    # Fetch server status to display on the homepage
    server_status_resp = client.get_server_status()
    return render_template('index.html', title='MCP Endpoint Demo', server_status=server_status_resp.get('data', {}), mcp_client_error=server_status_resp.get('status') == 'error')

# --- Tensor Related Routes ---
@app.route('/tensor/get', methods=['POST'])
def get_tensor_route():
    tensor_id = request.form.get('tensor_id')
    if not tensor_id:
        return jsonify({"status": "error", "message": "Tensor ID is required."}), 400
    response = client.get_tensor(tensor_id)
    return jsonify(response)

@app.route('/tensor/list', methods=['GET'])
def list_tensors_route():
    # Simple listing, could add pagination params later
    response = client.list_tensors()
    return jsonify(response)

@app.route('/tensor/create', methods=['POST'])
def create_tensor_route():
    try:
        # Assuming form data for simplicity, could also be JSON
        name = request.form.get('name')
        shape = request.form.get('shape')
        dtype = request.form.get('dtype')
        data_str = request.form.get('data') # Will be a string, client might parse if necessary

        if not all([name, shape, dtype, data_str]):
            return jsonify({"status": "error", "message": "Missing one or more fields: name, shape, dtype, data"}), 400

        tensor_data = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "data": data_str # The mock client expects data, actual client might handle complex structures
        }
        response = client.create_tensor(tensor_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Model Related Routes (Placeholders for now) ---
@app.route('/model/get', methods=['POST'])
def get_model_info_route():
    model_id = request.form.get('model_id')
    if not model_id:
        return jsonify({"status": "error", "message": "Model ID is required."}), 400
    response = client.get_model_info(model_id)
    return jsonify(response)

@app.route('/model/list', methods=['GET'])
def list_models_route():
    response = client.list_models()
    return jsonify(response)

@app.route('/model/predict', methods=['POST'])
def model_predict_route():
    try:
        model_id = request.form.get('model_id')
        # For simplicity, taking raw input_data string.
        # A real app might parse this as JSON or other structured format.
        input_data_str = request.form.get('input_data')

        if not model_id or input_data_str is None: # Check for None as empty string could be valid input
            return jsonify({"status": "error", "message": "Model ID and Input Data are required."}), 400

        # The mock client expects a dict, let's try to parse if it's JSON, otherwise pass as is.
        try:
            parsed_input_data = json.loads(input_data_str)
        except json.JSONDecodeError:
            # If not JSON, pass the string as part of a dict (adjust based on actual client needs)
            # This is a common pattern for simple key-value inputs.
            parsed_input_data = {"raw_input": input_data_str}


        response = client.predict(model_id, parsed_input_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/tensor/update', methods=['POST'])
def update_tensor_route():
    try:
        tensor_id = request.form.get('tensor_id')
        # For simplicity, taking update_data as a JSON string from a textarea
        update_data_str = request.form.get('update_data')

        if not tensor_id or not update_data_str:
            return jsonify({"status": "error", "message": "Tensor ID and Update Data (JSON string) are required."}), 400

        try:
            update_data = json.loads(update_data_str)
        except json.JSONDecodeError:
            return jsonify({"status": "error", "message": "Update Data is not valid JSON."}), 400

        response = client.update_tensor(tensor_id, update_data)
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/tensor/delete', methods=['POST'])
def delete_tensor_route():
    tensor_id = request.form.get('tensor_id')
    if not tensor_id:
        return jsonify({"status": "error", "message": "Tensor ID is required."}), 400
    response = client.delete_tensor(tensor_id)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
