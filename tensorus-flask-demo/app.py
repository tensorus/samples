from flask import Flask, render_template, request, jsonify
import json
import time # For mock log timestamps

app = Flask(__name__)

mock_schemas = {
    "image_schema": {
        "required_metadata": ["source_url", "resolution"],
        "expected_shape_prefix": "[1, 224, 224,",
        "allowed_dtypes": ["float32", "uint8"]
    },
    "audio_schema": {
        "required_metadata": ["sample_rate", "duration_seconds"],
        "expected_shape_prefix": "[1, ",
        "allowed_dtypes": ["float32", "int16"]
    }
}

mock_tensors = [
    {'id': 1, 'name': 'InitialTensor', 'shape': '[3, 4]', 'dtype': 'float32', 'data': '[[1,2,3,4],[5,6,7,8],[9,10,11,12]]', 'metadata': {'source': 'manual'}},
    {'id': 2, 'name': 'AnotherTensor', 'shape': '[2, 2]', 'dtype': 'int16', 'data': '[[10,20],[30,40]]', 'metadata': {'source': 'generated'}}
]
next_tensor_id = 3

mock_models_data = [
    {"id": "xgboost_regressor", "name": "XGBoost Regressor", "category": "Classical ML - Regression", "description": "A powerful gradient boosting algorithm for regression tasks.", "example_input": "Features: {'feature1': 0.5, 'feature2': 1.2, 'feature3': -0.8}", "example_output": "Predicted Value: 15.7", "doc_link": "https://xgboost.readthedocs.io/"},
    {"id": "arima_model", "name": "ARIMA Model", "category": "Time Series Analysis", "description": "Autoregressive Integrated Moving Average model for time series forecasting.", "example_input": "Time Series Data: [10, 12, 11, 13, 14, 13], Forecast Steps: 3", "example_output": "Forecast: [13.5, 13.8, 14.1]", "doc_link": "https://www.statsmodels.org/stable/tsa.html"},
    {"id": "transformer_model_nlp", "name": "Transformer Model (NLP)", "category": "Natural Language Processing", "description": "A sequence-to-sequence model using attention mechanisms, foundational for many NLP tasks.", "example_input": "Input Text: 'Hello world, how are you?'", "example_output": "Processed/Translated Text: '[Mock Translation/Summary]'", "doc_link": "https://huggingface.co/docs/transformers/index"},
    {"id": "dqn_model_rl", "name": "DQN Model (Reinforcement Learning)", "category": "Reinforcement Learning", "description": "Deep Q-Network for learning policies in environments with discrete action spaces.", "example_input": "Environment State: {'player_pos': [10,20], 'enemy_pos': [50,60]}", "example_output": "Selected Action: 'MoveRight'", "doc_link": "#"},
    {"id": "alexnet_cnn", "name": "AlexNet (Image Classification)", "category": "Computer Vision - Classification", "description": "A pioneering Convolutional Neural Network for image classification.", "example_input": "Image: (Conceptually, an image file or features)", "example_output": "Predicted Class: 'Cat' (Confidence: 0.85)", "doc_link": "#"}
]

mock_datasets_data = [
    {"id": "cifar10_subset", "name": "CIFAR-10 Subset (Images)", "category": "Image Data", "source": "Real-world (Subset)", "description": "A smaller version of the CIFAR-10 dataset...", "properties": {"Number of Images": "1,000 (mock)", "Classes": "10", "Image Size": "32x32 pixels", "Channels": "3 (RGB)"}, "example_data_description": "Typically image files or tensors."},
    {"id": "synthetic_iot_sensors", "name": "Synthetic IoT Sensor Data", "category": "Time Series Data", "source": "Synthetic", "description": "Generated time series data mimicking IoT sensors...", "properties": {"Number of Sensors": "5 (mock)", "Data Points per Sensor": "10,000 (mock)", "Frequency": "1 reading/minute"}, "example_data_description": "[{'ts': '...', 'temp': 22.5}, ...]"},
    {"id": "synthetic_customer_profiles", "name": "Synthetic Customer Profiles", "category": "Tabular Data", "source": "Synthetic", "description": "Generated tabular data for customer profiles...", "properties": {"Number of Profiles": "500 (mock)", "Features": "8 (mock)"}, "example_data_description": "{'id': 'CUST001', 'age': 34, ...}"}
]

mock_agents_data = [
    {
        "id": "ingestion_agent_images",
        "name": "Image Ingestion Agent",
        "type": "Data Ingestion",
        "status": "Running",
        "description": "Monitors a directory for new image files, preprocesses them, and ingests them as tensors into TensorStorage.",
        "config": {
            "source_directory": "/mnt/raw_images",
            "polling_interval_sec": 60,
            "target_dataset": "raw_image_tensors"
        },
        "mock_logs": [
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Agent started. Monitoring /mnt/raw_images.",
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Found 5 new images. Starting ingestion...",
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Successfully ingested image_001.png to 'raw_image_tensors'.",
        ]
    },
    {
        "id": "rl_agent_trading",
        "name": "RL Trading Agent",
        "type": "Reinforcement Learning",
        "status": "Idle",
        "description": "A Deep Q-Network (DQN) agent learning a mock trading strategy.",
        "config": {
            "environment": "MockStockMarketEnv",
            "episodes_to_train": 1000,
            "learning_rate": 0.001
        },
        "mock_logs": [
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Agent initialized. Awaiting start command.",
        ]
    },
    {
        "id": "automl_agent_classification",
        "name": "AutoML Classification Agent",
        "type": "AutoML",
        "status": "Idle",
        "description": "Performs hyperparameter optimization for a classification model on a specified dataset.",
        "config": {
            "dataset_name": "customer_churn_data",
            "model_type": "XGBoostClassifier",
            "search_trials": 50
        },
        "mock_logs": [
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Agent initialized. Ready for hyperparameter search.",
        ]
    }
]


@app.route('/')
def home():
    return render_template('index.html', title='Home')

@app.route('/core-features')
def core_features_page():
    return render_template('core_features.html', title='Core Features')

@app.route('/models')
def models_page():
    return render_template('models.html', title='Model Showcase')

@app.route('/datasets')
def datasets_page():
    return render_template('datasets.html', title='Dataset Showcase')

@app.route('/agents')
def agents_page():
    return render_template('agents.html', title='Agent Dashboard')

# ... (existing API endpoints for models, tensors, nql_query) ...
@app.route('/api/models', methods=['GET'])
def get_models_data():
    return jsonify(mock_models_data)

@app.route('/api/models/<string:model_id>/mock_predict', methods=['POST'])
def mock_model_predict(model_id):
    model_info = next((m for m in mock_models_data if m['id'] == model_id), None)
    if not model_info:
        return jsonify({'error': 'Model not found'}), 404
    try:
        input_data = request.get_json()
        if input_data and input_data.get('mock_input'):
            return jsonify({'prediction': f"{model_info.get('example_output', 'No example output.')} (Input length: {len(str(input_data.get('mock_input')))})"})
        return jsonify({'prediction': model_info.get('example_output', 'No example output defined.')})
    except Exception as e:
        print(f"Error in mock_model_predict: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/datasets', methods=['GET'])
def get_datasets_data():
    return jsonify(mock_datasets_data)

@app.route('/api/tensors', methods=['GET', 'POST'])
def manage_tensors():
    global next_tensor_id, mock_tensors
    if request.method == 'GET':
        return jsonify(mock_tensors)
    elif request.method == 'POST':
        try:
            new_tensor_data = request.get_json()
            if not new_tensor_data or not all(k in new_tensor_data for k in ['name', 'shape', 'dtype', 'data']):
                return jsonify({'error': 'Missing required tensor fields: name, shape, dtype, data'}), 400
            schema_name = new_tensor_data.get('schema_name')
            if schema_name and schema_name in mock_schemas:
                schema = mock_schemas[schema_name]
                current_metadata = new_tensor_data.get('metadata', {})
                for req_meta_key in schema.get('required_metadata', []):
                    if req_meta_key not in current_metadata:
                        return jsonify({'error': f'Schema validation failed: Missing required metadata field: {req_meta_key}'}), 400
                expected_prefix = schema.get('expected_shape_prefix', '')
                if expected_prefix and not new_tensor_data.get('shape', '').startswith(expected_prefix):
                    return jsonify({'error': f'Schema validation failed: Shape must start with {expected_prefix}'}), 400
                allowed_dtypes = schema.get('allowed_dtypes', [])
                if allowed_dtypes and new_tensor_data.get('dtype') not in allowed_dtypes:
                    return jsonify({'error': f'Schema validation failed: DType must be one of {allowed_dtypes}'}), 400
            new_tensor = {
                'id': next_tensor_id, 'name': new_tensor_data['name'], 'shape': new_tensor_data['shape'],
                'dtype': new_tensor_data['dtype'], 'data': new_tensor_data['data'],
                'metadata': new_tensor_data.get('metadata', {}), 'schema_name': schema_name if schema_name else None
            }
            mock_tensors.append(new_tensor)
            next_tensor_id += 1
            return jsonify(new_tensor), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/api/nql_query', methods=['POST'])
def handle_nql_query():
    global mock_tensors
    try:
        query_data = request.get_json(); query_string = query_data.get('query_string', '').lower()
        if not query_string: return jsonify({'error': 'Empty query string'}), 400
        results = []; parts = query_string.split(" with ")
        if len(parts) < 2 and query_string != "find all tensors":
            if query_string == "find all tensors": return jsonify(mock_tensors)
            return jsonify({'error': 'Unsupported query format. Use "find tensors with ..." or "find all tensors"'}), 400
        elif query_string == "find all tensors": return jsonify(mock_tensors)
        conditions_part = parts[1]
        if "name containing " in conditions_part:
            term = conditions_part.split("name containing ")[1].strip().replace("'", "").replace('"', '')
            results = [t for t in mock_tensors if term in t.get('name', '').lower()]
        elif "dtype " in conditions_part:
            term = conditions_part.split("dtype ")[1].strip()
            results = [t for t in mock_tensors if t.get('dtype', '').lower() == term]
        elif "metadata " in conditions_part:
            meta_parts = conditions_part.split("metadata ")[1].split(" ", 1)
            meta_key = meta_parts[0].strip()
            if len(meta_parts) > 1:
                meta_val = meta_parts[1].strip().replace("'", "").replace('"', '')
                results = [t for t in mock_tensors if t.get('metadata', {}).get(meta_key, '').lower() == meta_val]
            else: results = [t for t in mock_tensors if meta_key in t.get('metadata', {})]
        elif "schema_name " in conditions_part:
            term = conditions_part.split("schema_name ")[1].strip().replace("'", "").replace('"', '')
            results = [t for t in mock_tensors if t.get('schema_name', '').lower() == term]
        else: results = []
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 400

@app.route('/api/tensor_operation', methods=['POST'])
def handle_tensor_operation():
    global mock_tensors
    try:
        op_data = request.get_json(); operation = op_data.get('operation')
        if not operation: return jsonify({'error': 'Operation not specified'}), 400
        def find_tensor(tensor_id):
            for t_ in mock_tensors:
                if t_['id'] == tensor_id: return t_
            return None
        result_tensor_desc = {'name': 'OperationResult', 'shape': 'N/A', 'dtype': 'N/A', 'data': 'Symbolic result', 'metadata': {}}
        if operation == 'add':
            t1_id = op_data.get('tensor_id1'); t2_id = op_data.get('tensor_id2')
            if t1_id is None or t2_id is None: return jsonify({'error': 'Missing tensor_id1 or tensor_id2 for add operation'}), 400
            t1 = find_tensor(int(t1_id)); t2 = find_tensor(int(t2_id))
            if not t1 or not t2: return jsonify({'error': 'One or both tensors not found'}), 404
            if t1['shape'] != t2['shape']: return jsonify({'error': f"Shape mismatch: {t1['shape']} vs {t2['shape']}. For this demo, shapes must match."}), 400
            result_tensor_desc['dtype'] = t1['dtype'] if t1['dtype'] == t2['dtype'] else f"Mixed ({t1['dtype']}/{t2['dtype']})"
            result_tensor_desc['name'] = f"SumOf_{t1['name']}_and_{t2['name']}"; result_tensor_desc['shape'] = t1['shape']
            result_tensor_desc['data'] = f"Symbolic: {t1['name']} (ID: {t1_id}) + {t2['name']} (ID: {t2_id})"
            result_tensor_desc['metadata'] = {'operation_performed': 'add', 'source_ids': [t1_id, t2_id]}
        elif operation == 'multiply_scalar':
            t_id = op_data.get('tensor_id'); scalar = op_data.get('scalar')
            if t_id is None or scalar is None: return jsonify({'error': 'Missing tensor_id or scalar for multiply_scalar operation'}), 400
            t = find_tensor(int(t_id))
            if not t: return jsonify({'error': 'Tensor not found'}), 404
            try: float(scalar)
            except ValueError: return jsonify({'error': 'Scalar must be a number'}), 400
            result_tensor_desc['name'] = f"{t['name']}_multiplied_by_{scalar}"; result_tensor_desc['shape'] = t['shape']; result_tensor_desc['dtype'] = t['dtype']
            result_tensor_desc['data'] = f"Symbolic: {t['name']} (ID: {t_id}) * {scalar}"
            result_tensor_desc['metadata'] = {'operation_performed': 'multiply_scalar', 'source_ids': [t_id], 'scalar_value': scalar}
        elif operation == 'transpose':
            t_id = op_data.get('tensor_id')
            if t_id is None: return jsonify({'error': 'Missing tensor_id for transpose operation'}), 400
            t = find_tensor(int(t_id))
            if not t: return jsonify({'error': 'Tensor not found'}), 404
            try:
                shape_str_json_compat = t['shape'].replace("'", '"')
                shape_list = json.loads(shape_str_json_compat)
                if len(shape_list) >= 2: shape_list[0], shape_list[1] = shape_list[1], shape_list[0]; result_tensor_desc['shape'] = json.dumps(shape_list)
                else: result_tensor_desc['shape'] = t['shape']
            except Exception as parse_ex: print(f"Shape parsing/transposing error: {parse_ex}"); result_tensor_desc['shape'] = "Transposed " + t['shape']
            result_tensor_desc['name'] = f"TransposeOf_{t['name']}"; result_tensor_desc['dtype'] = t['dtype']
            result_tensor_desc['data'] = f"Symbolic: Transpose({t['name']} (ID: {t_id}))"
            result_tensor_desc['metadata'] = {'operation_performed': 'transpose', 'source_ids': [t_id]}
        else: return jsonify({'error': f"Unknown operation: {operation}"}), 400
        return jsonify(result_tensor_desc), 200
    except Exception as e: print(f"Error in handle_tensor_operation: {e}"); return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

# Agent API Endpoints
@app.route('/api/agents', methods=['GET'])
def get_agents_overview():
    summary_data = [{k: v for k, v in agent.items() if k not in ['mock_logs', 'config']} for agent in mock_agents_data]
    return jsonify(summary_data)

@app.route('/api/agents/<string:agent_id>', methods=['GET'])
def get_agent_details(agent_id):
    agent = next((a for a in mock_agents_data if a['id'] == agent_id), None)
    if agent:
        return jsonify(agent)
    return jsonify({'error': 'Agent not found'}), 404

@app.route('/api/agents/<string:agent_id>/status', methods=['GET'])
def get_agent_status(agent_id): # Note: This endpoint is not directly used by the proposed JS, which gets full details.
    agent = next((a for a in mock_agents_data if a['id'] == agent_id), None)
    if agent:
        if agent['status'] == 'Running' and agent['id'] == 'ingestion_agent_images':
            if len(agent['mock_logs']) < 10:
                 agent['mock_logs'].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Found 2 new images. Ingesting...")
        return jsonify({'status': agent['status']})
    return jsonify({'error': 'Agent not found'}), 404

@app.route('/api/agents/<string:agent_id>/logs', methods=['GET'])
def get_agent_logs(agent_id): # Note: This endpoint is not directly used by the proposed JS.
    agent = next((a for a in mock_agents_data if a['id'] == agent_id), None)
    if agent:
        return jsonify({'logs': agent.get('mock_logs', [])[-10:]})
    return jsonify({'error': 'Agent not found'}), 404

@app.route('/api/agents/<string:agent_id>/action', methods=['POST'])
def agent_action(agent_id):
    agent = next((a for a in mock_agents_data if a['id'] == agent_id), None)
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404

    action = request.json.get('action')
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_limit_reached = len(agent['mock_logs']) >= 20 # Limit total logs per agent for demo

    if action == 'start':
        if agent['status'] == 'Idle' or agent['status'] == 'Error' or agent['status'] == 'Stopped': # Allow start from Idle, Error, or Stopped
            agent['status'] = 'Running'
            if not log_limit_reached: agent['mock_logs'].append(f"{timestamp} - Agent start command received. Starting...")
        else:
            return jsonify({'message': 'Agent already running or in a non-startable state'}), 200 # Not an error, just info
    elif action == 'stop':
        if agent['status'] == 'Running':
            agent['status'] = 'Stopped' # Using 'Stopped' for a more definitive state
            if not log_limit_reached: agent['mock_logs'].append(f"{timestamp} - Agent stop command received. Stopping...")
        else:
            return jsonify({'message': 'Agent not running'}), 200
    elif action == 'mock_error' and agent['id'] == 'rl_agent_trading': # Specific demo action
        agent['status'] = 'Error'
        if not log_limit_reached: agent['mock_logs'].append(f"{timestamp} - CRITICAL ERROR: Mock failure occurred.")
    else:
        return jsonify({'error': f'Invalid action: {action}'}), 400

    return jsonify({'status': agent['status'], 'message': f"Action '{action}' processed."})

if __name__ == '__main__':
    # For development, you might use: app.run(host='0.0.0.0', port=5000, debug=True)
    # For production via Gunicorn, Gunicorn directly uses the 'app' object,
    # so this block might not even be strictly necessary if only Gunicorn is used.
    # However, to allow running with `python app.py` for simple local checks (not full dev mode):
    app.run(host='0.0.0.0', port=5000) # debug=True removed
