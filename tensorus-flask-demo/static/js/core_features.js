document.addEventListener('DOMContentLoaded', function () {
    const tensorListDiv = document.getElementById('tensor-list');
    const addTensorForm = document.getElementById('add-tensor-form');

    // Tensor Operations Elements
    const opTypeSelect = document.getElementById('op-type');
    const opTensor1Select = document.getElementById('op-tensor1-id');
    const opTensor2Select = document.getElementById('op-tensor2-id');
    const opTensor1Group = document.getElementById('op-tensor1-group');
    const opTensor2Group = document.getElementById('op-tensor2-group');
    const opScalarGroup = document.getElementById('op-scalar-group');
    const opScalarInput = document.getElementById('op-scalar');
    const tensorOpForm = document.getElementById('tensor-op-form');
    const tensorOpResultDiv = document.getElementById('tensor-op-result');

    // NQL Query Elements
    const nqlQueryForm = document.getElementById('nql-query-form');
    const nqlQueryInput = document.getElementById('nql-query-string');
    const nqlResultsDiv = document.getElementById('nql-results-list');


    // Helper function to escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') {
            return '';
        }
        // Ensure it's a string before trying to replace
        const strUnsafe = String(unsafe);
        return strUnsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    function populateTensorSelects(tensors) {
        if (!opTensor1Select || !opTensor2Select) return; // In case elements are not on the page

        opTensor1Select.innerHTML = '';
        opTensor2Select.innerHTML = '';
        if (tensors.length === 0) {
            const defaultOption = '<option value="">No tensors available</option>';
            opTensor1Select.innerHTML = defaultOption;
            opTensor2Select.innerHTML = defaultOption;
            return;
        }
        tensors.forEach(tensor => {
            const option = document.createElement('option');
            option.value = tensor.id;
            option.textContent = `${escapeHtml(tensor.name)} (ID: ${tensor.id}, Shape: ${escapeHtml(tensor.shape)})`;
            opTensor1Select.appendChild(option.cloneNode(true));
            opTensor2Select.appendChild(option.cloneNode(true));
        });
    }

    async function fetchTensorsAndPopulate() {
        try {
            const response = await fetch('/api/tensors');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const tensors = await response.json();

            if (tensorListDiv) {
                tensorListDiv.innerHTML = '';
                if (tensors.length === 0) {
                    tensorListDiv.innerHTML = '<p>No tensors stored yet.</p>';
                } else {
                    const ul = document.createElement('ul');
                    ul.className = 'list-group';
                    tensors.forEach(tensor => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        let schemaText = tensor.schema_name ? ` <span class="badge bg-secondary">${escapeHtml(tensor.schema_name)}</span>` : '';
                        li.innerHTML = `
                            <h5>${escapeHtml(tensor.name)} (ID: ${tensor.id})${schemaText}</h5>
                            <small>Shape: ${escapeHtml(tensor.shape)}, DType: ${escapeHtml(tensor.dtype)}</small>
                            <p>Data: <code>${escapeHtml(tensor.data)}</code></p>
                            <p><small>Metadata: <code>${escapeHtml(JSON.stringify(tensor.metadata))}</code></small></p>
                        `;
                        ul.appendChild(li);
                    });
                    tensorListDiv.appendChild(ul);
                }
            }
            populateTensorSelects(tensors);

        } catch (error) {
            if (tensorListDiv) {
                tensorListDiv.innerHTML = `<p class="text-danger">Error fetching tensors: ${error.message}</p>`;
            }
            console.error('Error fetching tensors:', error);
            populateTensorSelects([]);
        }
    }

    async function addTensor(event) {
        event.preventDefault();

        const name = document.getElementById('tensor-name').value;
        const shape = document.getElementById('tensor-shape').value;
        const dtype = document.getElementById('tensor-dtype').value;
        const data = document.getElementById('tensor-data').value;
        const metadataString = document.getElementById('tensor-metadata').value;
        const schemaName = document.getElementById('tensor-schema-name').value.trim();

        let metadata = {};
        if (metadataString.trim() !== '') {
            try {
                metadata = JSON.parse(metadataString);
            } catch (e) {
                alert('Invalid JSON format for metadata: ' + e.message);
                return;
            }
        }

        const newTensorPayload = { name, shape, dtype, data, metadata };
        if (schemaName) {
            newTensorPayload.schema_name = schemaName;
        }

        try {
            const response = await fetch('/api/tensors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newTensorPayload),
            });

            const responseData = await response.json();
            if (!response.ok) {
                throw new Error(responseData.error || `HTTP error! status: ${response.status}`);
            }

            addTensorForm.reset();
            fetchTensorsAndPopulate(); // Refresh the list AND dropdowns
        } catch (error) {
            alert(`Error adding tensor: ${error.message}`);
            console.error('Error adding tensor:', error);
        }
    }

    if (addTensorForm) {
        addTensorForm.addEventListener('submit', addTensor);
    }

    if (opTypeSelect) {
        opTypeSelect.addEventListener('change', function() {
            const selectedOp = this.value;
            opTensor1Group.style.display = 'block';
            if (selectedOp === 'add') {
                opTensor2Group.style.display = 'block';
                opScalarGroup.style.display = 'none';
            } else if (selectedOp === 'multiply_scalar') {
                opTensor2Group.style.display = 'none';
                opScalarGroup.style.display = 'block';
            } else if (selectedOp === 'transpose') {
                opTensor2Group.style.display = 'none';
                opScalarGroup.style.display = 'none';
            } else {
                opTensor2Group.style.display = 'block';
                opScalarGroup.style.display = 'none';
            }
        });
        opTypeSelect.dispatchEvent(new Event('change')); // Initial UI setup
    }

    if (tensorOpForm) {
        tensorOpForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const operation = opTypeSelect.value;
            const tensorId1 = opTensor1Select.value;
            const payload = { operation };

            if (!tensorId1 && (operation === 'add' || operation === 'multiply_scalar' || operation === 'transpose')) {
                 alert('Please select Tensor 1 for the operation. If no tensors are available, add one first.'); return;
            }

            if (operation === 'add') {
                const tensorId2 = opTensor2Select.value;
                if (!tensorId2) { alert('Please select Tensor 2 for addition.'); return; }
                payload.tensor_id1 = parseInt(tensorId1);
                payload.tensor_id2 = parseInt(tensorId2);
            } else if (operation === 'multiply_scalar') {
                const scalar = opScalarInput.value;
                if (scalar === '' || isNaN(parseFloat(scalar))) { alert('Please enter a valid scalar value.'); return; }
                payload.tensor_id = parseInt(tensorId1);
                payload.scalar = parseFloat(scalar);
            } else if (operation === 'transpose') {
                payload.tensor_id = parseInt(tensorId1);
            } else {
                alert('Invalid operation selected.');
                return;
            }

            try {
                const response = await fetch('/api/tensor_operation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.error || `HTTP error ${response.status}`);
                }
                tensorOpResultDiv.innerHTML = `
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">${escapeHtml(result.name)}</h5>
                            <p class="card-text">Shape: ${escapeHtml(result.shape)}</p>
                            <p class="card-text">DType: ${escapeHtml(result.dtype)}</p>
                            <p class="card-text">Data: <code>${escapeHtml(result.data)}</code></p>
                            <p class="card-text"><small>Metadata: <code>${escapeHtml(JSON.stringify(result.metadata))}</code></small></p>
                        </div>
                    </div>`;
            } catch (error) {
                tensorOpResultDiv.innerHTML = `<p class="text-danger">Operation failed: ${error.message}</p>`;
                console.error('Tensor operation error:', error);
            }
        });
    }

    // NQL Query Handling
    if (nqlQueryForm) {
        nqlQueryForm.addEventListener('submit', async function handleNqlQuery(event) {
            event.preventDefault();
            const queryString = nqlQueryInput.value.trim();
            if (!queryString) {
                alert('Please enter an NQL query.');
                return;
            }
            try {
                const response = await fetch('/api/nql_query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query_string: queryString }),
                });
                const responseData = await response.json();
                if (!response.ok) {
                    throw new Error(responseData.error || `HTTP error! status: ${response.status}`);
                }
                displayNqlResults(responseData);
            } catch (error) {
                nqlResultsDiv.innerHTML = `<p class="text-danger">Error processing NQL query: ${error.message}</p>`;
                console.error('Error processing NQL query:', error);
            }
        });
    }

    function displayNqlResults(tensors) {
        if (!nqlResultsDiv) return;
        nqlResultsDiv.innerHTML = '';
        if (!tensors || tensors.length === 0) {
            nqlResultsDiv.innerHTML = '<p>No tensors found matching your query.</p>';
            return;
        }
        const ul = document.createElement('ul');
        ul.className = 'list-group';
        tensors.forEach(tensor => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            let schemaText = tensor.schema_name ? ` <span class="badge bg-secondary">${escapeHtml(tensor.schema_name)}</span>` : '';
            li.innerHTML = `
                <h5>${escapeHtml(tensor.name)} (ID: ${tensor.id})${schemaText}</h5>
                <small>Shape: ${escapeHtml(tensor.shape)}, DType: ${escapeHtml(tensor.dtype)}</small>
                <p>Data: <code>${escapeHtml(tensor.data)}</code></p>
                <p><small>Metadata: <code>${escapeHtml(JSON.stringify(tensor.metadata))}</code></small></p>
            `;
            ul.appendChild(li);
        });
        nqlResultsDiv.appendChild(ul);
    }

    // Initial data fetch and population of UI elements
    if (tensorListDiv) { // Check if on a page that needs this
        fetchTensorsAndPopulate();
    }
});
