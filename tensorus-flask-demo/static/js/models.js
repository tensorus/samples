document.addEventListener('DOMContentLoaded', function () {
    const modelListContainer = document.getElementById('model-list-container');

    // Re-define escapeHtml or ensure it's globally available if this were a multi-file setup for real
    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') {
            return '';
        }
        const strUnsafe = String(unsafe); // Ensure it's a string
        return strUnsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    async function fetchModels() {
        if (!modelListContainer) return; // Exit if container not found

        try {
            const response = await fetch('/api/models');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const models = await response.json();

            modelListContainer.innerHTML = ''; // Clear placeholder or previous content
            if (models.length === 0) {
                modelListContainer.innerHTML = '<div class="col"><p>No models available to display.</p></div>';
                return;
            }

            models.forEach(model => {
                const col = document.createElement('div');
                col.className = 'col'; // Bootstrap will handle responsive column behavior in row-cols-*

                let predictButtonHtml = '';
                // Only add mock predict button for xgboost_regressor as per initial spec
                if (model.id === 'xgboost_regressor') {
                    predictButtonHtml = `<button class="btn btn-sm btn-outline-primary mt-auto mock-predict-btn" data-model-id="${escapeHtml(model.id)}" data-model-name="${escapeHtml(model.name)}">Mock Predict</button>`;
                }

                col.innerHTML = `
                    <div class="card h-100 shadow-sm">
                        <div class="card-body d-flex flex-column">
                            <h5 class="card-title">${escapeHtml(model.name)}</h5>
                            <h6 class="card-subtitle mb-2 text-muted">${escapeHtml(model.category)}</h6>
                            <p class="card-text small flex-grow-1">${escapeHtml(model.description)}</p>
                            <p class="card-text mb-1"><small><strong>Example Input:</strong><br><code>${escapeHtml(model.example_input)}</code></small></p>
                            <p class="card-text"><small><strong>Example Output:</strong><br><code>${escapeHtml(model.example_output)}</code></small></p>
                            <div class="mock-result-display mt-2"></div> <!-- Div for prediction result -->
                            ${predictButtonHtml}
                        </div>
                        <div class="card-footer bg-transparent border-top-0 pt-0">
                            <a href="${escapeHtml(model.doc_link || '#')}" class="btn btn-sm btn-outline-secondary" target="_blank" rel="noopener noreferrer">Documentation</a>
                        </div>
                    </div>
                `;
                modelListContainer.appendChild(col);
            });

            // Add event listeners for predict buttons
            document.querySelectorAll('.mock-predict-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const modelId = this.dataset.modelId;
                    const modelName = this.dataset.modelName;
                    // Pass the specific div for results related to this button's card
                    const resultDisplayDiv = this.closest('.card-body').querySelector('.mock-result-display');
                    handleMockPredict(modelId, modelName, this, resultDisplayDiv);
                });
            });

        } catch (error) {
            modelListContainer.innerHTML = `<div class="col"><p class="text-danger">Error fetching models: ${error.message}</p></div>`;
            console.error('Error fetching models:', error);
        }
    }

    async function handleMockPredict(modelId, modelName, buttonElement, resultDisplayDiv) {
        const mockInputFromUser = prompt(`Enter mock input for ${modelName} (or leave blank for default defined in API):`, "");
        // If user cancels prompt, mockInputFromUser will be null.
        // If user enters nothing and clicks OK, it will be an empty string.
        // The backend API is designed to handle this by using a default if no input is provided.

        const originalButtonText = buttonElement.textContent;
        buttonElement.textContent = 'Predicting...';
        buttonElement.disabled = true;
        resultDisplayDiv.innerHTML = ''; // Clear previous results

        try {
            const response = await fetch(`/api/models/${modelId}/mock_predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // Send the input, even if it's null or empty, let backend decide
                body: JSON.stringify({ mock_input: mockInputFromUser })
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || `HTTP error ${response.status}`);
            }

            resultDisplayDiv.innerHTML = `<p class="mb-0"><small><strong>Mock Prediction:</strong> <code>${escapeHtml(result.prediction)}</code></small></p>`;

        } catch (error) {
            resultDisplayDiv.innerHTML = `<p class="text-danger mb-0"><small>Error: ${escapeHtml(error.message)}</small></p>`;
            console.error('Mock prediction error:', error);
        } finally {
            buttonElement.textContent = originalButtonText;
            buttonElement.disabled = false;
        }
    }

    fetchModels();
});
