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

            // Clear placeholder (which is now a specific div, not the container itself)
            const loadingPlaceholder = document.getElementById('loading-placeholder-container');
            if (loadingPlaceholder) {
                loadingPlaceholder.remove();
            }

            if (models.length === 0) {
                modelListContainer.innerHTML = '<div class="col-span-1 md:col-span-2 lg:col-span-3 text-center py-8"><p class="text-gray-500 dark:text-gray-400">No models available to display.</p></div>';
                return;
            }

            models.forEach(model => {
                const modelCard = document.createElement('div');
                // Tailwind classes for the card, matching the example in models.html head_extra
                modelCard.className = 'bg-white dark:bg-gray-800 shadow-lg rounded-lg overflow-hidden flex flex-col h-full';

                let predictButtonHtml = '';
                if (model.id === 'xgboost_regressor') {
                    predictButtonHtml = `<button class="mt-auto text-sm py-1 px-3 border border-blue-500 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-700/30 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-800 mock-predict-btn" data-model-id="${escapeHtml(model.id)}" data-model-name="${escapeHtml(model.name)}">Mock Predict</button>`;
                }

                // Using example structure from models.html comments
                modelCard.innerHTML = `
                    <div class="p-6 flex flex-col flex-grow">
                        <h5 class="text-xl font-semibold mb-1 text-gray-800 dark:text-white">${escapeHtml(model.name)}</h5>
                        <p class="text-xs text-gray-500 dark:text-gray-400 mb-3">${escapeHtml(model.category)}</p>
                        <p class="text-sm text-gray-600 dark:text-gray-300 mb-4 flex-grow">${escapeHtml(model.description)}</p>
                        <div class="mb-3">
                            <p class="text-xs text-gray-500 dark:text-gray-400 mb-1"><strong>Example Input:</strong></p>
                            <code class="block bg-gray-100 dark:bg-gray-700 p-1.5 rounded-sm text-xs overflow-x-auto whitespace-pre-wrap break-all">${escapeHtml(model.example_input)}</code>
                        </div>
                        <div>
                            <p class="text-xs text-gray-500 dark:text-gray-400 mb-1"><strong>Example Output:</strong></p>
                            <code class="block bg-gray-100 dark:bg-gray-700 p-1.5 rounded-sm text-xs overflow-x-auto whitespace-pre-wrap break-all">${escapeHtml(model.example_output)}</code>
                        </div>
                        <div class="mock-result-display mt-3 text-xs font-mono break-all p-2 rounded bg-gray-50 dark:bg-gray-700/60 border border-gray-200 dark:border-gray-600 min-h-[30px]"></div>
                        ${predictButtonHtml ? `<div class="mt-4">${predictButtonHtml}</div>` : ''}
                    </div>
                    <div class="p-4 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                        <a href="${escapeHtml(model.doc_link || '#')}" class="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-800 rounded-sm" target="_blank" rel="noopener noreferrer">Documentation</a>
                    </div>
                `;
                modelListContainer.appendChild(modelCard);
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
            modelListContainer.innerHTML = `<div class="col-span-1 md:col-span-2 lg:col-span-3"><p class="text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 p-3 rounded-md">Error fetching models: ${error.message}</p></div>`;
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

            resultDisplayDiv.innerHTML = `<p class="mb-0"><strong class="text-gray-700 dark:text-gray-300">Mock Prediction:</strong> <code class="text-pink-600 dark:text-pink-400 bg-gray-100 dark:bg-gray-700 px-1 rounded-sm">${escapeHtml(result.prediction)}</code></p>`;

        } catch (error) {
            resultDisplayDiv.innerHTML = `<p class="text-red-600 dark:text-red-400 mb-0"><small>Error: ${escapeHtml(error.message)}</small></p>`;
            console.error('Mock prediction error:', error);
        } finally {
            buttonElement.textContent = originalButtonText;
            buttonElement.disabled = false;
        }
    }

    fetchModels();
});
