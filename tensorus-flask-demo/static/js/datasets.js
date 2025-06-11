document.addEventListener('DOMContentLoaded', function () {
    const datasetListContainer = document.getElementById('dataset-list-container');

    // Function to remove placeholder elements
    function clearPlaceholders() {
        if (datasetListContainer) {
            const placeholders = datasetListContainer.querySelectorAll('.placeholder-dataset-card');
            placeholders.forEach(p => p.remove());
            const loadingText = datasetListContainer.querySelector('.loading-text');
            if (loadingText) loadingText.remove();
        }
    }

    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        if (unsafe === null || typeof unsafe === 'undefined') {
            return '';
        }
        const strUnsafe = String(unsafe);
        return strUnsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    function formatProperties(properties) {
        if (!properties || typeof properties !== 'object' || Object.keys(properties).length === 0) {
            return '<p><small>No properties listed.</small></p>';
        }
        let html = '<ul class="list-unstyled mb-0 properties-list">';
        for (const key in properties) {
            html += `<li><small><strong>${escapeHtml(key)}:</strong> ${escapeHtml(properties[key])}</small></li>`;
        }
        html += '</ul>';
        return html;
    }

    function getCategoryIcon(category) {
        const catLower = category.toLowerCase();
        if (catLower.includes('image')) return 'bi-image-fill'; // Filled icon
        if (catLower.includes('time series')) return 'bi-graph-up-arrow'; // More specific
        if (catLower.includes('tabular')) return 'bi-grid-3x3-gap-fill'; // Filled icon
        return 'bi-file-earmark-text-fill'; // Default filled icon
    }

    async function fetchDatasets() {
        if (!datasetListContainer) return;

        try {
            const response = await fetch('/api/datasets');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const datasets = await response.json();

            clearPlaceholders();

            if (datasets.length === 0) {
                datasetListContainer.innerHTML = '<div class="col"><p>No datasets available to display.</p></div>';
                return;
            }

            datasets.forEach(dataset => {
                const col = document.createElement('div');
                col.className = 'col';

                const iconClass = getCategoryIcon(dataset.category);

                col.innerHTML = `
                    <div class="card h-100 shadow-sm">
                        <div class="card-body">
                            <h5 class="card-title"><i class="bi ${iconClass} icon-prepend"></i>${escapeHtml(dataset.name)}</h5>
                            <h6 class="card-subtitle mb-2 text-muted">
                                ${escapeHtml(dataset.category)} -
                                <span class="badge bg-info">${escapeHtml(dataset.source)}</span>
                            </h6>
                            <p class="card-text small">${escapeHtml(dataset.description)}</p>
                            <div class="mb-2">
                                <strong class="d-block mb-1">Properties:</strong>
                                ${formatProperties(dataset.properties)}
                            </div>
                            <div>
                                <strong class="d-block mb-1">Example Data Description:</strong>
                                <p class="card-text small fst-italic"><code>${escapeHtml(dataset.example_data_description)}</code></p>
                            </div>
                        </div>
                    </div>
                `;
                datasetListContainer.appendChild(col);
            });

        } catch (error) {
            clearPlaceholders();
            datasetListContainer.innerHTML = `<div class="col"><p class="text-danger">Error fetching datasets: ${error.message}</p></div>`;
            console.error('Error fetching datasets:', error);
        }
    }

    fetchDatasets();
});
