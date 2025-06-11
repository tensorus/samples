document.addEventListener('DOMContentLoaded', function () {
    const datasetListContainer = document.getElementById('dataset-list-container');

    // Function to remove placeholder elements
    function clearPlaceholders() {
        if (datasetListContainer) {
            // The new placeholder is a single div with ID 'loading-placeholder-container'
            const placeholderContainer = document.getElementById('loading-placeholder-container');
            if (placeholderContainer) {
                placeholderContainer.remove();
            }
            // Remove any lingering "Loading datasets..." text if it's separate and not part of the placeholder div
            const loadingTextElements = datasetListContainer.querySelectorAll('.loading-text'); // Assuming class for direct text
            loadingTextElements.forEach(lt => lt.remove());
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
            return '<p class="text-xs text-gray-500 dark:text-gray-400">No properties listed.</p>';
        }
        // Tailwind classes for the list and items
        let html = '<ul class="space-y-1 text-xs text-gray-600 dark:text-gray-400">';
        for (const key in properties) {
            html += `<li><strong class="font-medium text-gray-700 dark:text-gray-300">${escapeHtml(key)}:</strong> ${escapeHtml(properties[key])}</li>`;
        }
        html += '</ul>';
        return html;
    }

    function getCategoryIconSVG(category) {
        // Basic placeholder SVG. A real implementation might have specific SVGs per category.
        // Returning a generic document icon for now.
        return `
            <svg class="w-8 h-8 text-blue-500 dark:text-blue-400 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
        `;
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
                datasetListContainer.innerHTML = '<div class="md:col-span-2 text-center py-8"><p class="text-gray-500 dark:text-gray-400">No datasets available to display.</p></div>';
                return;
            }

            datasets.forEach(dataset => {
                const datasetCard = document.createElement('div');
                // Tailwind classes for the card from datasets.html example
                datasetCard.className = 'bg-white dark:bg-gray-800 shadow-lg rounded-lg overflow-hidden flex flex-col h-full';

                const iconSVG = getCategoryIconSVG(dataset.category);
                const sourceBadge = `<span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-sky-100 text-sky-700 dark:bg-sky-700 dark:text-sky-100">${escapeHtml(dataset.source)}</span>`;

                datasetCard.innerHTML = `
                    <div class="p-6 flex flex-col flex-grow">
                        <div class="flex items-start mb-2">
                            ${iconSVG}
                            <div>
                                <h5 class="text-xl font-semibold text-gray-800 dark:text-white">${escapeHtml(dataset.name)}</h5>
                                <p class="text-sm text-gray-500 dark:text-gray-400 mb-1">${escapeHtml(dataset.category)} - ${sourceBadge}</p>
                            </div>
                        </div>
                        <p class="text-sm text-gray-600 dark:text-gray-300 mb-4 flex-grow">${escapeHtml(dataset.description)}</p>

                        <div class="mb-3">
                            <strong class="block mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">Properties:</strong>
                            ${formatProperties(dataset.properties)}
                        </div>

                        <div>
                            <strong class="block mb-1 text-sm font-medium text-gray-700 dark:text-gray-300">Example Data Description:</strong>
                            <p class="text-sm text-gray-600 dark:text-gray-400 italic">
                                <code class="block bg-gray-100 dark:bg-gray-700/50 p-2 rounded-sm text-xs overflow-x-auto whitespace-pre-wrap break-all">${escapeHtml(dataset.example_data_description)}</code>
                            </p>
                        </div>
                    </div>
                    <div class="p-4 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                        <a href="#" class="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-800 rounded-sm">Explore Dataset</a>
                    </div>
                `;
                datasetListContainer.appendChild(datasetCard);
            });

        } catch (error) {
            clearPlaceholders();
            datasetListContainer.innerHTML = `<div class="md:col-span-2"><p class="text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 p-3 rounded-md">Error fetching datasets: ${error.message}</p></div>`;
            console.error('Error fetching datasets:', error);
        }
    }

    fetchDatasets();
});
