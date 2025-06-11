document.addEventListener('DOMContentLoaded', function () {
    const agentDashboardContainer = document.getElementById('agent-dashboard-container');

    let activeTimers = {}; // To store interval timers for each agent

    function clearPlaceholders() {
        if (agentDashboardContainer) {
            const placeholderContainer = document.getElementById('loading-placeholder-container');
            if (placeholderContainer) {
                placeholderContainer.remove();
            }
            const loadingTextElements = agentDashboardContainer.querySelectorAll('.loading-text');
            loadingTextElements.forEach(lt => lt.remove());
        }
    }

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

    function formatAgentConfig(config) {
        if (!config || typeof config !== 'object' || Object.keys(config).length === 0) return '<p class="text-xs text-gray-500 dark:text-gray-400">N/A</p>';
        let html = '<ul class="text-xs text-gray-600 dark:text-gray-400 space-y-0.5">';
        for (const key in config) {
            html += `<li><strong class="font-medium text-gray-700 dark:text-gray-300">${escapeHtml(key)}:</strong> ${escapeHtml(config[key])}</li>`;
        }
        html += '</ul>';
        return html;
    }

    function formatAgentLogs(logs) {
        if (!logs || logs.length === 0) return '<p class="text-xs text-gray-500 dark:text-gray-400">No logs yet.</p>';
        let html = ''; // Container div classes are applied directly where this function is called
        logs.slice().reverse().forEach(log => { // Display latest logs first
            html += `<div class="p-0.5">${escapeHtml(log)}</div>`;
        });
        return html;
    }

    function getStatusBadgeClasses(status) {
        status = String(status || "Unknown").toLowerCase();
        let baseClasses = "px-2 py-0.5 text-xs font-semibold rounded-full";
        if (status === 'running') return `${baseClasses} bg-green-100 text-green-800 dark:bg-green-700 dark:text-green-100`;
        if (status === 'idle' || status === 'stopped') return `${baseClasses} bg-gray-200 text-gray-800 dark:bg-gray-600 dark:text-gray-100`;
        if (status === 'error') return `${baseClasses} bg-red-100 text-red-800 dark:bg-red-700 dark:text-red-100`;
        return `${baseClasses} bg-yellow-100 text-yellow-800 dark:bg-yellow-700 dark:text-yellow-100`; // For Unknown or other statuses
    }

    async function fetchInitialAgents() {
        if (!agentDashboardContainer) return;

        try {
            const response = await fetch('/api/agents');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const agentsSummary = await response.json();

            clearPlaceholders();

            if (agentsSummary.length === 0) {
                agentDashboardContainer.innerHTML = '<div class="lg:col-span-2 text-center py-8"><p class="text-gray-500 dark:text-gray-400">No agents configured.</p></div>'; return;
            }

            agentsSummary.forEach(agent => {
                const agentCard = document.createElement('div');
                agentCard.className = 'bg-white dark:bg-gray-800 shadow-lg rounded-lg overflow-hidden flex flex-col h-full';
                agentCard.setAttribute('id', `agent-card-${agent.id}`);

                agentCard.innerHTML = `
                    <div class="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
                        <h5 class="text-lg font-semibold text-gray-800 dark:text-white card-title-agent-name">${escapeHtml(agent.name)}</h5>
                        <span id="status-${agent.id}" class="${getStatusBadgeClasses(agent.status)}">${escapeHtml(agent.status)}</span>
                    </div>
                    <div class="p-4 space-y-4 flex-grow">
                        <p class="text-xs text-gray-500 dark:text-gray-400 mb-2">${escapeHtml(agent.type)}</p>
                        <p class="text-sm text-gray-600 dark:text-gray-300" id="desc-${agent.id}">${escapeHtml(agent.description)}</p>
                        <div>
                            <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">Configuration:</h6>
                            <div id="config-${agent.id}"><p class="text-xs text-gray-500 dark:text-gray-400">Loading config...</p></div>
                        </div>
                        <div>
                            <h6 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">Recent Logs (Last 10):</h6>
                            <div id="logs-${agent.id}" class="text-xs font-mono max-h-36 overflow-y-auto border border-gray-200 dark:border-gray-600 p-2 bg-gray-50 dark:bg-gray-900/50 whitespace-pre-wrap break-all rounded">
                                <p class="text-xs text-gray-500 dark:text-gray-400">Loading logs...</p>
                            </div>
                        </div>
                    </div>
                    <div class="p-4 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700 flex-shrink-0 space-x-2">
                        <button class="text-xs py-1 px-3 border border-green-500 text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-700/30 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 dark:focus:ring-offset-gray-800 agent-action-btn" data-agent-id="${agent.id}" data-action="start">Start</button>
                        <button class="text-xs py-1 px-3 border border-red-500 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-700/30 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 dark:focus:ring-offset-gray-800 agent-action-btn" data-agent-id="${agent.id}" data-action="stop">Stop</button>
                        ${agent.id === 'rl_agent_trading' ? `<button class="text-xs py-1 px-3 border border-yellow-500 text-yellow-600 dark:text-yellow-400 hover:bg-yellow-50 dark:hover:bg-yellow-700/30 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500 dark:focus:ring-offset-gray-800 agent-action-btn" data-agent-id="${agent.id}" data-action="mock_error">Mock Error</button>` : ''}
                    </div>
                `;
                agentDashboardContainer.appendChild(agentCard);
                updateAgentCardDetails(agent.id);

                if (activeTimers[agent.id]) clearInterval(activeTimers[agent.id]);
                activeTimers[agent.id] = setInterval(() => updateAgentCardDetails(agent.id), 7000);
            });

            document.querySelectorAll('.agent-action-btn').forEach(button => {
                button.addEventListener('click', function() {
                    handleAgentAction(this.dataset.agentId, this.dataset.action);
                });
            });

        } catch (error) {
            clearPlaceholders();
            agentDashboardContainer.innerHTML = `<div class="lg:col-span-2"><p class="text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 p-3 rounded-md">Error fetching agents: ${error.message}</p></div>`;
            console.error('Error fetching agents:', error);
        }
    }

    async function updateAgentCardDetails(agentId) {
        try {
            const detailsResponse = await fetch(`/api/agents/${agentId}`);
            if (!detailsResponse.ok) {
                const statusEl = document.getElementById(`status-${agentId}`);
                if(statusEl) {
                    statusEl.textContent = "Update Err";
                    statusEl.className = getStatusBadgeClasses("error");
                }
                return;
            }
            const agent = await detailsResponse.json();

            const statusEl = document.getElementById(`status-${agentId}`);
            const configEl = document.getElementById(`config-${agentId}`);
            const logsEl = document.getElementById(`logs-${agentId}`);
            const descEl = document.getElementById(`desc-${agentId}`);
            const nameEl = document.querySelector(`#agent-card-${agentId} .card-title-agent-name`);

            if (nameEl) nameEl.textContent = escapeHtml(agent.name);
            if (descEl) descEl.textContent = escapeHtml(agent.description);

            if (statusEl) {
                statusEl.textContent = escapeHtml(agent.status);
                statusEl.className = getStatusBadgeClasses(agent.status);
            }
            if (configEl) configEl.innerHTML = formatAgentConfig(agent.config);
            if (logsEl) logsEl.innerHTML = formatAgentLogs(agent.mock_logs);

        } catch (error) {
            console.error(`Error updating card for agent ${agentId}:`, error);
            const cardContent = document.querySelector(`#agent-card-${agentId} .p-4.space-y-4.flex-grow`); // Target body
            if (cardContent) {
                 // Optionally indicate error on card without wiping content, e.g. special border or small message
            }
        }
    }

    async function handleAgentAction(agentId, action) {
        const actionButton = document.querySelector(`.agent-action-btn[data-agent-id="${agentId}"][data-action="${action}"]`);
        let originalButtonText = '';
        if(actionButton) {
            originalButtonText = actionButton.textContent;
            actionButton.innerHTML = `Processing...`;
            actionButton.disabled = true;
        }

        try {
            const response = await fetch(`/api/agents/${agentId}/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || `Action HTTP error ${response.status}`);
            }
            updateAgentCardDetails(agentId);
        } catch (error) {
            alert(`Error performing action '${action}' for agent ${agentId}: ${error.message}`);
            console.error('Agent action error:', error);
        } finally {
            if(actionButton){
                actionButton.innerHTML = originalButtonText;
                actionButton.disabled = false;
            }
        }
    }

    window.addEventListener('beforeunload', () => {
        for (const agentId in activeTimers) {
            clearInterval(activeTimers[agentId]);
        }
        activeTimers = {};
    });

    if (agentDashboardContainer) {
        fetchInitialAgents();
    }
});
