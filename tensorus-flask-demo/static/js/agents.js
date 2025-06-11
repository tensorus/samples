document.addEventListener('DOMContentLoaded', function () {
    const agentDashboardContainer = document.getElementById('agent-dashboard-container');

    let activeTimers = {}; // To store interval timers for each agent

    function clearPlaceholders() {
        if (agentDashboardContainer) {
            const placeholders = agentDashboardContainer.querySelectorAll('.placeholder-agent-card');
            placeholders.forEach(p => p.remove());
            const loadingText = agentDashboardContainer.querySelector('.loading-text');
            if (loadingText) loadingText.remove();
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
        if (!config || typeof config !== 'object' || Object.keys(config).length === 0) return '<small>N/A</small>';
        let html = '<ul class="config-list">';
        for (const key in config) {
            html += `<li><span class="fw-bold">${escapeHtml(key)}:</span> ${escapeHtml(config[key])}</li>`;
        }
        html += '</ul>';
        return html;
    }

    function formatAgentLogs(logs) {
        if (!logs || logs.length === 0) return '<div class="logs-container"><small>No logs yet.</small></div>';
        let html = '<div class="logs-container">';
        // Display latest logs first by reversing a copy of the array
        logs.slice().reverse().forEach(log => {
            html += `<div>${escapeHtml(log)}</div>`;
        });
        html += '</div>';
        return html;
    }

    function getStatusBadgeClass(status) {
        status = status || "Unknown";
        if (status.toLowerCase() === 'running') return 'bg-success';
        if (status.toLowerCase() === 'idle' || status.toLowerCase() === 'stopped') return 'bg-secondary';
        if (status.toLowerCase() === 'error') return 'bg-danger';
        return 'bg-light text-dark';
    }

    async function fetchInitialAgents() {
        if (!agentDashboardContainer) return;

        try {
            const response = await fetch('/api/agents'); // This gets summary data
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const agentsSummary = await response.json();

            clearPlaceholders();

            if (agentsSummary.length === 0) {
                agentDashboardContainer.innerHTML = '<div class="col"><p>No agents configured.</p></div>'; return;
            }

            agentsSummary.forEach(agent => {
                const col = document.createElement('div');
                col.className = 'col';
                col.setAttribute('id', `agent-card-col-${agent.id}`); // Unique ID for the column div

                // Create the basic card structure
                col.innerHTML = `
                    <div class="card h-100 shadow-sm agent-card" id="agent-card-${agent.id}">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0 card-title-agent-name">${escapeHtml(agent.name)}</h5>
                            <span id="status-${agent.id}" class="badge ${getStatusBadgeClass(agent.status)}">${escapeHtml(agent.status)}</span>
                        </div>
                        <div class="card-body">
                            <p class="card-subtitle mb-2 text-muted small">${escapeHtml(agent.type)}</p>
                            <p class="card-text small" id="desc-${agent.id}">${escapeHtml(agent.description)}</p>

                            <div class="mb-2">
                                <h6 class="mt-3">Configuration:</h6>
                                <div id="config-${agent.id}"><small>Loading config...</small></div>
                            </div>
                            <div class="mb-2">
                                <h6 class="mt-3">Logs (Last 10):</h6>
                                <div id="logs-${agent.id}"><small>Loading logs...</small></div>
                            </div>
                        </div>
                        <div class="card-footer bg-light">
                            <button class="btn btn-sm btn-success agent-action-btn" data-agent-id="${agent.id}" data-action="start">Start</button>
                            <button class="btn btn-sm btn-warning agent-action-btn" data-agent-id="${agent.id}" data-action="stop">Stop</button>
                            ${agent.id === 'rl_agent_trading' ? `<button class="btn btn-sm btn-danger agent-action-btn" data-agent-id="${agent.id}" data-action="mock_error">Mock Error</button>` : ''}
                        </div>
                    </div>
                `;
                agentDashboardContainer.appendChild(col);
                updateAgentCardDetails(agent.id); // Initial fetch of full details

                if (activeTimers[agent.id]) clearInterval(activeTimers[agent.id]);
                activeTimers[agent.id] = setInterval(() => updateAgentCardDetails(agent.id), 7000); // Refresh every 7 seconds
            });

            // Add event listeners after all cards are in the DOM
            document.querySelectorAll('.agent-action-btn').forEach(button => {
                button.addEventListener('click', function() {
                    handleAgentAction(this.dataset.agentId, this.dataset.action);
                });
            });

        } catch (error) {
            clearPlaceholders();
            agentDashboardContainer.innerHTML = `<div class="col"><p class="text-danger">Error fetching agents: ${error.message}</p></div>`;
            console.error('Error fetching agents:', error);
        }
    }

    async function updateAgentCardDetails(agentId) {
        try {
            const detailsResponse = await fetch(`/api/agents/${agentId}`);
            if (!detailsResponse.ok) {
                 // Try to update status with a warning if details fail
                const statusEl = document.getElementById(`status-${agentId}`);
                if(statusEl) {
                    statusEl.textContent = "Update Error";
                    statusEl.className = "badge bg-warning text-dark";
                }
                throw new Error(`Details HTTP error! status: ${detailsResponse.status}`);
            }
            const agent = await detailsResponse.json();

            const statusEl = document.getElementById(`status-${agentId}`);
            const configEl = document.getElementById(`config-${agentId}`);
            const logsEl = document.getElementById(`logs-${agentId}`);
            const descEl = document.getElementById(`desc-${agentId}`); // If description can change
            const nameEl = document.querySelector(`#agent-card-${agentId} .card-title-agent-name`);


            if (nameEl) nameEl.textContent = escapeHtml(agent.name); // Assuming name might change
            if (descEl) descEl.textContent = escapeHtml(agent.description); // Assuming description might change

            if (statusEl) {
                statusEl.textContent = escapeHtml(agent.status);
                statusEl.className = `badge ${getStatusBadgeClass(agent.status)}`;
            }
            if (configEl) configEl.innerHTML = formatAgentConfig(agent.config);
            if (logsEl) logsEl.innerHTML = formatAgentLogs(agent.mock_logs);

        } catch (error) {
            console.error(`Error updating card for agent ${agentId}:`, error);
            // Optionally update UI to show error for this specific agent card
            const cardContent = document.querySelector(`#agent-card-${agentId} .card-body`);
            if (cardContent) {
                // cardContent.innerHTML = `<p class="text-danger small">Could not update details for this agent.</p>`;
            }
        }
    }

    async function handleAgentAction(agentId, action) {
        const actionButton = document.querySelector(`.agent-action-btn[data-agent-id="${agentId}"][data-action="${action}"]`);
        let originalButtonText = '';
        if(actionButton) {
            originalButtonText = actionButton.innerHTML;
            actionButton.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ${action}...`;
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
            // Optional: Show a small success message, but immediate refresh is often enough
            // console.log(`Action '${action}' for agent ${agentId}: ${result.message || 'Completed'}`);
            updateAgentCardDetails(agentId);
        } catch (error) {
            alert(`Error performing action '${action}' for agent ${agentId}: ${error.message}`);
            console.error('Agent action error:', error);
        } finally {
            if(actionButton){
                actionButton.innerHTML = originalButtonText.replace(/<span.*<\/span>\s*/, ''); // Remove spinner if any
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
