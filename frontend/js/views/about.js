/**
 * Ziro Predict - About & Settings View Controller
 */

import { getApiBaseUrl, setApiBaseUrl, CONFIG } from '../config.js';
import { api } from '../api.js';

export function renderAbout(container) {
  const currentUrl = getApiBaseUrl();

  container.innerHTML = `
    <div class="about-view">
      <!-- Backend Settings Card -->
      <div class="card" style="margin-bottom: var(--space-6);">
        <div class="card-header">
          <div>
            <h3 class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              FastAPI Backend Endpoint Configuration
            </h3>
            <span class="card-subtitle">Configure the active REST API base URL for this frontend</span>
          </div>
        </div>

        <div style="max-width: 650px;">
          <div class="form-group">
            <label class="form-label" for="setting-api-url">
              <span>Backend Base URL</span>
              <span class="form-hint">Default: ${CONFIG.DEFAULT_API_BASE_URL}</span>
            </label>
            <div style="display: flex; gap: var(--space-2);">
              <input type="url" id="setting-api-url" class="form-input" value="${currentUrl}" style="flex: 1;" placeholder="http://127.0.0.1:8000" />
              <button id="btn-save-url" class="btn btn-primary btn-sm">Save</button>
              <button id="btn-test-url" class="btn btn-secondary btn-sm">Test Connection</button>
            </div>
          </div>
          <div id="url-test-feedback" style="font-size: 0.82rem; margin-top: var(--space-2); display: none;"></div>
        </div>
      </div>

      <!-- Project Architecture Overview -->
      <div class="grid-2" style="margin-bottom: var(--space-6);">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Project Mission &amp; Purpose</h3>
          </div>
          <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; margin-bottom: var(--space-3);">
            The <strong>Ziro NSE Stock Movement Prediction System</strong> is an academic and research-driven machine learning system designed to evaluate multi-horizon directional stock classification on the National Stock Exchange of India (NSE).
          </p>
          <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6;">
            By separating feature generation strictly up to the reference timestamp from target evaluation, the system guarantees <strong>zero lookahead leakage</strong> across all training, validation, testing, and forward paper evaluation pipelines.
          </p>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Architecture Stack</h3>
          </div>
          <div style="display: flex; flex-direction: column; gap: var(--space-2); font-size: 0.85rem; color: var(--text-secondary);">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span>Frontend:</span>
              <strong style="color: var(--text-primary);">HTML5, CSS3 Tokens, Modular ES6 JS</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span>Backend:</span>
              <strong style="color: var(--brand-accent);">FastAPI REST (Uvicorn / ASGI)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span>Market Data:</span>
              <strong style="color: var(--text-primary);">PyArrow Dataset (Frozen Parquet)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span>Paper Trading State:</span>
              <strong style="color: var(--text-primary);">SQLite (Local ACID persistence)</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span>Deployment:</span>
              <strong style="color: var(--color-up);">100% Self-Contained (Zero External DB)</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- REST API Endpoint Reference -->
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">FastAPI Endpoint Reference</h3>
          <span class="card-subtitle">Production endpoints available on the backend</span>
        </div>
        <div class="table-container" style="border: none;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Endpoint Path</th>
                <th>Classification</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span class="badge" style="background: rgba(16, 185, 129, 0.2); color: var(--color-up);">GET</span></td>
                <td><code>/health</code></td>
                <td>Public</td>
                <td>System health, model diagnostics, and supported horizon scope</td>
              </tr>
              <tr>
                <td><span class="badge" style="background: rgba(2, 132, 199, 0.2); color: var(--brand-accent);">POST</span></td>
                <td><code>/predict</code></td>
                <td>Public</td>
                <td>Stateless movement classification using locked production model</td>
              </tr>
              <tr>
                <td><span class="badge" style="background: rgba(2, 132, 199, 0.2); color: var(--brand-accent);">POST</span></td>
                <td><code>/paper/predict</code></td>
                <td>Paper Evaluation</td>
                <td>Stateful paper prediction logged to SQLite tracker</td>
              </tr>
              <tr>
                <td><span class="badge" style="background: rgba(16, 185, 129, 0.2); color: var(--color-up);">GET</span></td>
                <td><code>/paper/predictions</code></td>
                <td>Paper Evaluation</td>
                <td>Filtered history of paper predictions</td>
              </tr>
              <tr>
                <td><span class="badge" style="background: rgba(16, 185, 129, 0.2); color: var(--color-up);">GET</span></td>
                <td><code>/paper/performance</code></td>
                <td>Governance</td>
                <td>Real-time paper performance vs Phase 5 historical test benchmark</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // Attach URL Save & Test handlers
  const urlInput = container.querySelector('#setting-api-url');
  const saveBtn = container.querySelector('#btn-save-url');
  const testBtn = container.querySelector('#btn-test-url');
  const feedbackEl = container.querySelector('#url-test-feedback');

  saveBtn.addEventListener('click', () => {
    const newUrl = urlInput.value.trim();
    if (!newUrl) {
      alert("Please enter a valid URL.");
      return;
    }
    setApiBaseUrl(newUrl);
    feedbackEl.innerHTML = `<span style="color: var(--color-up);">Saved API Base URL: <strong>${newUrl}</strong></span>`;
    feedbackEl.style.display = 'block';
  });

  testBtn.addEventListener('click', async () => {
    const testUrl = urlInput.value.trim();
    feedbackEl.innerHTML = `<span style="color: var(--brand-accent);">Pinging ${testUrl}/health...</span>`;
    feedbackEl.style.display = 'block';
    testBtn.disabled = true;

    try {
      const res = await fetch(`${testUrl.replace(/\/+$/, '')}/health`);
      if (res.ok) {
        const data = await res.json();
        feedbackEl.innerHTML = `<span style="color: var(--color-up);">Connection Successful! Backend reports: <strong>${data.status}</strong> (${data.model_name})</span>`;
      } else {
        feedbackEl.innerHTML = `<span style="color: var(--status-danger);">Server responded with HTTP ${res.status}</span>`;
      }
    } catch (err) {
      feedbackEl.innerHTML = `<span style="color: var(--status-danger);">Connection failed: ${err.message}. Ensure FastAPI is running and CORS is enabled.</span>`;
    } finally {
      testBtn.disabled = false;
    }
  });
}
