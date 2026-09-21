/**
 * Ziro Predict - Prediction History View Controller
 */

import { api } from '../api.js';

export async function renderHistory(container) {
  container.innerHTML = `
    <div class="history-view">
      <!-- Filter Bar -->
      <div class="filter-bar">
        <div class="form-group">
          <label class="form-label" for="filter-symbol">Symbol</label>
          <input type="text" id="filter-symbol" class="form-input" placeholder="All symbols..." style="padding: 6px 10px;" />
        </div>

        <div class="form-group">
          <label class="form-label" for="filter-status">Status</label>
          <select id="filter-status" class="form-select" style="padding: 6px 10px;">
            <option value="">All Statuses</option>
            <option value="OPEN">OPEN</option>
            <option value="RESOLVED">RESOLVED</option>
            <option value="EXPIRED">EXPIRED</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label" for="filter-horizon">Horizon</label>
          <select id="filter-horizon" class="form-select" style="padding: 6px 10px;">
            <option value="">All Horizons</option>
            <option value="1">H1 (1 session)</option>
            <option value="2">H2 (2 sessions)</option>
            <option value="3">H3 (3 sessions)</option>
            <option value="4">H4 (4 sessions)</option>
            <option value="5">H5 (5 sessions)</option>
            <option value="6">H6 (6 sessions)</option>
            <option value="7">H7 (7 sessions)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label" for="filter-class">Predicted Class</label>
          <select id="filter-class" class="form-select" style="padding: 6px 10px;">
            <option value="">All Classes</option>
            <option value="UP">UP</option>
            <option value="STABLE">STABLE</option>
            <option value="DOWN">DOWN</option>
          </select>
        </div>

        <div class="form-group" style="max-width: 90px;">
          <label class="form-label" for="filter-limit">Limit</label>
          <select id="filter-limit" class="form-select" style="padding: 6px 10px;">
            <option value="50">50</option>
            <option value="100">100</option>
            <option value="200" selected>200</option>
            <option value="500">500</option>
          </select>
        </div>

        <div style="display: flex; gap: var(--space-2); align-self: flex-end;">
          <button id="btn-apply-filters" class="btn btn-primary btn-sm">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <span>Filter</span>
          </button>
          <button id="btn-resolve-all" class="btn btn-secondary btn-sm" title="Batch resolve all eligible OPEN predictions">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            <span>Resolve Due</span>
          </button>
        </div>
      </div>

      <!-- Table Container -->
      <div class="card" style="padding: 0; overflow: hidden;">
        <div id="history-table-container">
          <div style="padding: var(--space-8); text-align: center;">
            <div class="spinner" style="margin: 0 auto var(--space-3); width: 28px; height: 28px;"></div>
            <span style="color: var(--text-muted); font-size: 0.85rem;">Loading prediction history...</span>
          </div>
        </div>
      </div>
    </div>
  `;

  const tableContainer = container.querySelector('#history-table-container');
  const filterBtn = container.querySelector('#btn-apply-filters');
  const resolveAllBtn = container.querySelector('#btn-resolve-all');

  // Load predictions
  async function loadTable() {
    const symbol = container.querySelector('#filter-symbol').value.trim().toUpperCase();
    const status = container.querySelector('#filter-status').value;
    const horizon = container.querySelector('#filter-horizon').value;
    const predClass = container.querySelector('#filter-class').value;
    const limit = container.querySelector('#filter-limit').value;

    tableContainer.innerHTML = `
      <div style="padding: var(--space-8); text-align: center;">
        <div class="spinner" style="margin: 0 auto var(--space-3); width: 28px; height: 28px;"></div>
        <span style="color: var(--text-muted); font-size: 0.85rem;">Fetching records...</span>
      </div>
    `;

    try {
      const records = await api.getPredictions({
        symbol: symbol || undefined,
        status: status || undefined,
        horizon: horizon ? parseInt(horizon, 10) : undefined,
        predicted_class: predClass || undefined,
        limit: parseInt(limit, 10) || 200
      });

      if (!records || records.length === 0) {
        tableContainer.innerHTML = `
          <div style="padding: var(--space-12); text-align: center; color: var(--text-muted);">
            <div style="font-size: 1.8rem; margin-bottom: var(--space-2);">📋</div>
            <h4 style="color: var(--text-secondary); margin-bottom: var(--space-1);">No Predictions Found</h4>
            <p style="font-size: 0.85rem;">No historical predictions match the selected filter criteria.</p>
          </div>
        `;
        return;
      }

      let rowsHtml = records.map(rec => {
        const predBadge = `<span class="badge-direction ${rec.predicted_class.toLowerCase()}" style="padding: 2px 8px; font-size: 0.72rem;">${rec.predicted_class}</span>`;
        
        let statusBadge = '';
        if (rec.status === 'RESOLVED') {
          statusBadge = `<span class="badge badge-resolved">RESOLVED</span>`;
        } else if (rec.status === 'OPEN') {
          statusBadge = `<span class="badge badge-open">OPEN</span>`;
        } else {
          statusBadge = `<span class="badge badge-expired">${rec.status}</span>`;
        }

        const actualClass = rec.actual_class 
          ? `<span class="badge-direction ${rec.actual_class.toLowerCase()}" style="padding: 2px 8px; font-size: 0.72rem;">${rec.actual_class}</span>`
          : '<span style="color: var(--text-muted);">-</span>';

        const returnPct = rec.future_return_pct !== null && rec.future_return_pct !== undefined
          ? `<span class="num" style="color: ${rec.future_return_pct >= 1.0 ? 'var(--color-up)' : rec.future_return_pct <= -1.0 ? 'var(--color-down)' : 'var(--color-stable)'}; font-weight: 600;">${rec.future_return_pct > 0 ? '+' : ''}${rec.future_return_pct.toFixed(2)}%</span>`
          : '<span style="color: var(--text-muted);">-</span>';

        const probsText = `<span class="num" style="font-size: 0.75rem; color: var(--text-secondary);">D: ${(rec.probability_down * 100).toFixed(0)}% S: ${(rec.probability_stable * 100).toFixed(0)}% U: ${(rec.probability_up * 100).toFixed(0)}%</span>`;

        const resolveAction = rec.status === 'OPEN'
          ? `<button class="btn btn-secondary btn-sm btn-resolve-single" data-id="${rec.prediction_id}" style="padding: 2px 8px; font-size: 0.72rem;">Resolve</button>`
          : '<span style="color: var(--text-muted); font-size: 0.75rem;">Settled</span>';

        return `
          <tr>
            <td class="num" style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--brand-accent);">${rec.prediction_id.slice(0, 12)}</td>
            <td><strong style="color: var(--text-primary);">${rec.symbol}</strong></td>
            <td class="num" style="font-size: 0.78rem; color: var(--text-secondary);">${rec.reference_timestamp}</td>
            <td class="num" style="font-size: 0.78rem; color: var(--text-secondary);">${rec.target_timestamp}</td>
            <td>${predBadge}</td>
            <td>${probsText}</td>
            <td>${statusBadge}</td>
            <td>${actualClass}</td>
            <td>${returnPct}</td>
            <td>${resolveAction}</td>
          </tr>
        `;
      }).join('');

      tableContainer.innerHTML = `
        <div class="table-container" style="border: none; border-radius: 0;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Prediction ID</th>
                <th>Symbol</th>
                <th>Reference (IST)</th>
                <th>Target (IST)</th>
                <th>Predicted</th>
                <th>Probabilities</th>
                <th>Status</th>
                <th>Actual</th>
                <th>Return</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${rowsHtml}
            </tbody>
          </table>
        </div>
      `;

      // Attach single resolve handlers
      tableContainer.querySelectorAll('.btn-resolve-single').forEach(btn => {
        btn.addEventListener('click', async () => {
          const id = btn.getAttribute('data-id');
          btn.disabled = true;
          btn.textContent = '...';
          try {
            await api.resolvePrediction(id);
            await loadTable();
          } catch (err) {
            alert(`Failed to resolve prediction ${id}: ${err.message}`);
            btn.disabled = false;
            btn.textContent = 'Resolve';
          }
        });
      });

    } catch (err) {
      tableContainer.innerHTML = `
        <div style="padding: var(--space-8); text-align: center; color: var(--status-danger);">
          <p>Failed to load prediction history: ${err.message}</p>
          <button id="btn-retry-history" class="btn btn-secondary btn-sm" style="margin-top: var(--space-3);">Retry</button>
        </div>
      `;
      const retryBtn = tableContainer.querySelector('#btn-retry-history');
      if (retryBtn) retryBtn.addEventListener('click', loadTable);
    }
  }

  filterBtn.addEventListener('click', loadTable);

  resolveAllBtn.addEventListener('click', async () => {
    resolveAllBtn.disabled = true;
    resolveAllBtn.innerHTML = `<div class="spinner"></div><span>Resolving...</span>`;
    try {
      const res = await api.resolveAll();
      alert(`Settlement Complete: ${res.resolution_counts.resolved} resolved, ${res.resolution_counts.expired} expired, ${res.resolution_counts.still_open} still open.`);
      await loadTable();
    } catch (err) {
      alert(`Batch resolution error: ${err.message}`);
    } finally {
      resolveAllBtn.disabled = false;
      resolveAllBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        <span>Resolve Due</span>
      `;
    }
  });

  // Initial load
  loadTable();
}
