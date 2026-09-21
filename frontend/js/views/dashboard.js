/**
 * Ziro Predict - Dashboard View Controller
 */

import { api } from '../api.js';

export async function renderDashboard(container, router) {
  container.innerHTML = `
    <div class="dashboard-view">
      <!-- Quick Stats Header -->
      <div class="grid-4" style="margin-bottom: var(--space-6);">
        <div class="metric-box">
          <span class="metric-label">Backend Status</span>
          <div class="metric-val" id="dash-health-status">
            <span style="font-size: 1rem; color: var(--text-muted);">Checking...</span>
          </div>
          <span class="metric-delta" id="dash-health-detail" style="color: var(--text-muted);">Connecting to FastAPI</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Production Model</span>
          <div class="metric-val" style="font-size: 1.15rem; color: var(--brand-accent);">
            Logistic Regression
          </div>
          <span class="metric-delta" style="color: var(--text-secondary);">L2 Regularized (C=1.0)</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Test Macro F1</span>
          <div class="metric-val num" style="color: var(--color-up);">
            0.3563
          </div>
          <span class="metric-delta" style="color: var(--text-muted);">Phase 5 Benchmark (N=1,235)</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Supported Scope</span>
          <div class="metric-val" style="font-size: 1.15rem;">
            1 to 7 Sessions
          </div>
          <span class="metric-delta" style="color: var(--text-secondary);">2,353 NSE Equities</span>
        </div>
      </div>

      <!-- Quick Action: Launch Predictions -->
      <div class="card" style="margin-bottom: var(--space-6);">
        <div class="card-header">
          <div>
            <h3 class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
              Quick Movement Forecast
            </h3>
            <span class="card-subtitle">Launch instant live movement prediction for liquid benchmark stocks</span>
          </div>
        </div>

        <div class="grid-4" style="margin-top: var(--space-4);">
          <div class="card prob-card" style="cursor: pointer;" data-symbol="RELIANCE">
            <div class="prob-name" style="color: var(--brand-accent); font-size: 0.95rem;">RELIANCE</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin: 6px 0;">Reliance Industries Ltd.</div>
            <button class="btn btn-secondary btn-sm" style="width: 100%;">Predict RELIANCE</button>
          </div>

          <div class="card prob-card" style="cursor: pointer;" data-symbol="TCS">
            <div class="prob-name" style="color: var(--brand-accent); font-size: 0.95rem;">TCS</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin: 6px 0;">Tata Consultancy Services</div>
            <button class="btn btn-secondary btn-sm" style="width: 100%;">Predict TCS</button>
          </div>

          <div class="card prob-card" style="cursor: pointer;" data-symbol="INFY">
            <div class="prob-name" style="color: var(--brand-accent); font-size: 0.95rem;">INFY</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin: 6px 0;">Infosys Limited</div>
            <button class="btn btn-secondary btn-sm" style="width: 100%;">Predict INFY</button>
          </div>

          <div class="card prob-card" style="cursor: pointer;" data-symbol="360ONE">
            <div class="prob-name" style="color: var(--brand-accent); font-size: 0.95rem;">360ONE</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin: 6px 0;">360 ONE WAM Limited</div>
            <button class="btn btn-secondary btn-sm" style="width: 100%;">Predict 360ONE</button>
          </div>
        </div>
      </div>

      <!-- Architecture & System Overview -->
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
              Classification Architecture
            </h3>
          </div>
          <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; margin-bottom: var(--space-3);">
            This system classifies directional stock movement into three distinct target classes over 1 to 7 observed trading sessions:
          </p>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: var(--space-2); font-size: 0.85rem;">
            <li style="display: flex; align-items: center; gap: var(--space-2);">
              <span class="badge-direction up" style="padding: 2px 8px; font-size: 0.75rem;">UP</span>
              <span style="color: var(--text-primary); font-weight: 600;">Return &gt; +1.0%</span>
              <span style="color: var(--text-muted);">(Bullish movement beyond threshold)</span>
            </li>
            <li style="display: flex; align-items: center; gap: var(--space-2);">
              <span class="badge-direction stable" style="padding: 2px 8px; font-size: 0.75rem;">STABLE</span>
              <span style="color: var(--text-primary); font-weight: 600;">Return in [-1.0%, +1.0%]</span>
              <span style="color: var(--text-muted);">(Consolidation / neutral range)</span>
            </li>
            <li style="display: flex; align-items: center; gap: var(--space-2);">
              <span class="badge-direction down" style="padding: 2px 8px; font-size: 0.75rem;">DOWN</span>
              <span style="color: var(--text-primary); font-weight: 600;">Return &lt; -1.0%</span>
              <span style="color: var(--text-muted);">(Bearish movement beyond threshold)</span>
            </li>
          </ul>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
              Self-Contained Data Engine
            </h3>
          </div>
          <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; margin-bottom: var(--space-3);">
            The backend executes strictly offline against an immutable frozen Parquet dataset with zero external database dependencies:
          </p>
          <div style="display: flex; flex-direction: column; gap: var(--space-2); font-size: 0.82rem; color: var(--text-secondary);">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px;">
              <span>Dataset Rows:</span>
              <strong class="num" style="color: var(--text-primary);">2,639,421 candles</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px;">
              <span>Universe Size:</span>
              <strong class="num" style="color: var(--text-primary);">2,353 NSE Equities</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px;">
              <span>Data Boundary:</span>
              <strong class="num" style="color: var(--text-primary);">2026-08-10 to 2026-09-10</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span>Engine:</span>
              <strong style="color: var(--brand-accent);">PyArrow Dataset (Zero PostgreSQL)</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Attach quick-action click handlers
  container.querySelectorAll('[data-symbol]').forEach(card => {
    card.addEventListener('click', () => {
      const sym = card.getAttribute('data-symbol');
      router.navigate('predict', { symbol: sym });
    });
  });

  // Fetch live health status
  try {
    const health = await api.getHealth();
    const statusEl = container.querySelector('#dash-health-status');
    const detailEl = container.querySelector('#dash-health-detail');
    if (statusEl && health.status === 'HEALTHY') {
      statusEl.innerHTML = `<span style="color: var(--status-healthy);">${health.status}</span>`;
      detailEl.textContent = `${health.model_name}`;
    }
  } catch (err) {
    const statusEl = container.querySelector('#dash-health-status');
    const detailEl = container.querySelector('#dash-health-detail');
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--status-danger);">OFFLINE</span>`;
      detailEl.textContent = err.message || 'Cannot reach FastAPI';
    }
  }
}
