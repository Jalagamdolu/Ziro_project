/**
 * Ziro Predict - Performance Dashboard View Controller
 */

import { api } from '../api.js';

export async function renderPerformance(container) {
  container.innerHTML = `
    <div class="performance-view">
      <!-- Loading State -->
      <div id="perf-loading" style="padding: var(--space-12); text-align: center;">
        <div class="spinner" style="margin: 0 auto var(--space-4); width: 32px; height: 32px;"></div>
        <h4 style="color: var(--text-secondary); font-family: var(--font-heading);">
          Aggregating Forward Paper Performance &amp; Historical Benchmarks...
        </h4>
      </div>

      <!-- Loaded Content Container -->
      <div id="perf-content" style="display: none;"></div>
    </div>
  `;

  const loadingEl = container.querySelector('#perf-loading');
  const contentEl = container.querySelector('#perf-content');

  try {
    const data = await api.getPerformance();
    loadingEl.style.display = 'none';
    contentEl.style.display = 'block';

    const hist = data.historical_benchmark || {};
    const fwd = data.forward_paper_metrics || {};
    const hasResolved = data.resolved_predictions > 0 && fwd.status === 'VALID';

    contentEl.innerHTML = `
      <!-- Top Summary Metrics -->
      <div class="grid-4" style="margin-bottom: var(--space-6);">
        <div class="metric-box">
          <span class="metric-label">Total Predictions</span>
          <div class="metric-val num">${data.total_predictions}</div>
          <span class="metric-delta" style="color: var(--text-muted);">Recorded in SQLite</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Resolved Predictions</span>
          <div class="metric-val num" style="color: var(--color-up);">${data.resolved_predictions}</div>
          <span class="metric-delta" style="color: var(--text-muted);">Evaluated against market close</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Open Predictions</span>
          <div class="metric-val num" style="color: var(--brand-accent);">${data.open_predictions}</div>
          <span class="metric-delta" style="color: var(--text-muted);">Pending target session</span>
        </div>

        <div class="metric-box">
          <span class="metric-label">Expired / Unresolvable</span>
          <div class="metric-val num" style="color: var(--text-muted);">${data.expired_predictions}</div>
          <span class="metric-delta" style="color: var(--text-muted);">Missing target data</span>
        </div>
      </div>

      <!-- Side-by-Side Benchmark Comparison -->
      <div class="benchmark-comparison">
        <!-- 1. Historical Test Benchmark -->
        <div class="benchmark-card historical">
          <div class="card-header">
            <div>
              <span class="badge" style="background: rgba(2, 132, 199, 0.2); color: var(--brand-accent); margin-bottom: 4px;">
                STATIC BASELINE
              </span>
              <h3 class="card-title">HISTORICAL TEST BENCHMARK</h3>
              <span class="card-subtitle">${hist.split_name || 'Phase 5 Historical Test Set'} (N=${hist.sample_count || 1235})</span>
            </div>
          </div>

          <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: var(--space-4);">
            ${hist.notes || 'Chronological out-of-sample test set (Sep 09 to Sep 10, 2026). Locked production baseline.'}
          </p>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); font-size: 0.85rem;">
            <div class="metric-box" style="padding: var(--space-3);">
              <span class="metric-label">Accuracy</span>
              <span class="metric-val num" style="font-size: 1.25rem;">${(hist.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="metric-box" style="padding: var(--space-3);">
              <span class="metric-label">Macro F1</span>
              <span class="metric-val num" style="font-size: 1.25rem; color: var(--brand-accent);">${hist.macro_f1?.toFixed(4) || '0.3563'}</span>
            </div>
            <div class="metric-box" style="padding: var(--space-3);">
              <span class="metric-label">Balanced Acc</span>
              <span class="metric-val num" style="font-size: 1.25rem;">${(hist.balanced_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="metric-box" style="padding: var(--space-3);">
              <span class="metric-label">UP Recall</span>
              <span class="metric-val num" style="font-size: 1.25rem; color: var(--status-warning);">${(hist.up_recall * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <!-- 2. Forward Paper Evaluation -->
        <div class="benchmark-card forward">
          <div class="card-header">
            <div>
              <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: var(--color-up); margin-bottom: 4px;">
                LIVE EVALUATION
              </span>
              <h3 class="card-title">FORWARD PAPER EVALUATION</h3>
              <span class="card-subtitle">Realized Forward Performance (N=${data.resolved_predictions})</span>
            </div>
          </div>

          ${hasResolved ? `
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: var(--space-4);">
              Performance measured on real forward paper predictions resolved after target timestamps elapsed.
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); font-size: 0.85rem;">
              <div class="metric-box" style="padding: var(--space-3);">
                <span class="metric-label">Accuracy</span>
                <span class="metric-val num" style="font-size: 1.25rem;">${(fwd.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div class="metric-box" style="padding: var(--space-3);">
                <span class="metric-label">Macro F1</span>
                <span class="metric-val num" style="font-size: 1.25rem; color: var(--color-up);">${fwd.macro_f1?.toFixed(4) || '-'}</span>
              </div>
              <div class="metric-box" style="padding: var(--space-3);">
                <span class="metric-label">Balanced Acc</span>
                <span class="metric-val num" style="font-size: 1.25rem;">${(fwd.balanced_accuracy * 100).toFixed(1)}%</span>
              </div>
              <div class="metric-box" style="padding: var(--space-3);">
                <span class="metric-label">Macro Recall</span>
                <span class="metric-val num" style="font-size: 1.25rem;">${(fwd.macro_recall * 100).toFixed(1)}%</span>
              </div>
            </div>
          ` : `
            <div style="padding: var(--space-8); text-align: center; color: var(--text-muted);">
              <div style="font-size: 1.6rem; margin-bottom: var(--space-2);">⏳</div>
              <h4 style="color: var(--text-secondary); margin-bottom: var(--space-1);">No Resolved Predictions Yet</h4>
              <p style="font-size: 0.82rem;">
                Forward metrics will automatically populate once target sessions are reached and predictions are resolved against market data.
              </p>
            </div>
          `}
        </div>
      </div>

      <!-- Per-Class Metrics & Confusion Matrix -->
      ${hasResolved && fwd.per_class ? `
        <div class="grid-2" style="margin-bottom: var(--space-6);">
          <!-- Per-Class Metrics Table -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">Per-Class Performance</h3>
              <span class="card-subtitle">Precision, Recall &amp; F1</span>
            </div>
            <div class="table-container" style="border: none;">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  ${['DOWN', 'STABLE', 'UP'].map(cls => {
                    const m = fwd.per_class[cls] || {};
                    return `
                      <tr>
                        <td><span class="badge-direction ${cls.toLowerCase()}" style="padding: 2px 8px; font-size: 0.72rem;">${cls}</span></td>
                        <td class="num">${m.precision !== undefined ? (m.precision * 100).toFixed(1) + '%' : '-'}</td>
                        <td class="num">${m.recall !== undefined ? (m.recall * 100).toFixed(1) + '%' : '-'}</td>
                        <td class="num">${m.f1 !== undefined ? m.f1.toFixed(4) : '-'}</td>
                        <td class="num">${m.support || 0}</td>
                      </tr>
                    `;
                  }).join('')}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Confusion Matrix Heatmap -->
          <div class="card">
            <div class="card-header">
              <h3 class="card-title">Confusion Matrix</h3>
              <span class="card-subtitle">Rows: Actual | Columns: Predicted</span>
            </div>
            ${fwd.confusion_matrix ? `
              <table class="confusion-matrix-table">
                <thead>
                  <tr>
                    <th>Actual \\ Pred</th>
                    <th>Pred DOWN</th>
                    <th>Pred STABLE</th>
                    <th>Pred UP</th>
                  </tr>
                </thead>
                <tbody>
                  ${['DOWN', 'STABLE', 'UP'].map((rowLabel, rIdx) => `
                    <tr>
                      <th style="text-align: left; font-weight: 600;">${rowLabel}</th>
                      ${[0, 1, 2].map(cIdx => {
                        const val = fwd.confusion_matrix[rIdx]?.[cIdx] ?? 0;
                        const isDiag = rIdx === cIdx;
                        return `<td class="${isDiag ? 'diagonal' : ''}">${val}</td>`;
                      }).join('')}
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            ` : '<p style="color: var(--text-muted);">No confusion matrix data.</p>'}
          </div>
        </div>
      ` : ''}

      <!-- Performance by Horizon (H1 - H7) -->
      ${data.by_horizon ? `
        <div class="card" style="margin-bottom: var(--space-6);">
          <div class="card-header">
            <div>
              <h3 class="card-title">Performance by Forecast Horizon</h3>
              <span class="card-subtitle">1 to 7 observed trading sessions ahead</span>
            </div>
          </div>
          <div class="table-container" style="border: none;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>Status</th>
                  <th>Sample Count</th>
                  <th>Accuracy</th>
                  <th>Balanced Acc</th>
                  <th>Macro F1</th>
                  <th>DOWN Recall</th>
                  <th>STABLE Recall</th>
                  <th>UP Recall</th>
                </tr>
              </thead>
              <tbody>
                ${Object.keys(data.by_horizon).map(hKey => {
                  const hData = data.by_horizon[hKey];
                  const isValid = hData.status === 'VALID';
                  return `
                    <tr>
                      <td><strong>H${hKey}</strong> (${hKey} session${hKey > 1 ? 's' : ''})</td>
                      <td>
                        <span class="badge ${isValid ? 'badge-resolved' : 'badge-expired'}">
                          ${hData.status}
                        </span>
                      </td>
                      <td class="num">${hData.sample_count}</td>
                      <td class="num">${isValid && hData.accuracy !== null ? (hData.accuracy * 100).toFixed(1) + '%' : '-'}</td>
                      <td class="num">${isValid && hData.balanced_accuracy !== null ? (hData.balanced_accuracy * 100).toFixed(1) + '%' : '-'}</td>
                      <td class="num" style="color: var(--brand-accent);">${isValid && hData.macro_f1 !== null ? hData.macro_f1.toFixed(4) : '-'}</td>
                      <td class="num">${isValid && hData.down_recall !== null ? (hData.down_recall * 100).toFixed(1) + '%' : '-'}</td>
                      <td class="num">${isValid && hData.stable_recall !== null ? (hData.stable_recall * 100).toFixed(1) + '%' : '-'}</td>
                      <td class="num" style="color: var(--status-warning);">${isValid && hData.up_recall !== null ? (hData.up_recall * 100).toFixed(1) + '%' : '-'}</td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        </div>
      ` : ''}

      <!-- Confidence / Probability Tiers -->
      ${data.confidence_tiers && data.confidence_tiers.length > 0 ? `
        <div class="card">
          <div class="card-header">
            <div>
              <h3 class="card-title">Confidence / Probability Tiers</h3>
              <span class="card-subtitle">Realized accuracy grouped by maximum predicted probability band</span>
            </div>
          </div>
          <div class="table-container" style="border: none;">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Confidence Tier</th>
                  <th>Prediction Count</th>
                  <th>Accuracy</th>
                  <th>Macro F1</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                ${data.confidence_tiers.map(tier => `
                  <tr>
                    <td><strong class="num">${tier.tier}</strong></td>
                    <td class="num">${tier.prediction_count}</td>
                    <td class="num">${tier.accuracy !== null ? (tier.accuracy * 100).toFixed(1) + '%' : '-'}</td>
                    <td class="num">${tier.macro_f1 !== null ? tier.macro_f1.toFixed(4) : '-'}</td>
                    <td style="color: var(--text-muted); font-size: 0.8rem;">${tier.notes || '-'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      ` : ''}
    </div>
  `;
  } catch (err) {
    loadingEl.style.display = 'none';
    contentEl.style.display = 'block';
    contentEl.innerHTML = `
      <div class="card" style="border-color: rgba(239, 68, 68, 0.4); text-align: center; padding: var(--space-8);">
        <div style="color: var(--status-danger); font-size: 2rem; margin-bottom: var(--space-2);">⚠️</div>
        <h4 style="color: #fca5a5; font-family: var(--font-heading); margin-bottom: var(--space-2);">Failed to Load Performance Metrics</h4>
        <p style="color: var(--text-secondary); font-size: 0.85rem; max-width: 500px; margin: 0 auto var(--space-4);">
          ${err.message || 'The server returned an error while computing performance metrics.'}
        </p>
        <button id="btn-retry-perf" class="btn btn-secondary btn-sm">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          <span>Retry</span>
        </button>
      </div>
    `;
    const retryBtn = contentEl.querySelector('#btn-retry-perf');
    if (retryBtn) {
      retryBtn.addEventListener('click', () => renderPerformance(container));
    }
  }
}
