/**
 * Ziro Predict - Live Movement Prediction View Controller
 */

import { api } from '../api.js';
import { CONFIG } from '../config.js';

export function renderLivePrediction(container, params = {}) {
  const defaultSymbol = params.symbol || 'RELIANCE';
  const defaultRefTs = params.ref || '2026-09-09 10:30';
  const defaultTgtTs = params.tgt || '2026-09-10 10:30';

  container.innerHTML = `
    <div class="predict-layout">
      <!-- Prediction Form Column -->
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              Inference Parameters
            </h3>
            <span class="card-subtitle">Select equity ticker and forecast horizon</span>
          </div>
        </div>

        <!-- Validation Alert Box -->
        <div id="predict-error-box" class="alert alert-danger" style="display: none;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          <div id="predict-error-text"></div>
        </div>

        <form id="prediction-form">
          <!-- Stock Symbol -->
          <div class="form-group">
            <label class="form-label" for="input-symbol">
              <span>NSE Symbol</span>
              <span class="form-hint">e.g. RELIANCE, TCS, INFY</span>
            </label>
            <input 
              type="text" 
              id="input-symbol" 
              class="form-input" 
              value="${defaultSymbol}" 
              placeholder="Search ticker..." 
              required 
              autocomplete="off"
              list="symbols-datalist"
            />
            <datalist id="symbols-datalist">
              ${CONFIG.POPULAR_SYMBOLS.map(s => `<option value="${s}">`).join('')}
            </datalist>
          </div>

          <!-- Reference Timestamp -->
          <div class="form-group">
            <label class="form-label" for="input-ref-ts">
              <span>Reference Timestamp (IST)</span>
              <span class="form-hint">Data boundary: ≤ 2026-09-10</span>
            </label>
            <input 
              type="text" 
              id="input-ref-ts" 
              class="form-input" 
              value="${defaultRefTs}" 
              placeholder="YYYY-MM-DD HH:MM" 
              required
            />
          </div>

          <!-- Target Timestamp -->
          <div class="form-group">
            <label class="form-label" for="input-tgt-ts">
              <span>Target Timestamp (IST)</span>
              <span class="form-hint">1 to 7 sessions ahead</span>
            </label>
            <input 
              type="text" 
              id="input-tgt-ts" 
              class="form-input" 
              value="${defaultTgtTs}" 
              placeholder="YYYY-MM-DD HH:MM" 
              required
            />
          </div>

          <!-- Model Selection -->
          <div class="form-group">
            <label class="form-label" for="select-model">
              <span>Model Architecture</span>
              <span class="form-hint">Production vs Dedicated H1</span>
            </label>
            <select id="select-model" class="form-select">
              <option value="pooled" selected>Pooled Logistic Regression (All Horizons H1-H7)</option>
              <option value="h1">Dedicated H1 Random Forest (1 Session Only)</option>
            </select>
          </div>

          <button type="submit" id="btn-submit-predict" class="btn btn-primary" style="width: 100%; margin-top: var(--space-2);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            <span>Generate Movement Prediction</span>
          </button>
        </form>
      </div>

      <!-- Prediction Output Column -->
      <div id="prediction-result-container">
        <!-- Initial Empty State -->
        <div class="card" style="text-align: center; padding: var(--space-12) var(--space-6);">
          <div style="width: 54px; height: 54px; border-radius: var(--radius-full); background-color: var(--bg-surface-elevated); display: inline-flex; align-items: center; justify-content: center; margin-bottom: var(--space-4);">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          </div>
          <h3 style="font-family: var(--font-heading); font-size: 1.2rem; color: var(--text-primary); margin-bottom: var(--space-2);">
            Ready for Inference
          </h3>
          <p style="color: var(--text-secondary); max-width: 420px; margin: 0 auto; font-size: 0.85rem;">
            Click <strong>Generate Movement Prediction</strong> to query market data up to the reference timestamp and run the production classification model.
          </p>
        </div>
      </div>
    </div>
  `;

  const form = container.querySelector('#prediction-form');
  const errorBox = container.querySelector('#predict-error-box');
  const errorText = container.querySelector('#predict-error-text');
  const submitBtn = container.querySelector('#btn-submit-predict');
  const resultContainer = container.querySelector('#prediction-result-container');

  // Form submission handler
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.style.display = 'none';

    const symbol = container.querySelector('#input-symbol').value.trim().toUpperCase();
    const refTs = container.querySelector('#input-ref-ts').value.trim();
    const tgtTs = container.querySelector('#input-tgt-ts').value.trim();
    const modelType = container.querySelector('#select-model').value;

    // Client-side quick validations
    if (!symbol) {
      showError("Please enter an NSE equity symbol.");
      return;
    }

    // Set loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <div class="spinner"></div>
      <span>Executing Inference...</span>
    `;

    resultContainer.innerHTML = `
      <div class="card" style="padding: var(--space-8); text-align: center;">
        <div class="spinner" style="margin: 0 auto var(--space-4); width: 32px; height: 32px; border-width: 3px;"></div>
        <h4 style="font-family: var(--font-heading); color: var(--text-primary); margin-bottom: var(--space-1);">
          Evaluating Market Data &amp; Scaling Features
        </h4>
        <p style="color: var(--text-muted); font-size: 0.82rem;">
          Querying frozen Parquet dataset &amp; calculating 26 model features...
        </p>
      </div>
    `;

    try {
      const result = await api.predict({
        symbol,
        reference_timestamp: refTs,
        target_timestamp: tgtTs,
        model_type: modelType
      });

      renderPredictionResult(resultContainer, result);
    } catch (err) {
      showError(err.message || "Prediction request failed.");
      resultContainer.innerHTML = `
        <div class="card" style="border-color: rgba(239, 68, 68, 0.4); text-align: center; padding: var(--space-8);">
          <div style="color: var(--status-danger); font-size: 2rem; margin-bottom: var(--space-2);">⚠️</div>
          <h4 style="color: #fca5a5; font-family: var(--font-heading); margin-bottom: var(--space-2);">Inference Failed</h4>
          <p style="color: var(--text-secondary); font-size: 0.85rem; max-width: 500px; margin: 0 auto;">
            ${err.message || 'The server returned an error during prediction.'}
          </p>
        </div>
      `;
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        <span>Generate Movement Prediction</span>
      `;
    }
  });

  function showError(msg) {
    errorText.textContent = msg;
    errorBox.style.display = 'flex';
  }
}

/**
 * Renders the real prediction result returned from FastAPI
 */
function renderPredictionResult(container, data) {
  const pDown = (data.probability_down * 100).toFixed(2);
  const pStable = (data.probability_stable * 100).toFixed(2);
  const pUp = (data.probability_up * 100).toFixed(2);

  const predClass = data.predicted_class.toUpperCase();
  const classLower = predClass.toLowerCase();

  // Highlight highest probability card
  const isDownHigh = data.probability_down >= data.probability_stable && data.probability_down >= data.probability_up;
  const isStableHigh = data.probability_stable >= data.probability_down && data.probability_stable >= data.probability_up;
  const isUpHigh = data.probability_up >= data.probability_down && data.probability_up >= data.probability_stable;

  container.innerHTML = `
    <!-- Hero Result Card -->
    <div class="result-hero">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3);">
        <span class="badge" style="background: var(--bg-surface); border: 1px solid var(--border-medium); font-size: 0.8rem; padding: 4px 10px;">
          ${data.symbol}
        </span>
        <span style="font-size: 0.78rem; color: var(--text-muted);">
          Horizon: ${data.observed_sessions_ahead} session${data.observed_sessions_ahead > 1 ? 's' : ''} (${data.calendar_days_ahead} day${data.calendar_days_ahead > 1 ? 's' : ''})
        </span>
      </div>

      <div style="font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">
        Predicted Direction
      </div>

      <div class="result-direction-banner ${classLower}">
        <span>${predClass}</span>
      </div>

      <div style="margin-top: var(--space-2);">
        <span style="font-size: 0.8rem; color: var(--text-muted);">Reference Price:</span>
        <span class="price-display num" style="margin-left: 6px;">₹${data.reference_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</span>
      </div>

      <!-- 3-Segment Probability Distribution Bar -->
      <div class="prob-bar-container" title="DOWN: ${pDown}%, STABLE: ${pStable}%, UP: ${pUp}%">
        <div class="prob-segment down" style="width: ${pDown}%;">
          ${pDown > 15 ? pDown + '%' : ''}
        </div>
        <div class="prob-segment stable" style="width: ${pStable}%;">
          ${pStable > 15 ? pStable + '%' : ''}
        </div>
        <div class="prob-segment up" style="width: ${pUp}%;">
          ${pUp > 15 ? pUp + '%' : ''}
        </div>
      </div>

      <!-- Probability Cards Grid -->
      <div class="probability-cards-grid">
        <div class="prob-card down ${isDownHigh ? 'highest' : ''}">
          <div class="prob-name">DOWN (&lt; -1%)</div>
          <div class="prob-percent num">${pDown}%</div>
        </div>

        <div class="prob-card stable ${isStableHigh ? 'highest' : ''}">
          <div class="prob-name">STABLE ([-1%, +1%])</div>
          <div class="prob-percent num">${pStable}%</div>
        </div>

        <div class="prob-card up ${isUpHigh ? 'highest' : ''}">
          <div class="prob-name">UP (&gt; +1%)</div>
          <div class="prob-percent num">${pUp}%</div>
        </div>
      </div>
    </div>

    <!-- Metadata Details Card -->
    <div class="card">
      <div class="card-header">
        <h4 class="card-title" style="font-size: 0.95rem;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
          Inference &amp; Horizon Audit
        </h4>
        <button id="btn-save-paper" class="btn btn-secondary btn-sm">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
          <span>Log to Paper Portfolio</span>
        </button>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); font-size: 0.82rem;">
        <div>
          <span style="color: var(--text-muted);">Reference Timestamp:</span>
          <div class="num" style="color: var(--text-primary); font-weight: 600;">${data.reference_timestamp} IST</div>
        </div>
        <div>
          <span style="color: var(--text-muted);">Target Timestamp:</span>
          <div class="num" style="color: var(--text-primary); font-weight: 600;">${data.target_timestamp} IST</div>
        </div>
        <div>
          <span style="color: var(--text-muted);">Model Architecture:</span>
          <div style="color: var(--brand-accent); font-weight: 600;">${data.model_name}</div>
        </div>
        <div>
          <span style="color: var(--text-muted);">Model Version:</span>
          <div class="num" style="color: var(--text-primary); font-weight: 600;">v${data.model_version}</div>
        </div>
      </div>

      <div id="paper-save-status" style="margin-top: var(--space-3); font-size: 0.8rem; display: none;"></div>
    </div>
  `;

  // Attach paper log action handler
  const saveBtn = container.querySelector('#btn-save-paper');
  const saveStatus = container.querySelector('#paper-save-status');

  saveBtn.addEventListener('click', async () => {
    saveBtn.disabled = true;
    saveBtn.innerHTML = `<div class="spinner"></div><span>Logging...</span>`;

    try {
      const rec = await api.createPaperPrediction({
        symbol: data.symbol,
        reference_timestamp: data.reference_timestamp,
        target_timestamp: data.target_timestamp,
        model_type: data.model_name.includes('Random Forest') ? 'h1' : 'pooled'
      });

      saveStatus.innerHTML = `
        <div class="alert alert-info" style="margin-bottom: 0; padding: var(--space-2) var(--space-3);">
          <span>Logged to paper tracker as <strong>${rec.prediction_id}</strong> (Status: ${rec.status}).</span>
        </div>
      `;
      saveStatus.style.display = 'block';
    } catch (err) {
      saveStatus.innerHTML = `
        <div class="alert alert-danger" style="margin-bottom: 0; padding: var(--space-2) var(--space-3);">
          <span>Failed to log prediction: ${err.message}</span>
        </div>
      `;
      saveStatus.style.display = 'block';
    } finally {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>
        <span>Logged</span>
      `;
    }
  });
}
