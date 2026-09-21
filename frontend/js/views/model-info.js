/**
 * Ziro Predict - Model Information & Governance View Controller
 */

export function renderModelInfo(container) {
  container.innerHTML = `
    <div class="model-info-view">
      <!-- Architecture Overview -->
      <div class="grid-2" style="margin-bottom: var(--space-6);">
        <div class="card">
          <div class="card-header">
            <div>
              <span class="badge" style="background: rgba(2, 132, 199, 0.2); color: var(--brand-accent); margin-bottom: 4px;">PRODUCTION</span>
              <h3 class="card-title">Locked Production Model</h3>
              <span class="card-subtitle">Global multi-horizon classifier (H1 to H7)</span>
            </div>
          </div>

          <div style="display: flex; flex-direction: column; gap: var(--space-3); font-size: 0.88rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Algorithm:</span>
              <strong style="color: var(--text-primary);">Logistic Regression</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Regularization:</span>
              <strong style="color: var(--text-primary);">L2 Penalty (Ridge)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Inverse Regularization Strength (C):</span>
              <strong class="num" style="color: var(--text-primary);">1.0</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Class Weighting:</span>
              <strong style="color: var(--brand-accent);">'balanced' (Inverse Frequency)</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: var(--text-muted);">Target Horizon Scope:</span>
              <strong style="color: var(--text-primary);">1 to 7 observed trading sessions</strong>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div>
              <span class="badge" style="background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); margin-bottom: 4px;">SPECIALIZED</span>
              <h3 class="card-title">Dedicated H1 Model</h3>
              <span class="card-subtitle">Specialized single-session classifier (H1 Only)</span>
            </div>
          </div>

          <div style="display: flex; flex-direction: column; gap: var(--space-3); font-size: 0.88rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Algorithm:</span>
              <strong style="color: var(--text-primary);">Random Forest Classifier</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Maximum Depth:</span>
              <strong class="num" style="color: var(--text-primary);">6</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Min Samples Leaf:</span>
              <strong class="num" style="color: var(--text-primary);">15</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 6px;">
              <span style="color: var(--text-muted);">Class Weighting:</span>
              <strong style="color: var(--brand-accent);">'balanced'</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: var(--text-muted);">Application:</span>
              <strong style="color: var(--text-primary);">1-session ahead predictions only</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- Target Definition -->
      <div class="card" style="margin-bottom: var(--space-6);">
        <div class="card-header">
          <h3 class="card-title">Target Classification Definition</h3>
          <span class="card-subtitle">Formula: ((Target Close - Ref Close) / Ref Close) * 100</span>
        </div>

        <div class="grid-3" style="margin-top: var(--space-4);">
          <div class="card" style="background-color: var(--bg-surface); border-left: 3px solid var(--color-down);">
            <span class="badge-direction down" style="margin-bottom: var(--space-2);">DOWN</span>
            <div class="metric-val num" style="font-size: 1.2rem; color: var(--color-down);">&lt; -1.0%</div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: var(--space-1);">
              Stock price decreases by more than 1.0% between reference and target session close.
            </p>
          </div>

          <div class="card" style="background-color: var(--bg-surface); border-left: 3px solid var(--color-stable);">
            <span class="badge-direction stable" style="margin-bottom: var(--space-2);">STABLE</span>
            <div class="metric-val num" style="font-size: 1.2rem; color: var(--color-stable);">[-1.0%, +1.0%]</div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: var(--space-1);">
              Stock price remains bounded within ±1.0% (consolidation range).
            </p>
          </div>

          <div class="card" style="background-color: var(--bg-surface); border-left: 3px solid var(--color-up);">
            <span class="badge-direction up" style="margin-bottom: var(--space-2);">UP</span>
            <div class="metric-val num" style="font-size: 1.2rem; color: var(--color-up);">&gt; +1.0%</div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: var(--space-1);">
              Stock price increases by more than 1.0% between reference and target session close.
            </p>
          </div>
        </div>
      </div>

      <!-- 26 Model Features -->
      <div class="card" style="margin-bottom: var(--space-6);">
        <div class="card-header">
          <div>
            <h3 class="card-title">26 Model Features</h3>
            <span class="card-subtitle">Source of truth extracted strictly up to reference timestamp (zero lookahead leakage)</span>
          </div>
        </div>

        <!-- Category 1: Returns -->
        <div class="feature-category">
          <h4>1. Return Signals (5 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">return_1m</span>
            <span class="feature-pill">return_5m</span>
            <span class="feature-pill">return_15m</span>
            <span class="feature-pill">return_30m</span>
            <span class="feature-pill">return_60m</span>
          </div>
        </div>

        <!-- Category 2: Missing Return Indicators -->
        <div class="feature-category">
          <h4>2. Missing Return Indicators (5 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">return_1m_missing</span>
            <span class="feature-pill">return_5m_missing</span>
            <span class="feature-pill">return_15m_missing</span>
            <span class="feature-pill">return_30m_missing</span>
            <span class="feature-pill">return_60m_missing</span>
          </div>
        </div>

        <!-- Category 3: Price & MA Technicals -->
        <div class="feature-category">
          <h4>3. Price &amp; Moving Average Technicals (4 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">price_range_pct</span>
            <span class="feature-pill">distance_from_ma_5</span>
            <span class="feature-pill">distance_from_ma_15</span>
            <span class="feature-pill">distance_from_ma_30</span>
          </div>
        </div>

        <!-- Category 4: Rolling Volatilities -->
        <div class="feature-category">
          <h4>4. Rolling Volatilities (4 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">volatility_5m</span>
            <span class="feature-pill">volatility_15m</span>
            <span class="feature-pill">volatility_30m</span>
            <span class="feature-pill">volatility_60m</span>
          </div>
        </div>

        <!-- Category 5: Volume Dynamics -->
        <div class="feature-category">
          <h4>5. Volume Dynamics (3 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">volume_ratio</span>
            <span class="feature-pill">log_volume_change</span>
            <span class="feature-pill">prev_vol_zero_flag</span>
          </div>
        </div>

        <!-- Category 6: Intraday Session Context -->
        <div class="feature-category">
          <h4>6. Intraday Session Context (3 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">price_change_from_open</span>
            <span class="feature-pill">session_range_pct</span>
            <span class="feature-pill">day_of_week</span>
          </div>
        </div>

        <!-- Category 7: Horizon Context -->
        <div class="feature-category">
          <h4>7. Forecast Horizon Context (2 features)</h4>
          <div class="feature-pill-container">
            <span class="feature-pill">observed_sessions_ahead</span>
            <span class="feature-pill">calendar_days_ahead</span>
          </div>
        </div>
      </div>

      <!-- Preprocessing Pipeline -->
      <div class="card" style="margin-bottom: var(--space-6);">
        <div class="card-header">
          <h3 class="card-title">Leakage-Safe Preprocessing Pipeline</h3>
          <span class="card-subtitle">Fitted strictly on 4,824 Clean Training samples</span>
        </div>

        <div style="display: flex; flex-direction: column; gap: var(--space-3); font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6;">
          <p>
            1. <strong>SimpleImputer(strategy='median')</strong>: Imputes missing return values using strictly training median statistics.
          </p>
          <p>
            2. <strong>Winsorization Clipping</strong>: Clips outlier feature values between training 1st and 99th percentiles to eliminate extreme intraday spikes.
          </p>
          <p>
            3. <strong>RobustScaler()</strong>: Centers data by median and scales by Interquartile Range (IQR), preserving resistance against outliers.
          </p>
        </div>
      </div>

      <!-- Risk Disclaimers -->
      <div class="alert alert-warning">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <div>
          <strong style="color: #fde68a;">Documented Real-World Limitations &amp; Risk Factors:</strong>
          <ul style="margin-top: 6px; padding-left: 18px; font-size: 0.82rem; color: #fcd34d; line-height: 1.5;">
            <li><strong>Weak UP-Class Recall</strong>: The model exhibits weak recall on bullish movements (~3.6% on test set) and must not be used as an unhedged long directional signal.</li>
            <li><strong>Movement Classification Only</strong>: This system predicts categorical directional movement (UP, DOWN, STABLE). It does NOT predict guaranteed stock prices.</li>
            <li><strong>Paper Evaluation Only</strong>: Real-world execution factors (slippage, bid-ask spread, liquidity constraints, order queue priority) are not simulated.</li>
          </ul>
        </div>
      </div>
    </div>
  `;
}
