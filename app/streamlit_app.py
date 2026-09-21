"""
Streamlit Production Dashboard for NSE Intraday Stock Movement Prediction.

Features:
- Global Banner: PAPER EVALUATION ONLY — NOT CONFIGURED FOR REAL-MONEY TRADING
- Page 1: Live Movement Prediction & Model Inference
- Page 2: Paper Prediction History & Interactive Resolution
- Page 3: Performance & Risk Dashboard (Forward Paper vs Historical Test Benchmark)
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    predict_movement,
    get_observed_sessions_between,
    PreprocessingPipeline,
    ensure_pipeline_registered
)

# Ensure custom unpickler class is present in Streamlit runner namespace
ensure_pipeline_registered()
if __name__ in sys.modules:
    setattr(sys.modules[__name__], 'PreprocessingPipeline', PreprocessingPipeline)
from src.paper_trading import (
    create_paper_prediction,
    resolve_all_pending,
    list_predictions,
    get_legitimate_symbols
)
from src.performance import (
    generate_paper_performance_summary,
    HISTORICAL_TEST_BENCHMARK,
    CLASS_LABELS
)

# Page configuration
st.set_page_config(
    page_title="NSE Intraday Movement — Paper Evaluation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and dark/light contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .safety-banner {
        background: linear-gradient(90deg, #FEF2F2 0%, #FFFBEB 100%);
        border-left: 5px solid #EF4444;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #991B1B;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-up {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-down {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
    .badge-stable {
        background-color: #E2E8F0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Mandatory Safety Banner
st.markdown("""
<div class="safety-banner">
    ⚠️ <strong>PAPER EVALUATION ONLY — NOT CONFIGURED FOR REAL-MONEY TRADING</strong><br>
    <span style="font-weight: 400; font-size: 0.85rem; color: #7F1D1D;">
    This dashboard provides simulated machine learning movement forecasts and paper evaluation tracking. 
    Zero real-money broker or order routing functionality exists.
    </span>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Workflow",
    ["1. Live Prediction", "2. Paper Prediction History", "3. Performance Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Locked Production Model:**")
st.sidebar.caption("Logistic Regression (L2, C=1.0, balanced)")
st.sidebar.markdown("**Target Definition:**")
st.sidebar.caption("UP (> +1%), DOWN (< -1%), STABLE ([-1%, +1%])")
st.sidebar.markdown("**Supported Scope:**")
st.sidebar.caption("1 to 7 observed trading sessions")

# ==============================================================================
# PAGE 1: LIVE PREDICTION
# ==============================================================================
if page == "1. Live Prediction":
    st.markdown('<div class="main-header">Real-Time Movement Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate calibrated movement forecasts using market data strictly up to reference time.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Inference Parameters")
        
        # Load sample legitimate symbols for dropdown/autofill
        try:
            legit_symbols = sorted(list(get_legitimate_symbols()))[:200]
        except Exception:
            legit_symbols = ["360ONE", "RELIANCE", "TCS", "INFY", "HDFCBANK"]
            
        symbol_input = st.selectbox(
            "Stock Symbol (NSE Equity)",
            options=["360ONE"] + [s for s in legit_symbols if s != "360ONE"],
            index=0
        )
        
        ref_time_input = st.text_input(
            "Reference Timestamp (IST)",
            value="2026-09-10 10:30",
            help="Reference time. Only market data up to this timestamp is loaded."
        )
        
        target_time_input = st.text_input(
            "Target Timestamp (IST)",
            value="2026-09-15 10:30",
            help="Future target time. Never queried during prediction."
        )
        
        model_choice = st.selectbox(
            "Model Architecture",
            options=["pooled (Production Locked)", "h1 (Dedicated 1-Day Specialist)"],
            index=0
        )
        
        save_to_paper = st.checkbox("Persist as OPEN Paper Prediction Record", value=True)
        predict_btn = st.button("Generate Prediction", type="primary", use_container_width=True)

    with col2:
        st.subheader("Forecast Output")
        
        if predict_btn:
            try:
                model_type = "h1" if "h1" in model_choice else "pooled"
                
                with st.spinner("Loading reference market features and generating prediction..."):
                    if save_to_paper:
                        res = create_paper_prediction(
                            symbol=symbol_input,
                            reference_timestamp=ref_time_input,
                            target_timestamp=target_time_input,
                            model_type=model_type
                        )
                        st.success(f"Prediction generated and persisted with ID: `{res['prediction_id']}` (Status: `OPEN`)")
                    else:
                        res = predict_movement(
                            symbol=symbol_input,
                            reference_timestamp=ref_time_input,
                            target_timestamp=target_time_input,
                            model_type=model_type
                        )
                        st.info("Stateless prediction generated (not stored).")

                # Metrics cards
                st.markdown("---")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Reference Price", f"₹{res['reference_price']:.2f}")
                with mc2:
                    obs_sessions = res.get('observed_sessions_ahead', res.get('target_horizon_observed_sessions'))
                    st.metric("Session Horizon", f"{obs_sessions} sessions")
                with mc3:
                    cal_days = res.get('calendar_days_ahead', res.get('target_horizon_calendar_days'))
                    st.metric("Calendar Span", f"{cal_days} days")

                # Predicted class badge
                pred_cls = res['predicted_class']
                if pred_cls == "UP":
                    badge_html = '<span class="badge-up" style="font-size: 1.3rem;">PREDICTED: UP (&gt; +1%)</span>'
                elif pred_cls == "DOWN":
                    badge_html = '<span class="badge-down" style="font-size: 1.3rem;">PREDICTED: DOWN (&lt; -1%)</span>'
                else:
                    badge_html = '<span class="badge-stable" style="font-size: 1.3rem;">PREDICTED: STABLE ([-1%, +1%])</span>'
                    
                st.markdown(f"<div style='margin: 1.2rem 0;'>{badge_html}</div>", unsafe_allow_html=True)

                # Probability bars
                st.markdown("#### Calibrated Probabilities")
                p_down = res['probability_down']
                p_stable = res['probability_stable']
                p_up = res['probability_up']
                
                st.write(f"**DOWN (< -1%):** {p_down * 100:.1f}%")
                st.progress(min(1.0, max(0.0, p_down)))
                
                st.write(f"**STABLE ([-1%, +1%]):** {p_stable * 100:.1f}%")
                st.progress(min(1.0, max(0.0, p_stable)))
                
                st.write(f"**UP (> +1%):** {p_up * 100:.1f}%")
                st.progress(min(1.0, max(0.0, p_up)))

                st.caption(f"Model: {res.get('model_name', 'Logistic Regression L2')} | Version: {res.get('model_version', '1.0.0')}")
                st.caption("Verification: Zero target OHLCV was accessed during feature generation or inference.")

            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
        else:
            st.info("Configure parameters on the left and click **Generate Prediction**.")

# ==============================================================================
# PAGE 2: PAPER PREDICTION HISTORY
# ==============================================================================
elif page == "2. Paper Prediction History":
    st.markdown('<div class="main-header">Paper Prediction Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Immutable record of all forward paper predictions and lifecycle transitions.</div>', unsafe_allow_html=True)

    # Resolution action bar
    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        st.write("Target timestamps that have passed can be verified against realized market data.")
    with r_col2:
        if st.button("🔄 Resolve Due Predictions", use_container_width=True, type="secondary"):
            with st.spinner("Checking market data and resolving pending predictions..."):
                counts = resolve_all_pending()
                st.success(f"Processed: {counts['total_checked']} | Resolved: {counts['resolved']} | Expired: {counts['expired']} | Still Open: {counts['still_open']}")

    # Filters
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sym_filter = st.text_input("Filter by Symbol", value="")
    with f2:
        status_filter = st.selectbox("Status", options=["ALL", "OPEN", "RESOLVED", "EXPIRED"], index=0)
    with f3:
        horizon_filter = st.selectbox("Horizon", options=["ALL", "1", "2", "3", "4", "5", "6", "7"], index=0)
    with f4:
        class_filter = st.selectbox("Predicted Class", options=["ALL", "DOWN", "STABLE", "UP"], index=0)

    # Query predictions
    s_arg = sym_filter if sym_filter else None
    stat_arg = None if status_filter == "ALL" else status_filter
    h_arg = None if horizon_filter == "ALL" else int(horizon_filter)
    cls_arg = None if class_filter == "ALL" else class_filter

    preds = list_predictions(symbol=s_arg, status=stat_arg, horizon=h_arg, predicted_class=cls_arg, limit=500)

    if not preds:
        st.info("No paper predictions found matching current filters.")
    else:
        df_display = pd.DataFrame(preds)
        
        # Format display columns
        cols_to_show = [
            'prediction_id', 'symbol', 'reference_timestamp', 'target_timestamp',
            'reference_price', 'observed_sessions_ahead', 'predicted_class',
            'probability_down', 'probability_stable', 'probability_up',
            'status', 'actual_price', 'future_return_pct', 'actual_class'
        ]
        available_cols = [c for c in cols_to_show if c in df_display.columns]
        df_sub = df_display[available_cols].copy()
        
        # Round floats
        if 'future_return_pct' in df_sub.columns:
            df_sub['future_return_pct'] = df_sub['future_return_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "—")
        if 'actual_price' in df_sub.columns:
            df_sub['actual_price'] = df_sub['actual_price'].apply(lambda x: f"₹{x:.2f}" if pd.notnull(x) else "—")
            
        st.dataframe(df_sub, use_container_width=True)
        st.caption(f"Showing {len(df_sub)} predictions. Records are strictly immutable.")

# ==============================================================================
# PAGE 3: PERFORMANCE DASHBOARD
# ==============================================================================
elif page == "3. Performance Dashboard":
    st.markdown('<div class="main-header">Forward Paper Performance & Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Continuous out-of-sample monitoring, horizon breakdowns, and historical benchmark comparison.</div>', unsafe_allow_html=True)

    summary = generate_paper_performance_summary()

    # KPI Row
    st.markdown("### Lifecycle Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Paper Predictions", summary['total_predictions'])
    with k2:
        st.metric("Resolved Predictions", summary['resolved_predictions'])
    with k3:
        st.metric("Open Predictions", summary['open_predictions'])
    with k4:
        st.metric("Expired / Missing", summary['expired_predictions'])

    st.markdown("---")

    # Comparison Section: Historical Baseline vs Forward Paper Performance
    st.markdown("### Model Governance: Historical Test Benchmark vs Forward Paper Performance")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("#### Phase 5 Historical Out-of-Sample Test Benchmark")
        st.caption("Chronological test set (Sep 09–10, 2026, N=1,235). Static evaluation.")
        hist = summary['historical_benchmark']
        st.markdown(f"""
        - **Macro F1**: `{hist['macro_f1']:.4f}`
        - **Accuracy**: `{hist['accuracy'] * 100:.2f}%`
        - **Balanced Accuracy**: `{hist['balanced_accuracy'] * 100:.2f}%`
        - **DOWN Recall**: `{hist['down_recall'] * 100:.2f}%`
        - **STABLE Recall**: `{hist['stable_recall'] * 100:.2f}%`
        - **UP Recall**: `{hist['up_recall'] * 100:.2f}%` (⚠️ Severe limitation)
        """)

    with comp_col2:
        st.markdown("#### Phase 6 Forward Paper Evaluation")
        fwd = summary['forward_paper_metrics']
        
        if fwd['status'] == "NO_RESOLVED_DATA":
            st.warning("No resolved forward paper predictions recorded yet. Forward metrics will populate as predictions resolve.")
        else:
            st.caption(f"Live forward tracking across N={fwd['sample_count']} resolved predictions.")
            st.markdown(f"""
            - **Macro F1**: `{fwd['macro_f1']:.4f}`
            - **Accuracy**: `{fwd['accuracy'] * 100:.2f}%`
            - **Balanced Accuracy**: `{fwd['balanced_accuracy'] * 100:.2f}%`
            - **Macro Precision**: `{fwd['macro_precision']:.4f}`
            - **Macro Recall**: `{fwd['macro_recall']:.4f}`
            """)

    st.markdown("---")

    # Horizon-wise breakdown
    st.markdown("### Performance by Horizon Breakdown (H1 to H7)")
    st.caption("The pooled model supports horizons 1–7. Empty or low-sample horizons are explicitly marked as NOT ENOUGH DATA.")
    
    df_horizons = pd.DataFrame(summary['by_horizon'])
    st.dataframe(df_horizons, use_container_width=True)

    # Class-wise performance if resolved data exists
    if fwd['status'] != "NO_RESOLVED_DATA" and fwd.get('per_class'):
        st.markdown("### Class-Wise Performance (Resolved Forward Predictions)")
        df_classes = pd.DataFrame.from_dict(fwd['per_class'], orient='index')
        st.dataframe(df_classes, use_container_width=True)

        if fwd.get('confusion_matrix'):
            st.markdown("### Confusion Matrix")
            cm_data = fwd['confusion_matrix']['matrix']
            df_cm = pd.DataFrame(cm_data, index=[f"Actual {c}" for c in CLASS_LABELS], columns=[f"Pred {c}" for c in CLASS_LABELS])
            st.dataframe(df_cm, use_container_width=True)

    # Confidence Buckets
    if summary.get('confidence_tiers'):
        st.markdown("### Confidence Analysis")
        st.caption("Distribution of model predictions by maximum predicted probability.")
        df_conf = pd.DataFrame(summary['confidence_tiers'])
        st.dataframe(df_conf, use_container_width=True)

    # Real-World Limitations
    st.markdown("---")
    st.markdown("### Documented Real-World Limitations & Risk Factors")
    st.warning("""
    1. **Weak UP-Class Recall**: The model exhibits very weak recall on bullish movements (~3.6% on test) and must not be used as an unhedged long directional signal.
    2. **Multi-Session Out-of-Sample Coverage**: H2 through H7 require further market sessions to accumulate statistically significant forward validation samples.
    3. **Paper Trading Only**: This environment operates in simulated paper evaluation mode. Slippage, liquidity, impact cost, and order queues are not simulated.
    """)
