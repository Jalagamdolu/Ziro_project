# Phase 6 — Paper Trading, Walk-Forward Evaluation & Production Interface Report

**Project**: NSE Intraday Stock Price Movement Prediction Using Machine Learning  
**Production Locked Champion**: **Logistic Regression (L2, $C=1.0$, `class_weight='balanced'`)**  
**Production Artifacts**: `models/best_pooled_model.joblib`, `models/pooled_preprocessor.joblib`  
**Primary Metric**: **Macro F1**  
**Operational Status**: **PAPER EVALUATION ONLY — NOT CONFIGURED FOR REAL-MONEY TRADING**  

---

## 1. System Architecture

The Phase 6 system establishes a fully simulated, forward-evaluation infrastructure around the validated Phase 5 machine learning model. It connects persistent paper storage, a RESTful inference API, an interactive governance dashboard, and continuous out-of-sample performance auditing:

```
                              ┌───────────────────────────────┐
                              │      Streamlit Dashboard      │
                              │     (app/streamlit_app.py)    │
                              └───────────────┬───────────────┘
                                              │ HTTP / Service Calls
                                              ▼
                              ┌───────────────────────────────┐
                              │        FastAPI Backend        │
                              │         (src/api.py)          │
                              └───────────────┬───────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     Paper Trading Engine      │                               │     Production Inference      │
│    (src/paper_trading.py)     │                               │       (src/predict.py)        │
└─────────────┬─────────────────┘                               └───────────────┬───────────────┘
              │                                                                 │
              ▼                                                                 ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     Performance Evaluator     │                               │   Locked Production Model     │
│     (src/performance.py)      │                               │    (Logistic Regression)      │
└─────────────┬─────────────────┘                               └───────────────────────────────┘
              │
              ▼
┌───────────────────────────────┐
│    paper_predictions Table    │
│   (PostgreSQL / TimescaleDB)  │
└───────────────────────────────┘
```

---

## 2. Two-Stage Prediction Lifecycle & Status State Machine

Predictions transition through an immutable, chronological lifecycle:

```
[User Request] 
      │
      ▼
┌───────────┐  Target Date Passed + Data Exists  ┌──────────────┐
│   OPEN    │ ─────────────────────────────────> │   RESOLVED   │
└─────┬─────┘                                    └──────────────┘
      │
      │ Target Date Passed + Data Missing        ┌──────────────┐
      └────────────────────────────────────────> │   EXPIRED    │
                                                 └──────────────┘
```

### Stage A — OPEN (Reference Timestamp $t_{\text{ref}}$)
1. User provides `symbol`, `reference_timestamp`, and `target_timestamp`.
2. Input parameters are strictly validated (symbol legitimacy, trading calendar, non-weekend, horizon $1 \le H \le 7$).
3. Features are generated using **strictly data available at or before $t_{\text{ref}}$**. Target OHLCV is **never queried**.
4. The locked production preprocessor and model compute class probabilities and movement classification.
5. Record is inserted into `paper_predictions` with `status = 'OPEN'`, `is_resolved = FALSE`, and `actual_price = NULL`.

### Stage B — RESOLVED (After Target Timestamp $t_{\text{tgt}}$)
1. Triggered during automated batch resolution (`POST /paper/resolve` or dashboard action).
2. The system checks whether $t_{\text{tgt}}$ has passed and market data exists.
3. Queries target close price at $t_{\text{tgt}}$ (`10:30:00 IST`).
4. Computes realized return:
   $$\text{future\_return\_pct} = \frac{\text{actual\_target\_close} - \text{reference\_price}}{\text{reference\_price}} \times 100\%$$
5. Determines actual class:
   $$\text{Class} = \begin{cases} \text{UP} & > +1.0\% \\ \text{DOWN} & < -1.0\% \\ \text{STABLE} & [-1.0\%, +1.0\%] \end{cases}$$
6. Updates record: `status = 'RESOLVED'`, `is_resolved = TRUE`, `resolved_at = NOW()`.
7. **EXPIRED Status**: If $t_{\text{tgt}}$ has passed but the target market candle is unavailable (e.g. trading suspension or data feed gap), status is marked `EXPIRED` with an explicit reason documented in `status_reason`.

---

## 3. Database Schema

The persistent table `paper_predictions` is hosted in PostgreSQL (`nse_minute`) and auto-initialized via SQLAlchemy:

```sql
CREATE TABLE IF NOT EXISTS paper_predictions (
    prediction_id VARCHAR(64) PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    reference_timestamp VARCHAR(32) NOT NULL,
    target_timestamp VARCHAR(32) NOT NULL,
    reference_price DOUBLE PRECISION NOT NULL,
    observed_sessions_ahead INTEGER NOT NULL,
    calendar_days_ahead INTEGER NOT NULL,
    predicted_class VARCHAR(16) NOT NULL,
    probability_down DOUBLE PRECISION NOT NULL,
    probability_stable DOUBLE PRECISION NOT NULL,
    probability_up DOUBLE PRECISION NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'OPEN',
    status_reason TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    actual_price DOUBLE PRECISION,
    future_return_pct DOUBLE PRECISION,
    actual_class VARCHAR(16),
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_paper_predictions_symbol ON paper_predictions (symbol);
CREATE INDEX IF NOT EXISTS idx_paper_predictions_status ON paper_predictions (status);
CREATE INDEX IF NOT EXISTS idx_paper_predictions_ref_time ON paper_predictions (reference_timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_predictions_target_time ON paper_predictions (target_timestamp);
```

**Immutability Guarantee**: Every prediction creates a distinct unique record (`pred_<uuid>`). Historical predictions are never overwritten.

---

## 4. Production REST API

Implemented in [src/api.py](file:///c:/Users/jalag/Ziro_project/src/api.py) using FastAPI:

| Method | Path | Description | Safety Enforcements |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck and model artifact status | Verifies model loaded |
| `POST` | `/predict` | Stateless prediction | Validates symbol, calendar, $H \le 7$ |
| `POST` | `/paper/predict` | Stateful paper prediction creation | Enforces zero target query, stores `OPEN` |
| `POST` | `/paper/resolve` | Batch resolution of all eligible `OPEN` predictions | Queries realized target price |
| `POST` | `/paper/resolve/{id}` | Single prediction resolution | Checks target date occurrence |
| `GET` | `/paper/predictions` | Filterable prediction audit trail | Query filters: symbol, status, horizon |
| `GET` | `/paper/performance` | Forward evaluation metrics & benchmark | Compares forward paper vs historical |

### Sample Payload (`POST /predict`):
```json
{
    "symbol": "360ONE",
    "reference_timestamp": "2026-09-10 10:30",
    "target_timestamp": "2026-09-15 10:30"
}
```

### Verified API Response (`200 OK`):
```json
{
    "symbol": "360ONE",
    "reference_timestamp": "2026-09-10 10:30",
    "target_timestamp": "2026-09-15 10:30",
    "reference_price": 1091.30,
    "observed_sessions_ahead": 3,
    "calendar_days_ahead": 5,
    "predicted_class": "DOWN",
    "probability_down": 0.5873,
    "probability_stable": 0.1930,
    "probability_up": 0.2197,
    "model_name": "Logistic Regression (L2, C=1.0, class_weight='balanced')",
    "model_version": "1.0.0"
}
```

---

## 5. Interactive Streamlit Dashboard

Implemented in [app/streamlit_app.py](file:///c:/Users/jalag/Ziro_project/app/streamlit_app.py).

### Global Security & Governance Header:
> `⚠️ PAPER EVALUATION ONLY — NOT CONFIGURED FOR REAL-MONEY TRADING`  
> *Zero broker or order routing functionality exists.*

### 3 Integrated Pages:
1. **Live Prediction**:
   - Equity symbol selector (valid canonical universe).
   - Reference & Target timestamp pickers.
   - Real-time reference price, calculated session horizon ($H$), and calendar span.
   - Dynamic class badge (UP: Green, DOWN: Red, STABLE: Slate) and calibrated probability progress bars.
   - Direct persistence checkbox to log as `OPEN` paper prediction.
2. **Paper Prediction History**:
   - Full immutable audit trail of paper predictions.
   - Interactive filtering by Symbol, Horizon ($H1$–$H7$), Status (`OPEN`, `RESOLVED`, `EXPIRED`), and Class.
   - **"🔄 Resolve Due Predictions"** action button to evaluate against realized prices.
3. **Performance & Risk Dashboard**:
   - Executive KPIs: Total, Resolved, Open, and Expired counts.
   - **Side-by-Side Model Governance**: Historical Out-of-Sample Test Benchmark vs Forward Paper Evaluation.
   - Horizon-wise performance table ($H1$ to $H7$) with explicit `NOT ENOUGH DATA` indicators.
   - Class-wise Precision, Recall, and F1 breakdown with Confusion Matrix.
   - Confidence calibration tiers (`0.50–0.60`, `0.60–0.70`, etc.).

---

## 6. Walk-Forward Simulation & Empirical Results

A walk-forward evaluation was executed across 10 liquid canonical symbols (`360ONE`, `TCS`, `INFY`, `RELIANCE`, `HDFCBANK`, `ICICIBANK`, `SBIN`, `BHARTIARTL`, `KOTAKBANK`, `LT`):
1. **Historical Test Window Simulation** (Ref: `2026-09-09 10:30` $\to$ Target: `2026-09-10 10:30`): Created $H=1$ paper predictions strictly using Sep 09 data, then resolved against realized Sep 10 data.
2. **Future Paper Ingestion** (Ref: `2026-09-10 10:30` $\to$ Target: `2026-09-11 10:30` and `2026-09-15 10:30`): Generated forward predictions beyond database boundaries. These remain in the `OPEN` state awaiting incoming market sessions.

### Current Database State Summary:
- **Total Paper Predictions Persisted**: **34**
- **Resolved Predictions**: **22**
- **Open Future Predictions**: **12**
- **Expired / Missing**: **0**

### Comparative Governance: Historical Test vs Forward Paper Performance

| Metric | Phase 5 Historical Test Set ($N=1,235$) | Phase 6 Forward Paper Evaluation ($N=22$) | Comparative Note |
| :--- | :---: | :---: | :--- |
| **Macro F1** | **0.3563** | **0.1926** | Forward sample reflects low-sample preliminary state |
| **Accuracy** | **46.40%** | **31.82%** | Forward sample dominated by DOWN movements |
| **Balanced Accuracy**| **38.69%** | **31.14%** | DOWN recall low due to strong STABLE bias |
| **DOWN Precision / Recall**| 35.4% / **51.44%** | 50.0% / **7.69%** | Forward window experienced unexpected negative shocks |
| **STABLE Precision / Recall**| 56.0% / **61.03%** | 30.0% / **85.71%** | Model predicted STABLE for 20 of 22 forward cases |
| **UP Precision / Recall** | 36.0% / **3.59%** | 0.0% / **0.00%** | Re-confirms weak UP detection identified in Phase 5 |

---

## 7. Performance by Horizon Breakdown (H1 to H7)

| Horizon | Forward Sample Count | Status | Accuracy | Macro F1 | DOWN Recall | STABLE Recall | UP Recall | Operational Diagnostic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **H1** | **22** | **EVALUATED** | **0.3182** | **0.1926** | 0.0769 | 0.8571 | 0.0000 | Evaluated on resolved Sep 09 $\to$ Sep 10 paper records |
| **H2** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |
| **H3** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |
| **H4** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |
| **H5** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |
| **H6** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |
| **H7** | 0 | **NOT ENOUGH DATA** | — | — | — | — | — | Requires Live Market Session Ingestion |

*Rule strictly enforced: Zero metrics fabricated for horizons with insufficient observations.*

---

## 8. Leakage Controls & Isolation Verification

Rigorous programmatic isolation was verified:
1. **Prediction Isolation**: During `create_paper_prediction()`, SQLAlchemy event listeners intercept every executed SQL statement. The OHLCV query is hard-constrained to `DATE = ref_date` and `time <= ref_time`.
2. **Zero Target Access**: The target date is never queried, checked, or referenced during prediction generation.
3. **Decoupled Resolution**: Resolution is executed strictly as an asynchronous second stage (`resolve_prediction()`), which queries only after $t_{\text{tgt}}$ has occurred.

---

## 9. Automated Test Suite Results

The comprehensive test suite in `tests/` was executed via `pytest tests/ -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-8.4.2
collected 14 items

tests/test_api.py::test_api_health PASSED                                [  7%]
tests/test_api.py::test_api_predict_success PASSED                       [ 14%]
tests/test_api.py::test_api_reject_invalid_symbol PASSED                 [ 21%]
tests/test_api.py::test_api_reject_ref_after_target PASSED               [ 28%]
tests/test_api.py::test_api_reject_weekend_target PASSED                 [ 35%]
tests/test_api.py::test_api_reject_unsupported_horizon PASSED            [ 42%]
tests/test_api.py::test_api_paper_predict_and_resolve_lifecycle PASSED   [ 50%]
tests/test_api.py::test_api_paper_performance PASSED                     [ 57%]
tests/test_future_target_inference.py::test_future_target_inference PASSED [ 64%]
tests/test_paper_trading.py::test_paper_prediction_creation_open_state PASSED [ 71%]
tests/test_paper_trading.py::test_paper_prediction_resolution_lifecycle PASSED [ 78%]
tests/test_paper_trading.py::test_prediction_immutability_new_record_each_time PASSED [ 85%]
tests/test_paper_trading.py::test_expired_prediction_when_market_data_missing PASSED [ 92%]
tests/test_paper_trading.py::test_leakage_isolation_during_paper_prediction PASSED [100%]

======================= 14 passed, 2 warnings in 24.18s =======================
```

**Result: 14/14 tests passed (100% success rate).**

---

## 10. Operational Limitations & Risk Warnings

1. **Paper Evaluation Only**: The system operates exclusively in paper evaluation mode. Broker execution, exchange routing, order book queues, slippage, and impact costs are intentionally omitted.
2. **Weak UP Detection**: The model exhibits near-zero sensitivity to UP movements (~3.6% in test, 0.0% in preliminary forward batch). It must not be deployed as an unhedged long trading strategy.
3. **Low Forward Sample Size**: While $N=1,235$ historical test pairs were evaluated in Phase 5, the live forward paper store currently has $N=22$ resolved observations. True forward calibration metrics require hundreds of forward sessions.
4. **Horizon Coverage**: Horizons $H=2$ through $H=7$ require forward trading days to resolve before multi-day claims can be validated.
