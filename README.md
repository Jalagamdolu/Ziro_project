# NSE Intraday Stock Price Movement Prediction & Paper Trading System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![PostgreSQL / TimescaleDB](https://img.shields.io/badge/Database-PostgreSQL%2FTimescaleDB-blue.svg)](https://www.timescale.com/)

An end-to-end, leakage-audited machine learning system for forecasting multi-day intraday price movements on National Stock Exchange (NSE) equities. Built with rigorous temporal purging, a production-grade inference engine, a stateful paper trading lifecycle, a RESTful FastAPI backend, and an interactive Streamlit governance dashboard.

> ⚠️ **IMPORTANT COMPLIANCE & SAFETY NOTICE**  
> **PAPER EVALUATION ONLY — NOT CONFIGURED FOR REAL-MONEY TRADING**  
> This system provides simulated machine learning movement forecasts and forward paper trading evaluation. Zero broker integration, exchange order routing, or capital execution functionality exists.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Key Features](#key-features)
3. [Project Directory Structure](#project-directory-structure)
4. [Installation & Setup](#installation--setup)
5. [Running the System](#running-the-system)
   - [FastAPI REST Interface](#1-fastapi-rest-interface)
   - [Streamlit Governance Dashboard](#2-streamlit-governance-dashboard)
   - [Automated Test Suite](#3-automated-test-suite)
6. [REST API Documentation](#rest-api-documentation)
7. [Streamlit Dashboard Overview](#streamlit-dashboard-overview)
8. [Model Governance & Performance Benchmark](#model-governance--performance-benchmark)
9. [Important Operational Limitations](#important-operational-limitations)

---

## System Architecture

```
                               ┌────────────────────────────────┐
                               │      Streamlit Dashboard       │
                               │     (app/streamlit_app.py)     │
                               └───────────────┬────────────────┘
                                               │ Service Calls
                                               ▼
                               ┌────────────────────────────────┐
                               │        FastAPI Backend         │
                               │          (src/api.py)          │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     Paper Trading Engine      │                               │     Production Inference      │
│    (src/paper_trading.py)     │                               │       (src/predict.py)        │
└──────────────┬────────────────┘                               └───────────────┬───────────────┘
               │                                                                │
               ▼                                                                ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     Performance Evaluator     │                               │   Locked Production Model     │
│     (src/performance.py)      │                               │    (Logistic Regression)      │
└──────────────┬────────────────┘                               └───────────────────────────────┘
               │
               ▼
┌───────────────────────────────┐
│    paper_predictions Table    │
│   (PostgreSQL / TimescaleDB)  │
└───────────────────────────────┘
```

---

## Key Features

- **Strict Leakage Prevention**: Features are extracted strictly at or before `10:30:00 IST` on the reference date. Target timestamps and prices are never queried during inference.
- **3-Class Movement Target**:
  - **`UP`**: Future return $> +1.0\%$
  - **`DOWN`**: Future return $< -1.0\%$
  - **`STABLE`**: Future return $[-1.0\%, +1.0\%]$
- **Multi-Horizon Coverage (H1 to H7)**: Supports forecasts across 1 to 7 observed trading sessions ahead (with business calendar calculation and weekend rejection).
- **Two-Stage Paper Trading Lifecycle**:
  - `OPEN`: Generated at reference time, logged immutably with zero actual price access.
  - `RESOLVED`: Evaluated against realized market data only after target timestamp has passed.
  - `EXPIRED`: Handled gracefully if target market candle was unavailable.
- **Purged Temporal Split Protocol**: Eliminates multi-day target overlap between training, validation, and out-of-sample test splits.
- **Honest Model Governance**: Evaluated strictly on **Macro F1** to prevent false confidence from non-stationary regime shifts.

---

## Project Directory Structure

```
Ziro_project/
├── .env.example                     # Environment template for DB credentials
├── .gitignore                        # Git exclusion rules (blocks credentials and binary data)
├── pytest.ini                        # Pytest configuration (targets tests/)
├── requirements.txt                  # Production and test dependencies
├── README.md                         # Project documentation and user guide
├── app/
│   └── streamlit_app.py              # 3-Page Streamlit governance dashboard
├── data/                             # Curated feature datasets (Parquet)
├── figures/                          # 8 EDA distribution and correlation figures (PNG)
├── models/
│   ├── best_pooled_model.joblib      # Production Champion (Logistic Regression L2)
│   ├── pooled_preprocessor.joblib    # Preprocessing Pipeline (Imputer + Winsorizer + Scaler)
│   ├── best_h1_model.joblib          # Dedicated H1 Specialist (Random Forest)
│   ├── h1_preprocessor.joblib        # Dedicated H1 Preprocessing Pipeline
│   └── model_metadata.json           # Model specification, feature lists, and metrics
├── reports/                          # Analytical summaries, CSV tables, and phase reports
├── scratch/                          # Simulation runners and standalone scripts
├── src/
│   ├── __init__.py
│   ├── api.py                        # FastAPI REST service
│   ├── config.py                     # Global constants, canonical source, and thresholds
│   ├── data_quality.py               # Assertion suite for leakage and data integrity
│   ├── eda.py                        # 11-step reproducible EDA pipeline
│   ├── feature_engineering.py        # 26 microstructure and session features
│   ├── horizon_analysis.py           # Sample feasibility and purged split design
│   ├── paper_trading.py              # Lifecycle state machine and persistence layer
│   ├── performance.py                # Out-of-sample evaluation and benchmark comparisons
│   ├── predict.py                    # Production inference engine
│   ├── target_creation.py            # Exact timestamp target pair generator
│   └── train_classification.py       # Model training, baselines, and ablation suite
└── tests/
    ├── test_api.py                   # REST endpoint and schema validation tests
    ├── test_future_target_inference.py # Zero-target query and SQL interception tests
    └── test_paper_trading.py         # Lifecycle and immutability validation tests
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- PostgreSQL / TimescaleDB instance running on port 5433 (or SQLite fallback)

### 2. Clone the Repository
```bash
git clone https://github.com/Jalagamdolu/Ziro_project.git
cd Ziro_project
```

### 3. Create a Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and configure your database credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
DATABASE_URL=postgresql+psycopg://nse_user:nse_password@localhost:5433/nse_minute
```

---

## Running the System

### 1. FastAPI REST Interface
Start the backend service:
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
- API Root: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 2. Streamlit Governance Dashboard
Launch the interactive web UI:
```bash
streamlit run app/streamlit_app.py
```
- Dashboard URL: `http://localhost:8501`

### 3. Automated Test Suite
Execute the comprehensive test suite:
```bash
pytest tests/ -v
```

---

## REST API Documentation

| Method | Endpoint | Description | Key Enforcements |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health and model verification | Validates model artifact load |
| `POST` | `/predict` | Stateless movement prediction | Enforces zero target query, $1 \le H \le 7$ |
| `POST` | `/paper/predict` | Stateful prediction logged to DB | Creates `OPEN` record with unique ID |
| `POST` | `/paper/resolve` | Batch resolution of due predictions | Evaluates against realized market data |
| `POST` | `/paper/resolve/{id}` | Resolves a single prediction by ID | Verifies target timestamp occurrence |
| `GET` | `/paper/predictions` | Filterable prediction audit trail | Filters: symbol, status, horizon, class |
| `GET` | `/paper/predictions/{id}` | Detailed single prediction lookup | Returns timestamps, probabilities, actuals |
| `GET` | `/paper/performance` | Forward paper performance summary | Compares forward paper vs historical test |

### Sample Prediction Request (`POST /predict`):
```json
{
  "symbol": "360ONE",
  "reference_timestamp": "2026-09-10 10:30",
  "target_timestamp": "2026-09-15 10:30",
  "model_type": "pooled"
}
```

### Sample Prediction Response (`200 OK`):
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

## Streamlit Dashboard Overview

The dashboard contains 3 integrated workflows:

1. **Page 1: Live Prediction**
   - Interactive symbol selector across canonical NSE equity universe.
   - Reference and target timestamp pickers with calendar validation.
   - Visual movement class badges (Green for `UP`, Red for `DOWN`, Slate for `STABLE`).
   - Calibrated probability progress bars.
   - Option to persist prediction directly as an `OPEN` paper record.

2. **Page 2: Paper Prediction History**
   - Immutable audit trail of all forward paper predictions.
   - Dynamic filters by Symbol, Status (`OPEN`, `RESOLVED`, `EXPIRED`), Horizon, and Predicted Class.
   - Interactive **"🔄 Resolve Due Predictions"** action button.

3. **Page 3: Performance & Risk Dashboard**
   - Executive KPIs: Total, Resolved, Open, and Expired predictions.
   - Side-by-side model governance: Historical Out-of-Sample Test Benchmark vs Forward Paper Evaluation.
   - Horizon breakdown table (H1 to H7) with explicit `NOT ENOUGH DATA` indicators.
   - Class-wise Precision, Recall, F1 breakdown and Confusion Matrix.
   - Documented operational limitations and risk warnings.

---

## Model Governance & Performance Benchmark

All models were evaluated on chronological splits with multi-day overlap purged:

### Model Comparison Table (Validation Selection vs Out-of-Sample Test)

| Model Architecture | Hyperparameters | Val Macro F1 (Selection Metric) | Val Balanced Acc | Val Acc | Test Macro F1 (Unseen Test) | Test Balanced Acc | Test Acc | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | L2, $C=1.0$, balanced | **0.3462** | 0.3671 | 0.4178 | **0.3563** | 0.3869 | 0.4640 | **LOCKED CHAMPION** |
| **Logistic Regression** | L2, $C=0.1$, balanced | **0.3305** | 0.3564 | 0.3828 | **0.3661** | 0.3868 | 0.4713 | Alternative |
| **Random Forest** | depth=8, trees=150 | **0.2687** | 0.4076 | 0.2635 | **0.3739** | 0.3923 | 0.4753 | Comparative Only |
| **Extra Trees** | depth=8, trees=150 | **0.2664** | 0.4194 | 0.2664 | **0.3635** | 0.3903 | 0.4972 | Comparative Only |
| **XGBoost** | depth=4, lr=0.05 | **0.2427** | 0.3934 | 0.2489 | **0.3531** | 0.3804 | 0.4818 | Comparative Only |

*Note: In strict machine learning methodology, the test set must never influence model selection. Logistic Regression was locked based on Validation Macro F1 = 0.3462.*

---

## Important Operational Limitations

1. **Weak UP-Class Recall (Severe Directional Limitation)**:
   - On the out-of-sample Test set ($N=1,235$):
     - **DOWN Recall**: **51.44%**
     - **STABLE Recall**: **61.03%**
     - **UP Recall**: **3.59%** (only 9 of 251 captured; 96.4% missed!)
   - In preliminary forward paper trading ($N=22$), UP recall was **0.00%**.
   - **Warning**: The model has near-zero sensitivity to upward movements and **must not be used as an unhedged long directional trading signal**.

2. **Multi-Horizon Out-of-Sample Forward Validation (H2–H7)**:
   - Forward paper trading has resolved 22 H1 predictions.
   - Horizons H2 through H7 have **zero resolved forward observations** and are marked `NOT ENOUGH DATA`. They require live market session feeds to accumulate validation samples.

3. **Paper Trading Scope**:
   - The environment operates in simulated paper evaluation mode. Bid-ask spread, order book depth, slippage, and exchange transaction fees are not simulated.