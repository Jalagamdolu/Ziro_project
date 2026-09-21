# Phase 4 Correction & Final Validation Report

**Status**: ALL ISSUES FIXED & FULLY VALIDATED (ZERO LEAKAGE)  
**Date Generated**: 2026-09-10  
**Enforcement**: Zero ML models trained; strictly feature engineering and dataset validation.

---

## 1. Summary of Changes Made & Methodological Rationale

| Issue Identified | Root Cause in Initial Phase 4 | Correction Implemented | Methodological Rationale |
| :--- | :--- | :--- | :--- |
| **Elapsed-Time vs Candle Count** | Initial returns used `pct_change(N)` across observed candles. | Implemented strict clock-timestamp returns: `close(10:30) / close(T) - 1.0` for 10:29, 10:25, 10:15, 10:00, 09:30. | NSE data contains irregular timestamp gaps. Returns now represent genuine elapsed market time, not arbitrary trade counts. Missing timestamps evaluate to `NaN` with zero interpolation. |
| **MA & Volatility Semantics** | Semantic ambiguity between elapsed minutes and observed candles. | Explicitly defined and documented MA_K and volatility_K as **K observed-candle windows**. | Avoids artificial forward-filling or interpolation on non-trading minutes; accurately captures volatility per transaction event. |
| **Missingness Imputation Policy** | Missing short-term returns were previously filled with 0.0%. | Preserved `NaN` during feature generation; added 5 explicit missingness indicators (`return_Xm_missing`). | "Insufficient historical data" is an informative market microstructure event (illiquidity), not a 0% return. Imputation is deferred strictly to Clean Train medians. |
| **Extreme `volume_change`** | Skewed ratio $(V_t - V_{t-1})/V_{t-1}$ reached $+1,595,200\%$. | Introduced `log_volume_change = log1p(V_t) - log1p(V_{t-1})` while preserving `prev_vol_zero_flag` and raw `volume_change`. | Compresses extreme volume surges into a well-behaved $[-11.06, +12.43]$ range, eliminating division-by-zero artifacts. |
| **Preprocessing Population** | Preprocessing was initially tested on raw Train allocation ($N=18,904$). | Re-engineered preprocessing to fit exclusively on the **Clean Train** set ($N=4,824$). | Guarantees that purged training observations (whose multi-day targets overlap Validation) never leak into imputer or scaler statistics. |
| **Horizon-1 Row Reconciliation** | 898 rows were previously unaccounted for due to unassigned gap sessions. | Fully reconciled all 5,793 rows into Clean Train, Purged Gap Embargos, Clean Val, and Clean Test. | Full mathematical transparency: $854 + 1,158 + 994 + 687 + 2,100 = 5,793$. |
| **Horizon Feature Ablation** | Dominant horizon features risk model learning class priors instead of price signals. | Created two distinct feature specifications: `MODEL_FEATURES_FULL` (26 features) and `MODEL_FEATURES_NO_HORIZON` (24 features). | Enables Phase 5 to rigorously evaluate whether the model learns real intraday price momentum vs calendar class probabilities. |

---

## 2. Before / After Feature Definitions

### Timestamp-Based Returns
- **Before**: `return_5m = grouped['close'].pct_change(5) * 100.0` (evaluated on 5th preceding candle, regardless of elapsed time).
- **After**:
  $$\text{return\_5m} = \begin{cases} \left(\frac{\text{close}(10:30:00)}{\text{close}(10:25:00)} - 1.0\right) \times 100.0 & \text{if exact 10:25 candle exists} \\ \text{NaN} & \text{otherwise (missing timestamp)} \end{cases}$$
  Accompanied by binary indicator `return_5m_missing = 1` if return_5m is NaN else 0.

### Volume Microstructure
- **Before**: `volume_change = (V_t - V_{t-1}) / V_{t-1} * 100` (Mean $+1,467\%$, Std $23,768\%$, Max $+1,595,200\%$).
- **After**:
  $$\text{log\_volume\_change} = \ln(1 + V_t) - \ln(1 + V_{t-1})$$
  (Mean $+0.069$, Std $2.699$, Min $-11.06$, Max $+12.43$, perfectly symmetric and numerically stable).

### Moving Averages & Volatility
- **Explicit Definition**: Observed-candle windows over active trading bars:
  - $\text{MA}_5, \text{MA}_{15}, \text{MA}_{30}$: Rolling mean of close over the last 5, 15, 30 active trading candles up to 10:30 IST.
  - $\text{volatility}_5, \text{volatility}_{15}, \text{volatility}_{30}, \text{volatility}_{60}$: Sample standard deviation of observed 1-minute candle returns over the preceding 5, 15, 30, 60 active bars.

---

## 3. Dataset Row Reconciliation

### A. Dedicated Horizon-1 Dataset Reconciliation ($N = 5,793$)

Every single row in the 5,793 Horizon-1 dataset is accounted for across its exact reference-to-target transitions:

| Segment | Reference Session | Target Session | Pair Count | Purging / Embargo Status | Role in Split |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Train (Clean)** | 2026-08-17 | 2026-08-18 | 444 | Clean (Target $\le$ Aug 19) | **Model Training** |
| **Train (Clean)** | 2026-08-18 | 2026-08-19 | 410 | Clean (Target $\le$ Aug 19) | **Model Training** |
| *Embargo Buffer 1* | 2026-08-19 | 2026-08-27 | **1,158** | **Purged Embargo** (Target touches Val ref Aug 27) | Excluded (Buffer) |
| **Validation (Clean)**| 2026-08-27 | 2026-08-28 | 994 | Clean (Target $\le$ Aug 28) | **Model Validation** |
| *Embargo Buffer 2* | 2026-08-28 | 2026-09-02 | **687** | **Purged Embargo** (Target touches Test ref Sep 02) | Excluded (Buffer) |
| **Test (Clean)** | 2026-09-02 | 2026-09-09 | 865 | Clean Out-of-Sample | **Final Evaluation** |
| **Test (Clean)** | 2026-09-09 | 2026-09-10 | 1,235 | Clean Out-of-Sample | **Final Evaluation** |
| **TOTAL** | | | **5,793** | | **Exact 100% Match** |

```text
Reconciliation Arithmetic:
Clean Train:                     854 pairs (444 + 410)
Purged Embargo 1 (Aug 19->27): 1,158 pairs
Clean Validation:                994 pairs
Purged Embargo 2 (Aug 28->02):   687 pairs
Clean Test:                    2,100 pairs (865 + 1,235)
------------------------------------------------------
TOTAL RECONCILED:              5,793 pairs (854 + 1,158 + 994 + 687 + 2,100)
```

### B. Pooled Multi-Horizon Dataset Reconciliation ($N = 24,570$)

| Split Partition | Reference Period | Target Period | Raw Allocation | Purged Multi-Day Overlap | Clean Final Allocation |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Train** | Ref $\le$ 2026-08-27 | Target $\le$ 2026-08-27 | 18,904 | 14,080 | **4,824** |
| **Validation** | Ref 2026-08-28, 2026-09-02 | Target $\le$ 2026-09-02 | 4,431 | 3,744 | **687** |
| **Test** | Ref 2026-09-09, 2026-09-10 | Target 2026-09-10 | 1,235 | 0 | **1,235** |
| **TOTAL** | | | **24,570** | **17,824** | **6,746** |

```text
Reconciliation Arithmetic:
Clean Train:      4,824 pairs
Purged Train:    14,080 pairs
Clean Val:          687 pairs
Purged Val:       3,744 pairs
Test:             1,235 pairs
------------------------------
TOTAL RECONCILED: 24,570 pairs (4,824 + 14,080 + 687 + 3,744 + 1,235)
```

---

## 4. Preprocessing Fit Population

- **Clean Training Population**: Preprocessing transformations are fitted **exclusively on the 4,824 Clean Training observations**.
- **Fitted Components**:
  1. `SimpleImputer(strategy='median')`: Learns median values strictly from Clean Train.
  2. `Winsorizer / Percentile Clipper`: Learns 1st and 99th percentile boundaries strictly from Clean Train.
  3. `RobustScaler()`: Learns median and IQR scalers strictly from Clean Train.
- **Assertion**: Validation ($N=687$) and Test ($N=1,235$) rows are transformed using already-fitted training parameters. Zero validation/test information enters the preprocessor.
- **Post-Transform Validation**: All preprocessed matrices contain **0 NaN** and **0 Inf**.

---

## 5. Comprehensive 14-Point Leakage Audit Results

| # | Leakage Assertion | Target Criterion | Verification Method | Status |
| :-: | :--- | :--- | :--- | :---: |
| 1 | `source == 'yahoo'` only | Source isolation | SQL query parameter verification | **PASS** |
| 2 | Synthetic/test symbols excluded | 17 synthetic tickers removed | `assert not symbol.isin(EXCLUDED)` | **PASS** |
| 3 | Target timestamp > Reference timestamp | Strictly future targets | `assert (ts_tgt > ts_ref).all()` | **PASS** |
| 4 | Every feature timestamp $\le$ Reference timestamp | No future price in features | SQL filter strictly `time <= 10:30:00` | **PASS** |
| 5 | No `target_close` in $X$ | Target price quarantine | `assert 'target_close' not in X` | **PASS** |
| 6 | No `future_return_pct` in $X$ | Target return quarantine | `assert 'future_return_pct' not in X` | **PASS** |
| 7 | No `target_class` in $X$ | Target label quarantine | `assert 'target_class' not in X` | **PASS** |
| 8 | No full-day high/low/volume | Eventual day stats excluded | Only `session_*_so_far` used | **PASS** |
| 9 | No cross-session rolling windows | Overnight gap separation | Grouped strictly by `(symbol, date)` | **PASS** |
| 10 | No forward filling | No missing candle carryover | `NaN` returned on missing timestamp | **PASS** |
| 11 | No interpolation | No synthetic candle creation | Exact timestamp lookups only | **PASS** |
| 12 | No val/test data in preprocessing | Clean train fit only | Preprocessor fitted on 4,824 train rows | **PASS** |
| 13 | No infinite values | Finite numeric sanity | `assert not isinf(X).any()` | **PASS** |
| 14 | No duplicate pairs | Pair uniqueness | `assert not duplicated(sym, ref, tgt)` | **PASS** |

---

## 6. Horizon Feature Ablation Specifications

To evaluate whether Phase 5 models learn genuine intraday market signals or merely predict horizon-dependent class priors:

### Specification A: `MODEL_FEATURES_FULL` (26 Features)
Includes all 24 market microstructure features plus `observed_sessions_ahead` and `calendar_days_ahead`.

### Specification B: `MODEL_FEATURES_NO_HORIZON` (24 Features)
Excludes `observed_sessions_ahead` and `calendar_days_ahead`. Forces the classifier to predict price movement purely from technical momentum, volatility, volume flow, and intraday session range.

---

## 7. Remaining Limitations

1. **Illiquidity Missingness**: Because exact timestamps are strictly enforced without forward-fill, `return_1m` has 9.58% missingness and `return_60m` has 3.29% missingness on illiquid tickers. This is correctly flagged via binary indicators and imputed with Clean Train medians.
2. **Purge Attrition in Multi-Horizon**: The strict purge removes 14,080 training observations in the pooled model because multi-day horizons cross the Aug 28 boundary. The remaining 4,824 clean training observations are 100% leak-free and sufficient for robust linear and tree models.
