# Comprehensive Exploratory Data Analysis (EDA) Summary

## 1. Verified Canonical Dataset
- **Canonical Source**: `yahoo`
- **Excluded Synthetic Symbols**: 17 confirmed test symbols (391 records removed).
- **Legitimate Traded Universe**: **2,353 symbols**
- **Total Canonical Rows**: **2,639,421 rows**
- **Observed Trading Dates**: **13 dates** (`2026-08-10 15:11:00` to `2026-09-10 14:37:00`)

## 2. Price Distribution & Range
- **Close Price Summary**:
  - Minimum: ₹0.17
  - 25th Percentile: ₹136.72
  - Median: ₹395.65
  - Mean: ₹1203.22
  - 75th Percentile: ₹1035.60
  - 95th Percentile: ₹4160.00
  - Maximum: ₹133,895.00
- **1-Minute Candle Range %** (`(High - Low) / Close * 100`):
  - Median: 0.0602%
  - Mean: 0.1159%
  - 95th Percentile: 0.4110%

## 3. Volume Distribution
- **Zero-Volume Proportion**: **8.37%** (220,903 candles).
- **Traded Volume Percentiles**:
  - Median: 340.0 shares/candle
  - 75th Percentile: 1,972.0 shares/candle
  - 95th Percentile: 22,579.0 shares/candle
  - Maximum: 27,569,936 shares/candle

## 4. Return Distributions (Full Sessions)
- **1-Minute Return**: Mean = 0.00017%, Median = 0.0000%, Std = 0.2689%
- **5-Minute Return**: Mean = 0.00333%, Median = 0.0000%, Std = 0.3632%
- **15-Minute Return**: Mean = 0.01151%, Median = -0.0073%, Std = 0.5409%
- **Extreme Returns (>20%)**: 5 occurrences in full session sample.

## 5. Intraday Seasonality & Correlations
- **U-Shaped Volatility**:
  - 09:16 AM Opening Peak: Mean Abs Return = 0.4565%
  - 12:30 PM Midday Trough: Mean Abs Return = 0.0817%
  - 15:29 PM Closing Surge: Mean Abs Return = 0.2807%
- **Volume vs Abs Return Correlation**:
  - Pearson r: 0.06602
  - Spearman rho: 0.17646
- **Volume vs Price Range % Correlation**:
  - Pearson r: 0.12836
  - Spearman rho: 0.49689

## 6. Reference Time Coverage & Recommendation
- **Optimal Time**: **10:30:00 IST** covers 12 dates and averages 843.5 symbols/day while preserving morning truncated dates.

## 7. Horizon Feasibility (10:30 IST Exact Pairs)
- **Total Valid Pairs**: **25,418 pairs**
- **1 Observed Session Ahead**: 5,915 pairs
- **2 Observed Sessions Ahead**: 5,486 pairs
- **3 Observed Sessions Ahead**: 4,756 pairs
