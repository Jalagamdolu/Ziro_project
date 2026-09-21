"""
Production Prediction Interface for NSE Intraday Stock Price Movement.

Accepts:
- symbol (e.g. '360ONE', 'RELIANCE')
- reference_timestamp (e.g. '2026-08-17 10:30 IST')
- target_timestamp (e.g. '2026-08-19 10:30 IST')

Outputs:
- predicted_class (UP, DOWN, STABLE)
- probability_down, probability_stable, probability_up
- reference_price
- target_horizon_observed_sessions
- target_horizon_calendar_days

STRICT RULE:
Zero target/future data is queried or utilized during prediction.
All features are generated strictly up to the reference timestamp.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import __main__
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.config import CANONICAL_SOURCE, TIMEZONE, MODELS_DIR
from src.market_data import (
    get_morning_candles as get_market_morning_candles,
    get_historical_dates as get_market_historical_dates
)

INV_CLASS_MAP = {0: 'DOWN', 1: 'STABLE', 2: 'UP'}

class PreprocessingPipeline:
    """Leakage-safe Preprocessing Pipeline fitted strictly on Clean Train."""
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()
        self.clip_bounds = {}
        
    def fit(self, X_train: pd.DataFrame):
        X_mat = X_train[self.feature_cols].values
        self.imputer.fit(X_mat)
        X_imp = self.imputer.transform(X_mat)
        
        p1 = np.percentile(X_imp, 1.0, axis=0)
        p99 = np.percentile(X_imp, 99.0, axis=0)
        self.clip_bounds = {'p1': p1, 'p99': p99}
        
        X_clip = np.clip(X_imp, p1, p99)
        self.scaler.fit(X_clip)
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_cols].values
        X_imp = self.imputer.transform(X_mat)
        X_clip = np.clip(X_imp, self.clip_bounds['p1'], self.clip_bounds['p99'])
        X_scaled = self.scaler.transform(X_clip)
        return X_scaled

__all__ = [
    "predict_movement",
    "get_observed_sessions_between",
    "PreprocessingPipeline",
    "ensure_pipeline_registered",
    "INV_CLASS_MAP",
]

def ensure_pipeline_registered():
    """Ensures PreprocessingPipeline is discoverable by pickle in __main__, main, and builtins."""
    import types
    import builtins
    for mod_name in ('__main__', 'main', '__mp_main__'):
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], 'PreprocessingPipeline', PreprocessingPipeline)
        else:
            mod = types.ModuleType(mod_name)
            mod.PreprocessingPipeline = PreprocessingPipeline
            sys.modules[mod_name] = mod
    setattr(builtins, 'PreprocessingPipeline', PreprocessingPipeline)
    if 'src.predict' in sys.modules:
        setattr(sys.modules['src.predict'], 'PreprocessingPipeline', PreprocessingPipeline)
    if 'predict' in sys.modules:
        setattr(sys.modules['predict'], 'PreprocessingPipeline', PreprocessingPipeline)

# Ensure PreprocessingPipeline is registered across all execution environments
ensure_pipeline_registered()

_HISTORICAL_DATES_CACHE = None

def get_observed_sessions_between(ref_date: str, tgt_date: str) -> int:
    """Calculates observed active trading session count between reference and target date."""
    global _HISTORICAL_DATES_CACHE
    ref_dt = pd.to_datetime(ref_date)
    tgt_dt = pd.to_datetime(tgt_date)
    
    if ref_dt.weekday() >= 5:
        day_name = ref_dt.strftime('%A')
        raise ValueError(
            f"Reference date '{ref_date}' is a {day_name} (non-trading weekend). "
            f"Please specify an active NSE trading day (Monday through Friday)."
        )

    if tgt_dt.weekday() >= 5:
        day_name = tgt_dt.strftime('%A')
        raise ValueError(
            f"Target date '{tgt_date}' is a {day_name} (non-trading weekend). "
            f"Please specify an active NSE trading day (Monday through Friday)."
        )
        
    if _HISTORICAL_DATES_CACHE is None:
        _HISTORICAL_DATES_CACHE = get_market_historical_dates(source="parquet")
        
    dates = _HISTORICAL_DATES_CACHE
    
    if ref_date in dates and tgt_date in dates:
        return dates.index(tgt_date) - dates.index(ref_date)
    else:
        # If target date is a future date beyond the database, calculate business days
        b_days = len(pd.bdate_range(ref_dt, tgt_dt)) - 1
        return max(1, b_days)

def predict_movement(
    symbol: str,
    reference_timestamp: str,
    target_timestamp: str,
    model_type: str = 'pooled',
    market_data_source: str = 'parquet'
) -> dict:
    """
    Generates movement prediction strictly using market data up to reference_timestamp.
    Self-contained: reads strictly from frozen Parquet dataset.
    """
    # Clean symbol
    symbol = symbol.strip().upper()
    
    # Parse timestamps
    ref_ts = pd.to_datetime(reference_timestamp.replace('IST', '').strip())
    tgt_ts = pd.to_datetime(target_timestamp.replace('IST', '').strip())
    
    if ref_ts >= tgt_ts:
        raise ValueError(f"Target timestamp ({tgt_ts}) must be strictly after reference timestamp ({ref_ts}).")
        
    ref_date_str = ref_ts.strftime('%Y-%m-%d')
    tgt_date_str = tgt_ts.strftime('%Y-%m-%d')
    ref_time_str = ref_ts.strftime('%H:%M:%S')
    
    calendar_days_ahead = (tgt_ts.date() - ref_ts.date()).days
    observed_sessions_ahead = get_observed_sessions_between(ref_date_str, tgt_date_str)
    
    if observed_sessions_ahead >= 8:
        print(f"Warning: Target horizon ({observed_sessions_ahead} sessions) exceeds supported range (1-7).")
        
    # Query exact morning candles for the symbol up to reference timestamp from Parquet
    df_candles = get_market_morning_candles(
        symbol=symbol,
        trade_date=ref_date_str,
        ref_time=ref_time_str,
        source='parquet'
    )
        
    if df_candles.empty:
        raise ValueError(f"No morning candles found for symbol '{symbol}' on date '{ref_date_str}' up to {ref_time_str}.")
        
    # Exact 10:30 candle
    ref_candle = df_candles[df_candles['time_ist'].astype(str) == ref_time_str]
    if ref_candle.empty:
        # If exact 10:30 candle is missing, use the latest candle before 10:30
        ref_candle = df_candles.iloc[[-1]]
        ref_price = float(ref_candle['close'].iloc[0])
    else:
        ref_price = float(ref_candle['close'].iloc[0])
        
    # 1. Exact timestamp returns (09:30, 10:00, 10:15, 10:25, 10:29)
    candles_by_time = {str(r['time_ist']): float(r['close']) for _, r in df_candles.iterrows()}
    
    def get_exact_return(t_str):
        if t_str in candles_by_time:
            return (ref_price / candles_by_time[t_str] - 1.0) * 100.0
        return np.nan
        
    ret_1m = get_exact_return('10:29:00')
    ret_5m = get_exact_return('10:25:00')
    ret_15m = get_exact_return('10:15:00')
    ret_30m = get_exact_return('10:00:00')
    ret_60m = get_exact_return('09:30:00')
    
    ret_1m_missing = 1 if np.isnan(ret_1m) else 0
    ret_5m_missing = 1 if np.isnan(ret_5m) else 0
    ret_15m_missing = 1 if np.isnan(ret_15m) else 0
    ret_30m_missing = 1 if np.isnan(ret_30m) else 0
    ret_60m_missing = 1 if np.isnan(ret_60m) else 0
    
    # 2. Observed-candle rolling moving averages & volatility
    closes = df_candles['close'].values
    volumes = df_candles['volume'].values
    highs = df_candles['high'].values
    lows = df_candles['low'].values
    
    ma_5 = float(np.mean(closes[-5:]))
    ma_15 = float(np.mean(closes[-15:]))
    ma_30 = float(np.mean(closes[-30:]))
    
    dist_ma_5 = (ref_price - ma_5) / ma_5 * 100.0
    dist_ma_15 = (ref_price - ma_15) / ma_15 * 100.0
    dist_ma_30 = (ref_price - ma_30) / ma_30 * 100.0
    
    # 1-min returns on observed candles
    ret_1m_arr = np.diff(closes) / closes[:-1] * 100.0
    vol_5m = float(np.std(ret_1m_arr[-5:], ddof=1)) if len(ret_1m_arr) >= 2 else 0.0
    vol_15m = float(np.std(ret_1m_arr[-15:], ddof=1)) if len(ret_1m_arr) >= 2 else 0.0
    vol_30m = float(np.std(ret_1m_arr[-30:], ddof=1)) if len(ret_1m_arr) >= 2 else 0.0
    vol_60m = float(np.std(ret_1m_arr[-60:], ddof=1)) if len(ret_1m_arr) >= 2 else 0.0
    
    # 3. Volume dynamics
    vol_t = float(volumes[-1])
    vol_prev = float(volumes[-2]) if len(volumes) >= 2 else 0.0
    vol_ma_15 = float(np.mean(volumes[-15:]))
    volume_ratio = vol_t / max(vol_ma_15, 1.0)
    log_vol_change = float(np.log1p(vol_t) - np.log1p(vol_prev))
    prev_vol_zero_flag = 1 if vol_prev == 0 else 0
    
    # 4. Session dynamics
    session_open = float(df_candles['open'].iloc[0])
    session_high = float(np.max(highs))
    session_low = float(np.min(lows))
    price_change_open = (ref_price - session_open) / session_open * 100.0
    session_range_pct = (session_high - session_low) / ref_price * 100.0
    price_range_pct = (float(ref_candle['high'].iloc[0]) - float(ref_candle['low'].iloc[0])) / ref_price * 100.0
    
    day_of_week = ref_ts.weekday()
    
    # Assemble feature dictionary
    feat_dict = {
        'return_1m': ret_1m,
        'return_5m': ret_5m,
        'return_15m': ret_15m,
        'return_30m': ret_30m,
        'return_60m': ret_60m,
        'return_1m_missing': ret_1m_missing,
        'return_5m_missing': ret_5m_missing,
        'return_15m_missing': ret_15m_missing,
        'return_30m_missing': ret_30m_missing,
        'return_60m_missing': ret_60m_missing,
        'price_range_pct': price_range_pct,
        'distance_from_ma_5': dist_ma_5,
        'distance_from_ma_15': dist_ma_15,
        'distance_from_ma_30': dist_ma_30,
        'volatility_5m': vol_5m,
        'volatility_15m': vol_15m,
        'volatility_30m': vol_30m,
        'volatility_60m': vol_60m,
        'volume_ratio': volume_ratio,
        'log_volume_change': log_vol_change,
        'prev_vol_zero_flag': prev_vol_zero_flag,
        'price_change_from_open': price_change_open,
        'session_range_pct': session_range_pct,
        'day_of_week': day_of_week,
        'observed_sessions_ahead': observed_sessions_ahead,
        'calendar_days_ahead': calendar_days_ahead
    }
    
    df_feat = pd.DataFrame([feat_dict])
    
    # Ensure PreprocessingPipeline is discoverable in unpickler namespace
    ensure_pipeline_registered()

    # Load model and preprocessor
    if model_type == 'h1' and observed_sessions_ahead == 1:
        model = joblib.load(MODELS_DIR / "best_h1_model.joblib")
        preprocessor = joblib.load(MODELS_DIR / "h1_preprocessor.joblib")
    else:
        model = joblib.load(MODELS_DIR / "best_pooled_model.joblib")
        preprocessor = joblib.load(MODELS_DIR / "pooled_preprocessor.joblib")
        
    # Transform and predict
    X_scaled = preprocessor.transform(df_feat)
    pred_idx = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    
    pred_class = INV_CLASS_MAP[pred_idx]
    
    result = {
        'symbol': symbol,
        'reference_timestamp': reference_timestamp,
        'target_timestamp': target_timestamp,
        'reference_price': round(ref_price, 2),
        'target_horizon_observed_sessions': int(observed_sessions_ahead),
        'target_horizon_calendar_days': int(calendar_days_ahead),
        'predicted_class': pred_class,
        'probability_down': round(float(probs[0]), 4),
        'probability_stable': round(float(probs[1]), 4),
        'probability_up': round(float(probs[2]), 4)
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock movement class for user-selected timestamps.")
    parser.add_argument("--symbol", type=str, default="360ONE", help="Stock ticker symbol")
    parser.add_argument("--ref-time", type=str, default="2026-08-17 10:30", help="Reference timestamp (IST)")
    parser.add_argument("--target-time", type=str, default="2026-08-19 10:30", help="Target timestamp (IST)")
    parser.add_argument("--model", type=str, default="pooled", choices=['pooled', 'h1'], help="Model type")
    
    args = parser.parse_args()
    
    print("\nRunning Inference...")
    res = predict_movement(args.symbol, args.ref_time, args.target_time, model_type=args.model)
    print("\nPrediction Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
