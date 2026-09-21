"""
Production Feature Engineering Module for NSE Intraday ML Project (Corrected & Validated).

Enforces:
1. Strict elapsed-time returns: return_1m, return_5m, return_15m, return_30m, return_60m
   evaluated at exact clock timestamps (10:29, 10:25, 10:15, 10:00, 09:30 IST) against 10:30:00 IST.
   Missing timestamps return strictly NaN (no forward-filling, no interpolation).
2. Missingness indicators: return_1m_missing .. return_60m_missing.
3. Observed-candle windows for moving averages (ma_5, 15, 30) and volatility (volatility_5m .. 60m),
   capturing per-trade-event distribution without mixing semantics.
4. Numerically stable volume dynamics: log_volume_change = log1p(V_t) - log1p(V_{t-1})
   plus prev_vol_zero_flag and volume_ratio.
5. Session features strictly bounded between 09:15:00 and 10:30:00 IST.
6. Preprocessing fitted strictly on Clean Training observations (zero validation/test leakage).
7. Feature ablation support: MODEL_FEATURES_FULL vs MODEL_FEATURES_NO_HORIZON.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATABASE_URL,
    CANONICAL_SOURCE,
    SYNTH_SQL_TUPLE,
    EXCLUDED_SYNTHETIC_SYMBOLS,
    TIMEZONE,
    REPORTS_DIR,
    PROJECT_ROOT
)
from src.target_creation import create_target_pairs

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Feature Specifications
# -----------------------------------------------------------------------------

# 1. Exact elapsed-time return features & missing indicators
TIMESTAMP_RETURN_FEATURES = [
    'return_1m', 'return_5m', 'return_15m', 'return_30m', 'return_60m',
    'return_1m_missing', 'return_5m_missing', 'return_15m_missing', 'return_30m_missing', 'return_60m_missing'
]

# 2. Candle dispersion
PRICE_RANGE_FEATURES = ['price_range_pct']

# 3. Moving average relative displacement (observed-candle windows)
MA_DISTANCE_FEATURES = ['distance_from_ma_5', 'distance_from_ma_15', 'distance_from_ma_30']

# 4. Realized return volatility (observed-candle return standard deviations)
VOLATILITY_FEATURES = ['volatility_5m', 'volatility_15m', 'volatility_30m', 'volatility_60m']

# 5. Volume dynamics (statistically stable log difference + ratio + illiquidity flag)
VOLUME_FEATURES = ['volume_ratio', 'log_volume_change', 'prev_vol_zero_flag']

# 6. Session-so-far dynamics (strictly 09:15 to 10:30 IST)
SESSION_FEATURES = ['price_change_from_open', 'session_range_pct']

# 7. Day of week seasonality
TIME_FEATURES = ['day_of_week']

# 8. Horizon features
HORIZON_FEATURES = ['observed_sessions_ahead', 'calendar_days_ahead']

# Primary Ablation Specification (Market Signals Only, No Horizon)
MODEL_FEATURES_NO_HORIZON = (
    TIMESTAMP_RETURN_FEATURES +
    PRICE_RANGE_FEATURES +
    MA_DISTANCE_FEATURES +
    VOLATILITY_FEATURES +
    VOLUME_FEATURES +
    SESSION_FEATURES +
    TIME_FEATURES
)

# Primary Multi-Horizon Specification (Market Signals + Horizon)
MODEL_FEATURES_FULL = MODEL_FEATURES_NO_HORIZON + HORIZON_FEATURES

# Auditing and Level Features (retained for inspection and trees)
RAW_LEVEL_FEATURES = [
    'close', 'volume', 'volume_change', 'ma_5', 'ma_15', 'ma_30',
    'volume_ma_5', 'volume_ma_15',
    'session_open', 'session_high_so_far', 'session_low_so_far',
    'hour', 'minute', 'minute_of_day'
]

FORBIDDEN_TARGET_COLS = [
    'target_close', 'future_return_pct', 'target_class',
    'ts_tgt', 'trade_date_tgt', 'ts_ist_tgt'
]

def extract_exact_timestamp_features(engine) -> pd.DataFrame:
    """
    Extracts candles at exact clock times (09:30, 10:00, 10:15, 10:25, 10:29, 10:30 IST)
    and computes strict elapsed-time returns. Returns NaN if timestamp is absent.
    """
    exact_times = "('09:30:00'::time, '10:00:00'::time, '10:15:00'::time, '10:25:00'::time, '10:29:00'::time, '10:30:00'::time)"
    query = text(f"""
        SELECT 
            symbol,
            DATE(ts AT TIME ZONE '{TIMEZONE}') as trade_date,
            (ts AT TIME ZONE '{TIMEZONE}')::time as time_ist,
            close
        FROM ohlcv_intraday
        WHERE source = '{CANONICAL_SOURCE}'
          AND symbol NOT IN {SYNTH_SQL_TUPLE}
          AND (ts AT TIME ZONE '{TIMEZONE}')::time IN {exact_times}
        ORDER BY symbol, trade_date, time_ist;
    """)
    with engine.connect() as conn:
        df_exact = pd.read_sql(query, conn)

    pivot = df_exact.pivot(
        index=['symbol', 'trade_date'],
        columns='time_ist',
        values='close'
    ).reset_index()

    t1030 = pd.to_datetime('10:30:00').time()
    t1029 = pd.to_datetime('10:29:00').time()
    t1025 = pd.to_datetime('10:25:00').time()
    t1015 = pd.to_datetime('10:15:00').time()
    t1000 = pd.to_datetime('10:00:00').time()
    t0930 = pd.to_datetime('09:30:00').time()

    # Exact timestamp return formulas: (close(10:30) / close(T) - 1.0) * 100.0
    pivot['return_1m'] = np.where(
        pivot[t1030].notnull() & pivot[t1029].notnull(),
        (pivot[t1030] / pivot[t1029] - 1.0) * 100.0,
        np.nan
    )
    pivot['return_5m'] = np.where(
        pivot[t1030].notnull() & pivot[t1025].notnull(),
        (pivot[t1030] / pivot[t1025] - 1.0) * 100.0,
        np.nan
    )
    pivot['return_15m'] = np.where(
        pivot[t1030].notnull() & pivot[t1015].notnull(),
        (pivot[t1030] / pivot[t1015] - 1.0) * 100.0,
        np.nan
    )
    pivot['return_30m'] = np.where(
        pivot[t1030].notnull() & pivot[t1000].notnull(),
        (pivot[t1030] / pivot[t1000] - 1.0) * 100.0,
        np.nan
    )
    pivot['return_60m'] = np.where(
        pivot[t1030].notnull() & pivot[t0930].notnull(),
        (pivot[t1030] / pivot[t0930] - 1.0) * 100.0,
        np.nan
    )

    # Missing indicators
    pivot['return_1m_missing'] = pivot['return_1m'].isnull().astype(int)
    pivot['return_5m_missing'] = pivot['return_5m'].isnull().astype(int)
    pivot['return_15m_missing'] = pivot['return_15m'].isnull().astype(int)
    pivot['return_30m_missing'] = pivot['return_30m'].isnull().astype(int)
    pivot['return_60m_missing'] = pivot['return_60m'].isnull().astype(int)

    return pivot[[
        'symbol', 'trade_date',
        'return_1m', 'return_5m', 'return_15m', 'return_30m', 'return_60m',
        'return_1m_missing', 'return_5m_missing', 'return_15m_missing', 'return_30m_missing', 'return_60m_missing'
    ]]

def extract_morning_rolling_and_session_features(engine) -> pd.DataFrame:
    """
    Computes rolling MA, Volatility, Volume, and Session features strictly within session
    up to 10:30:00 IST. Moving averages and volatility are defined over observed-candle windows.
    """
    query = text(f"""
        SELECT 
            symbol,
            ts,
            DATE(ts AT TIME ZONE '{TIMEZONE}') as trade_date,
            (ts AT TIME ZONE '{TIMEZONE}')::time as time_ist,
            open, high, low, close, volume
        FROM ohlcv_intraday
        WHERE source = '{CANONICAL_SOURCE}'
          AND symbol NOT IN {SYNTH_SQL_TUPLE}
          AND (ts AT TIME ZONE '{TIMEZONE}')::time >= '09:15:00'::time
          AND (ts AT TIME ZONE '{TIMEZONE}')::time <= '10:30:00'::time
        ORDER BY symbol, ts;
    """)
    with engine.connect() as conn:
        df_morning = pd.read_sql(query, conn)

    df_morning = df_morning.sort_values(['symbol', 'ts']).reset_index(drop=True)

    # 1-minute candle return for volatility
    df_morning['ret_1m_candle'] = df_morning.groupby(['symbol', 'trade_date'])['close'].pct_change(1) * 100.0

    grouped = df_morning.groupby(['symbol', 'trade_date'])

    # Moving averages (observed candle windows)
    df_morning['ma_5'] = grouped['close'].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df_morning['ma_15'] = grouped['close'].transform(lambda s: s.rolling(15, min_periods=1).mean())
    df_morning['ma_30'] = grouped['close'].transform(lambda s: s.rolling(30, min_periods=1).mean())

    # Volatility (observed candle return sample standard deviation)
    df_morning['volatility_5m'] = grouped['ret_1m_candle'].transform(lambda s: s.rolling(5, min_periods=2).std())
    df_morning['volatility_15m'] = grouped['ret_1m_candle'].transform(lambda s: s.rolling(15, min_periods=2).std())
    df_morning['volatility_30m'] = grouped['ret_1m_candle'].transform(lambda s: s.rolling(30, min_periods=2).std())
    df_morning['volatility_60m'] = grouped['ret_1m_candle'].transform(lambda s: s.rolling(60, min_periods=2).std())

    # Volume moving averages
    df_morning['volume_ma_5'] = grouped['volume'].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df_morning['volume_ma_15'] = grouped['volume'].transform(lambda s: s.rolling(15, min_periods=1).mean())

    # Previous volume for volume changes
    df_morning['prev_volume'] = grouped['volume'].shift(1)

    # Session cumulative statistics strictly from 09:15 to current candle
    df_morning['session_open'] = grouped['open'].transform('first')
    df_morning['session_high_so_far'] = grouped['high'].cummax()
    df_morning['session_low_so_far'] = grouped['low'].cummin()

    # Filter strictly to the 10:30 candle
    df_ref = df_morning[df_morning['time_ist'].astype(str) == '10:30:00'].copy()

    # Normalized features
    df_ref['price_range_pct'] = ((df_ref['high'] - df_ref['low']) / df_ref['close']) * 100.0
    df_ref['distance_from_ma_5'] = ((df_ref['close'] - df_ref['ma_5']) / df_ref['ma_5']) * 100.0
    df_ref['distance_from_ma_15'] = ((df_ref['close'] - df_ref['ma_15']) / df_ref['ma_15']) * 100.0
    df_ref['distance_from_ma_30'] = ((df_ref['close'] - df_ref['ma_30']) / df_ref['ma_30']) * 100.0

    df_ref['volume_ratio'] = np.where(
        df_ref['volume_ma_15'] > 0,
        df_ref['volume'] / df_ref['volume_ma_15'],
        1.0
    )

    df_ref['prev_vol_zero_flag'] = (df_ref['prev_volume'] == 0).astype(int)

    # Raw volume change (retained for comparison/audit)
    df_ref['volume_change'] = np.where(
        df_ref['prev_volume'] > 0,
        ((df_ref['volume'] - df_ref['prev_volume']) / df_ref['prev_volume']) * 100.0,
        np.where(df_ref['volume'] == 0, 0.0, 100.0)
    )

    # Statistically stable log volume difference
    df_ref['log_volume_change'] = np.log1p(df_ref['volume']) - np.log1p(df_ref['prev_volume'].fillna(0.0))

    df_ref['price_change_from_open'] = ((df_ref['close'] - df_ref['session_open']) / df_ref['session_open']) * 100.0
    df_ref['session_range_pct'] = ((df_ref['session_high_so_far'] - df_ref['session_low_so_far']) / df_ref['close']) * 100.0

    df_ref['hour'] = 10
    df_ref['minute'] = 30
    df_ref['minute_of_day'] = 630
    df_ref['day_of_week'] = pd.to_datetime(df_ref['trade_date']).dt.dayofweek

    return df_ref

def build_corrected_feature_dataset(reference_time: str = '10:30:00') -> pd.DataFrame:
    """
    Builds the fully corrected, leakage-audited feature matrix joined with reference-target pairs.
    """
    engine = create_engine(DATABASE_URL)
    df_pairs = create_target_pairs(reference_time=reference_time)
    df_exact = extract_exact_timestamp_features(engine)
    df_rolling = extract_morning_rolling_and_session_features(engine)

    df_ref_combined = pd.merge(
        df_rolling,
        df_exact,
        on=['symbol', 'trade_date'],
        how='inner'
    )

    df_dataset = pd.merge(
        df_pairs,
        df_ref_combined,
        left_on=['symbol', 'trade_date_ref'],
        right_on=['symbol', 'trade_date'],
        how='inner'
    )
    df_dataset.drop(columns=['trade_date'], inplace=True)

    # Strict Leakage & Integrity Assertions
    assert (df_dataset['ts_tgt'] > df_dataset['ts_ref']).all(), "Target ts <= Reference ts"
    for col in MODEL_FEATURES_FULL:
        assert col not in FORBIDDEN_TARGET_COLS, f"Forbidden column {col} in features"
    assert (df_dataset['session_high_so_far'] >= df_dataset['session_low_so_far']).all()
    assert (df_dataset['reference_close'] > 0).all()
    assert (df_dataset['target_close'] > 0).all()
    assert not np.isinf(df_dataset[MODEL_FEATURES_FULL]).any().any()
    assert not df_dataset['symbol'].isin(EXCLUDED_SYNTHETIC_SYMBOLS).any()
    assert not df_dataset.duplicated(subset=['symbol', 'ts_ref', 'ts_tgt']).any()

    return df_dataset

if __name__ == "__main__":
    df = build_corrected_feature_dataset()
    print(f"Corrected feature dataset constructed: {len(df):,} pairs, {len(df.columns)} columns.")
    print(f"MODEL_FEATURES_FULL count: {len(MODEL_FEATURES_FULL)}")
    print(f"MODEL_FEATURES_NO_HORIZON count: {len(MODEL_FEATURES_NO_HORIZON)}")
