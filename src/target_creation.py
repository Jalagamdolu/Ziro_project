"""
Strict Target Creation Module for NSE Intraday ML Project.

Enforces:
1. Exact timestamp matching (symbol, 10:30 reference, 10:30 target).
2. Purely future targets (target_ts > reference_ts).
3. Explicit classification threshold (±1.0% future return).
4. Dual horizon calculation (calendar days ahead and observed sessions ahead).
5. Comprehensive data validation assertions.
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
    TIMEZONE,
    UP_THRESHOLD_PCT,
    DOWN_THRESHOLD_PCT,
    REPORTS_DIR
)

engine = create_engine(DATABASE_URL)

def get_session_classification(trade_date_str: str) -> str:
    """Classify date based on verified operational session status."""
    if trade_date_str in ('2026-08-10', '2026-08-11', '2026-08-12', '2026-08-13'):
        return 'pilot/canary'
    elif trade_date_str == '2026-08-14':
        return 'partial'
    elif trade_date_str in ('2026-08-27', '2026-09-02'):
        return 'truncated'
    elif trade_date_str == '2026-09-10':
        return 'active/incomplete'
    elif trade_date_str in ('2026-08-17', '2026-08-19', '2026-08-28', '2026-09-09'):
        return 'full'
    elif trade_date_str == '2026-08-18':
        return 'partial'
    else:
        return 'full'

def create_target_pairs(reference_time: str = '10:30:00') -> pd.DataFrame:
    """
    Constructs exact reference-target pairs for the specified reference clock time.
    Strictly joins on (symbol, target_ts > reference_ts) where exact candles exist.
    """
    print(f"Querying exact candles for reference/target time: {reference_time} IST...")
    with engine.connect() as conn:
        df_candles = pd.read_sql(text(f"""
            SELECT 
                symbol,
                ts,
                DATE(ts AT TIME ZONE '{TIMEZONE}') as trade_date,
                ts AT TIME ZONE '{TIMEZONE}' as ts_ist,
                close
            FROM ohlcv_intraday
            WHERE source = '{CANONICAL_SOURCE}'
              AND symbol NOT IN {SYNTH_SQL_TUPLE}
              AND (ts AT TIME ZONE '{TIMEZONE}')::time = '{reference_time}'::time
            ORDER BY symbol, ts;
        """), conn)

    print(f"Total {reference_time} candles retrieved: {len(df_candles):,}")
    print(f"Distinct symbols: {df_candles['symbol'].nunique():,}")
    print(f"Distinct dates: {df_candles['trade_date'].nunique()}")

    # Build observed session indexing across unique dates present
    obs_dates = sorted(df_candles['trade_date'].unique())
    obs_date_to_idx = {d: i for i, d in enumerate(obs_dates)}

    # Exact Self-Join by Symbol
    pairs = pd.merge(
        df_candles,
        df_candles,
        on='symbol',
        suffixes=('_ref', '_tgt')
    )

    # Filter strictly future targets
    pairs = pairs[pairs['ts_tgt'] > pairs['ts_ref']].copy()

    # Calculate returns and horizons
    pairs['reference_close'] = pairs['close_ref'].astype(float)
    pairs['target_close'] = pairs['close_tgt'].astype(float)
    pairs['future_return_pct'] = ((pairs['target_close'] - pairs['reference_close']) / pairs['reference_close']) * 100.0

    # 3-Class Target Classification
    def assign_target_class(ret: float) -> str:
        if ret > UP_THRESHOLD_PCT:
            return 'UP'
        elif ret < DOWN_THRESHOLD_PCT:
            return 'DOWN'
        else:
            return 'STABLE'

    pairs['target_class'] = pairs['future_return_pct'].apply(assign_target_class)

    # Calculate calendar days ahead and observed sessions ahead
    pairs['ref_date'] = pd.to_datetime(pairs['trade_date_ref'])
    pairs['tgt_date'] = pd.to_datetime(pairs['trade_date_tgt'])
    pairs['calendar_days_ahead'] = (pairs['tgt_date'] - pairs['ref_date']).dt.days
    pairs['observed_sessions_ahead'] = pairs.apply(
        lambda r: obs_date_to_idx[r['trade_date_tgt']] - obs_date_to_idx[r['trade_date_ref']], axis=1
    )

    # Session Quality Classifications
    pairs['ref_session_type'] = pairs['trade_date_ref'].astype(str).apply(get_session_classification)
    pairs['tgt_session_type'] = pairs['trade_date_tgt'].astype(str).apply(get_session_classification)

    # Eligible for benchmark supervised learning:
    # Explicitly exclude early pilot/canary sessions (Aug 10-14: 3 to 206 symbols)
    # Aug 17 onward (776 to 2,292 symbols) are production multi-stock benchmark sessions
    PILOT_CANARY_DATES = {'2026-08-10', '2026-08-11', '2026-08-12', '2026-08-13', '2026-08-14'}
    pairs['is_benchmark_eligible'] = (
        ~pairs['trade_date_ref'].astype(str).isin(PILOT_CANARY_DATES) &
        ~pairs['trade_date_tgt'].astype(str).isin(PILOT_CANARY_DATES)
    )

    # Format cleanly
    result_df = pairs[[
        'symbol',
        'trade_date_ref',
        'ts_ref',
        'ts_ist_ref',
        'reference_close',
        'trade_date_tgt',
        'ts_tgt',
        'ts_ist_tgt',
        'target_close',
        'future_return_pct',
        'target_class',
        'calendar_days_ahead',
        'observed_sessions_ahead',
        'ref_session_type',
        'tgt_session_type',
        'is_benchmark_eligible'
    ]].copy()

    # -------------------------------------------------------------
    # RIGOROUS DATA VALIDATION ASSERTIONS
    # -------------------------------------------------------------
    print("\nRunning strict data quality & leakage assertions...")
    assert (result_df['ts_tgt'] > result_df['ts_ref']).all(), "Assertion Failure: target_ts must be strictly > reference_ts"
    assert (result_df['reference_close'] > 0).all(), "Assertion Failure: reference_close must be strictly positive"
    assert (result_df['target_close'] > 0).all(), "Assertion Failure: target_close must be strictly positive"
    assert np.isfinite(result_df['future_return_pct']).all(), "Assertion Failure: future_return_pct must be finite"
    assert set(result_df['target_class'].unique()).issubset({'UP', 'DOWN', 'STABLE'}), "Assertion Failure: invalid target class"
    assert not result_df.duplicated(subset=['symbol', 'ts_ref', 'ts_tgt']).any(), "Assertion Failure: duplicate pair detected"
    assert (result_df['calendar_days_ahead'] > 0).all(), "Assertion Failure: calendar days ahead must be > 0"
    assert (result_df['observed_sessions_ahead'] > 0).all(), "Assertion Failure: observed sessions ahead must be > 0"
    print("ALL ASSERTIONS PASSED (100% compliant with strict target creation rules).")

    return result_df

if __name__ == "__main__":
    df_pairs = create_target_pairs()
    print(f"\nGenerated {len(df_pairs):,} exact reference-target pairs.")
    print("\nTarget Class Distribution:")
    print(df_pairs['target_class'].value_counts(normalize=True).mul(100).round(2).to_string())
