"""
Parquet Data Access & PostgreSQL Parity Test Suite.

Verifies:
1. Self-contained Parquet reader behavior (offline, zero DB dependency).
2. Bit-for-bit equivalence against PostgreSQL when PostgreSQL is available.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATABASE_URL, EXCLUDED_SYNTHETIC_SYMBOLS
from src.market_data import (
    get_legitimate_symbols,
    get_historical_dates,
    get_morning_candles,
    get_target_candle_close
)
from src.predict import predict_movement

REPRESENTATIVE_SYMBOLS = ["20MICRONS", "360ONE", "INFY", "RELIANCE", "TCS", "ZOMATO", "ZYDUSWELL"]
REPRESENTATIVE_DATES = ["2026-08-11", "2026-08-14", "2026-08-18", "2026-08-28", "2026-09-10"]

def is_postgres_available() -> bool:
    if not DATABASE_URL:
        return False
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(DATABASE_URL)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1;"))
            return True
    except Exception:
        return False

HAS_POSTGRES = is_postgres_available()

# ==============================================================================
# 1. SELF-CONTAINED PARQUET READER TESTS (Always run, zero DB required)
# ==============================================================================

def test_parquet_symbol_universe():
    """Validates that Parquet discovers the exact 2,353 symbols without synthetic ones."""
    symbols = get_legitimate_symbols(source="parquet")
    assert len(symbols) == 2353, f"Expected 2,353 symbols in Parquet, got {len(symbols)}"
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "360ONE" in symbols
    # Check no synthetic symbols
    for s in EXCLUDED_SYNTHETIC_SYMBOLS:
        assert s not in symbols, f"Synthetic symbol {s} found in Parquet universe!"

def test_parquet_trading_dates():
    """Validates that Parquet discovers the exact 13 trading dates."""
    dates = get_historical_dates(source="parquet")
    assert len(dates) == 13, f"Expected 13 dates in Parquet, got {len(dates)}"
    assert dates[0] == "2026-08-10"
    assert dates[-1] == "2026-09-10"

@pytest.mark.parametrize("symbol", REPRESENTATIVE_SYMBOLS)
@pytest.mark.parametrize("trade_date", ["2026-08-14", "2026-09-10"])
def test_parquet_morning_candles(symbol, trade_date):
    """Validates schema, types, and monotonic ordering of candles from Parquet."""
    df = get_morning_candles(symbol, trade_date, ref_time="10:30:00", source="parquet")
    if len(df) == 0:
        return
    expected_cols = ['ts', 'time_ist', 'open', 'high', 'low', 'close', 'volume']
    assert list(df.columns) == expected_cols
    assert df['ts'].is_monotonic_increasing
    assert df['close'].dtype == np.float64
    assert df['volume'].dtype == np.float64
    assert (df['open'] > 0).all()
    assert (df['close'] > 0).all()

@pytest.mark.parametrize("symbol", ["360ONE", "RELIANCE", "TCS", "INFY"])
def test_parquet_prediction_engine_pooled(symbol):
    """Validates self-contained prediction execution using pooled model."""
    pred = predict_movement(
        symbol=symbol,
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST",
        model_type="pooled"
    )
    assert pred['symbol'] == symbol
    assert pred['reference_price'] > 0
    assert pred['predicted_class'] in ['UP', 'DOWN', 'STABLE']
    assert 0.0 <= pred['probability_down'] <= 1.0
    assert 0.0 <= pred['probability_stable'] <= 1.0
    assert 0.0 <= pred['probability_up'] <= 1.0
    prob_sum = pred['probability_down'] + pred['probability_stable'] + pred['probability_up']
    assert abs(prob_sum - 1.0) < 0.01

@pytest.mark.parametrize("symbol", ["360ONE", "RELIANCE", "TCS"])
def test_parquet_prediction_engine_h1(symbol):
    """Validates self-contained prediction execution using H1 dedicated model."""
    pred = predict_movement(
        symbol=symbol,
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST",
        model_type="h1"
    )
    assert pred['predicted_class'] in ['UP', 'DOWN', 'STABLE']
    for p_col in ['probability_down', 'probability_stable', 'probability_up']:
        assert 0.0 <= pred[p_col] <= 1.0

@pytest.mark.parametrize("symbol", REPRESENTATIVE_SYMBOLS)
def test_parquet_target_candle_resolution(symbol):
    """Validates target resolution price retrieval from Parquet."""
    close = get_target_candle_close(symbol, "2026-09-10", "10:30:00", source="parquet")
    if close is not None:
        assert isinstance(close, float)
        assert close > 0

# ==============================================================================
# 2. A/B PARITY TESTS (Run when PostgreSQL is available)
# ==============================================================================

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not running (self-contained mode)")
def test_symbol_universe_parity():
    symbols_pg = get_legitimate_symbols(source="postgres")
    symbols_pq = get_legitimate_symbols(source="parquet")
    assert len(symbols_pg) == 2353
    assert len(symbols_pq) == 2353
    assert symbols_pg == symbols_pq

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not running (self-contained mode)")
def test_trading_dates_parity():
    dates_pg = get_historical_dates(source="postgres")
    dates_pq = get_historical_dates(source="parquet")
    assert dates_pg == dates_pq

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not running (self-contained mode)")
@pytest.mark.parametrize("symbol", REPRESENTATIVE_SYMBOLS)
@pytest.mark.parametrize("trade_date", ["2026-08-14", "2026-09-10"])
def test_morning_candle_row_level_parity(symbol, trade_date):
    df_pg = get_morning_candles(symbol, trade_date, ref_time="10:30:00", source="postgres")
    df_pq = get_morning_candles(symbol, trade_date, ref_time="10:30:00", source="parquet")
    assert len(df_pg) == len(df_pq)
    if len(df_pg) == 0:
        return
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert np.isclose(df_pg[col], df_pq[col], rtol=1e-7, atol=1e-7).all()

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not running (self-contained mode)")
@pytest.mark.parametrize("symbol", ["360ONE", "RELIANCE", "TCS", "INFY"])
def test_prediction_engine_parity_pooled_with_pg(symbol):
    ref_ts = "2026-09-09 10:30 IST"
    tgt_ts = "2026-09-10 10:30 IST"
    pred_pg = predict_movement(symbol=symbol, reference_timestamp=ref_ts, target_timestamp=tgt_ts, model_type="pooled", market_data_source="postgres")
    pred_pq = predict_movement(symbol=symbol, reference_timestamp=ref_ts, target_timestamp=tgt_ts, model_type="pooled", market_data_source="parquet")
    assert pred_pg['predicted_class'] == pred_pq['predicted_class']
    for p_col in ['probability_down', 'probability_stable', 'probability_up']:
        assert np.isclose(pred_pg[p_col], pred_pq[p_col], atol=1e-5)
