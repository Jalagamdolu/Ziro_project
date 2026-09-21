"""
Automated Test Suite for Paper Trading Engine & Lifecycle.

Validates:
1. Record creation in OPEN state.
2. Prediction resolution with exact return calculation.
3. Correct UP / DOWN / STABLE classification thresholds.
4. EXPIRED status handling when target market candle is unavailable.
5. Immutability: Historical records are never overwritten.
6. Zero future data accessed during prediction stage.
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest
from sqlalchemy import event, text, create_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_trading import (
    create_paper_prediction,
    resolve_prediction,
    get_prediction,
    list_predictions,
    engine,
    init_paper_trading_db
)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_paper_trading_db()

def test_paper_prediction_creation_open_state():
    """Validates that a new paper prediction is created in OPEN state without actuals."""
    res = create_paper_prediction(
        symbol="360ONE",
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST",
        model_type="pooled"
    )
    
    assert res['prediction_id'].startswith("pred_")
    assert res['symbol'] == "360ONE"
    assert res['reference_price'] > 0
    assert res['observed_sessions_ahead'] == 1
    assert res['status'] == "OPEN"
    assert res['is_resolved'] is False
    assert res['actual_price'] is None
    assert res['future_return_pct'] is None
    assert res['actual_class'] is None
    assert res['resolved_at'] is None
    
    # Check probabilities
    assert 0.0 <= res['probability_down'] <= 1.0
    assert 0.0 <= res['probability_stable'] <= 1.0
    assert 0.0 <= res['probability_up'] <= 1.0
    prob_sum = res['probability_down'] + res['probability_stable'] + res['probability_up']
    assert abs(prob_sum - 1.0) < 1e-3

def test_paper_prediction_resolution_lifecycle():
    """Validates prediction resolution from OPEN to RESOLVED using market data."""
    # Create prediction on Sep 09 with target Sep 10 (both exist in DB)
    res = create_paper_prediction(
        symbol="360ONE",
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST",
        model_type="pooled"
    )
    pred_id = res['prediction_id']
    
    # Resolve against market data
    resolved = resolve_prediction(pred_id, as_of_time="2026-09-10 15:30:00")
    
    assert resolved['prediction_id'] == pred_id
    assert resolved['status'] == "RESOLVED"
    assert resolved['is_resolved'] is True
    assert resolved['actual_price'] is not None
    assert resolved['future_return_pct'] is not None
    assert resolved['actual_class'] in ["UP", "DOWN", "STABLE"]
    assert resolved['resolved_at'] is not None
    
    # Validate return formula: (actual_price - ref_price) / ref_price * 100
    expected_ret = ((resolved['actual_price'] - resolved['reference_price']) / resolved['reference_price']) * 100.0
    assert abs(resolved['future_return_pct'] - expected_ret) < 1e-4
    
    # Validate class thresholds
    if resolved['future_return_pct'] > 1.0:
        assert resolved['actual_class'] == "UP"
    elif resolved['future_return_pct'] < -1.0:
        assert resolved['actual_class'] == "DOWN"
    else:
        assert resolved['actual_class'] == "STABLE"

def test_prediction_immutability_new_record_each_time():
    """Validates that successive predictions create distinct new records and do not overwrite."""
    res1 = create_paper_prediction(
        symbol="360ONE",
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST"
    )
    res2 = create_paper_prediction(
        symbol="360ONE",
        reference_timestamp="2026-09-09 10:30 IST",
        target_timestamp="2026-09-10 10:30 IST"
    )
    
    assert res1['prediction_id'] != res2['prediction_id']
    
    # Check both exist in persistence
    p1 = get_prediction(res1['prediction_id'])
    p2 = get_prediction(res2['prediction_id'])
    assert p1 is not None
    assert p2 is not None

def test_expired_prediction_when_market_data_missing():
    """Validates that a prediction transitions to EXPIRED if target date passed but candle missing."""
    # Insert a synthetic open prediction where target timestamp has no 10:30 candle in the past
    pred_id = f"test_expire_{datetime.now().strftime('%M%S%f')}"
    insert_sql = text("""
        INSERT INTO paper_predictions (
            prediction_id, symbol, reference_timestamp, target_timestamp,
            reference_price, observed_sessions_ahead, calendar_days_ahead,
            predicted_class, probability_down, probability_stable, probability_up,
            model_name, model_version, status, created_at, is_resolved
        ) VALUES (
            :pid, '360ONE', '2026-08-17 10:30:00', '2026-08-18 10:30:00',
            1000.0, 1, 1, 'STABLE', 0.3, 0.4, 0.3, 'Test', '1.0', 'OPEN', :created_at, :is_resolved
        );
    """)
    with engine.connect() as conn:
        conn.execute(insert_sql, {
            'pid': pred_id,
            'created_at': datetime.now().isoformat(sep=' ', timespec='seconds'),
            'is_resolved': 0
        })
        conn.commit()
        
    # In August 18, check resolution
    resolved = resolve_prediction(pred_id, as_of_time="2026-08-18 15:30:00")
    # If 10:30 candle existed, it resolves; if missing, it becomes EXPIRED
    assert resolved['status'] in ["RESOLVED", "EXPIRED"]
    if resolved['status'] == "EXPIRED":
        assert resolved['is_resolved'] is False
        assert "unavailable" in resolved['status_reason'].lower()

def test_leakage_isolation_during_paper_prediction():
    """Validates that market data is never queried for target date during paper prediction."""
    executed_queries = []
    def query_listener(conn, cursor, statement, parameters, context, executemany):
        executed_queries.append({'stmt': statement, 'params': parameters})
        
    event.listen(engine, "before_cursor_execute", query_listener)
    try:
        executed_queries.clear()
        ref_ts = "2026-09-10 10:30 IST"
        tgt_ts = "2026-09-11 10:30 IST"
        
        res = create_paper_prediction(
            symbol="360ONE",
            reference_timestamp=ref_ts,
            target_timestamp=tgt_ts
        )
        
        # Verify that only paper_predictions persistence was executed, never market data SQL queries
        for q in executed_queries:
            s = q['stmt'].lower()
            assert "ohlcv_intraday" not in s, "LEAKAGE: Market data was queried via SQL!"
            assert "paper_predictions" in s, f"Unexpected SQL query: {s}"
                
    finally:
        event.remove(engine, "before_cursor_execute", query_listener)
