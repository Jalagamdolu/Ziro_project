"""
Automated Test: Future-Target Inference & Zero-Target-Query Verification.

Validates:
1. Prediction for a future target timestamp not existing in database (e.g. 2026-09-11 10:30 IST).
2. Proves that NO OHLCV query touches the target date or any timestamp > reference timestamp.
3. Proves target data does not need to exist in the database.
4. Validates rejection of weekend target timestamps (e.g. Saturday 2026-09-12).
5. Validates multi-day business day calendar calculation for future dates.
"""

import sys
from pathlib import Path
import pandas as pd
from unittest.mock import patch
from src.predict import predict_movement, get_observed_sessions_between

def test_future_target_inference():
    print("\n========================================================")
    print("TEST 1: Reference Sep 10 -> Target Sep 11 (Non-existent in DB/Dataset)")
    print("========================================================")

    ref_ts = "2026-09-10 10:30 IST"
    tgt_ts = "2026-09-11 10:30 IST"
    symbol = "360ONE"

    # Track calls to market data candle retrieval to verify zero leakage
    import src.predict as predict_mod
    original_get_candles = predict_mod.get_market_morning_candles
    retrieval_calls = []

    def candle_tracker(*args, **kwargs):
        retrieval_calls.append({'args': args, 'kwargs': kwargs})
        return original_get_candles(*args, **kwargs)

    with patch.object(predict_mod, 'get_market_morning_candles', side_effect=candle_tracker):
        res = predict_movement(symbol=symbol, reference_timestamp=ref_ts, target_timestamp=tgt_ts, model_type='pooled')

    print("Prediction Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")

    # Assertions on output
    assert res['symbol'] == symbol
    assert res['reference_price'] > 0
    assert res['target_horizon_observed_sessions'] == 1
    assert res['target_horizon_calendar_days'] == 1
    assert res['predicted_class'] in ['UP', 'DOWN', 'STABLE']
    assert 0.0 <= res['probability_down'] <= 1.0
    assert 0.0 <= res['probability_stable'] <= 1.0
    assert 0.0 <= res['probability_up'] <= 1.0
    prob_sum = res['probability_down'] + res['probability_stable'] + res['probability_up']
    assert abs(prob_sum - 1.0) < 0.01

    # Verify no query requested target date or future prices
    assert len(retrieval_calls) == 1, f"Expected 1 candle retrieval call, got {len(retrieval_calls)}"
    call_kwargs = retrieval_calls[0]['kwargs']
    assert call_kwargs.get('trade_date') == "2026-09-10", "LEAKAGE: Candle retrieval trade_date is not 2026-09-10!"
    assert call_kwargs.get('ref_time') == "10:30:00", "LEAKAGE: Candle retrieval ref_time is not 10:30:00!"
    assert "2026-09-11" not in str(call_kwargs), "LEAKAGE: Target date 2026-09-11 was passed to market data retrieval!"

    print("SUCCESS: Zero target/future OHLCV data was queried. Prediction succeeded strictly from reference data.")

    print("\n========================================================")
    print("TEST 2: Weekend Target Timestamp Rejection (Saturday Sep 12)")
    print("========================================================")
    weekend_ts = "2026-09-12 10:30 IST"
    rejected = False
    try:
        predict_movement(symbol=symbol, reference_timestamp=ref_ts, target_timestamp=weekend_ts)
    except ValueError as e:
        rejected = True
        print(f"Correctly caught expected exception: {e}")
        assert "Saturday" in str(e) or "weekend" in str(e)

    assert rejected, "FAILED: System should have rejected Saturday target date!"
    print("SUCCESS: Weekend target was cleanly rejected with informative error message.")

    print("\n========================================================")
    print("TEST 3: Multi-Day Future Target Across Weekend (Sep 10 -> Sep 15)")
    print("========================================================")
    multi_ts = "2026-09-15 10:30 IST" # Tuesday
    res_multi = predict_movement(symbol=symbol, reference_timestamp=ref_ts, target_timestamp=multi_ts)
    print("Multi-day Prediction Result:")
    for k, v in res_multi.items():
        print(f"  {k}: {v}")

    assert res_multi['target_horizon_observed_sessions'] == 3  # Fri Sep 11, Mon Sep 14, Tue Sep 15
    assert res_multi['target_horizon_calendar_days'] == 5
    print("SUCCESS: Multi-day session horizon correctly calculated from business calendar.")

    print("\n========================================================")
    print("ALL FUTURE-TARGET INFERENCE TESTS PASSED COMPLETELY.")
    print("========================================================")

if __name__ == "__main__":
    test_future_target_inference()
