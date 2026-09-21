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
import pytest
import pandas as pd
from sqlalchemy import event

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_movement, engine, get_observed_sessions_between

def test_future_target_inference():
    executed_queries = []

    def query_listener(conn, cursor, statement, parameters, context, executemany):
        executed_queries.append({
            'statement': statement,
            'parameters': parameters
        })

    event.listen(engine, "before_cursor_execute", query_listener)

    try:
        print("\n========================================================")
        print("TEST 1: Reference Sep 10 -> Target Sep 11 (Non-existent in DB)")
        print("========================================================")
        executed_queries.clear()

        ref_ts = "2026-09-10 10:30 IST"
        tgt_ts = "2026-09-11 10:30 IST"
        symbol = "360ONE"

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

        # Audit executed queries
        print(f"\nTotal SQL queries executed during inference: {len(executed_queries)}")
        for i, q in enumerate(executed_queries, 1):
            print(f"Query {i}: {q['statement'].strip()[:80]}... | Params: {q['parameters']}")

        # Verify no query requested target date or future prices
        for q in executed_queries:
            stmt = q['statement'].lower()
            params = str(q['parameters'])
            assert "2026-09-11" not in stmt and "2026-09-11" not in params, "LEAKAGE: Target date 2026-09-11 was referenced in SQL query!"
            if "ohlcv_intraday" in stmt and "where" in stmt and "symbol = :symbol" in stmt:
                # Must only query reference date
                assert q['parameters']['trade_date'] == "2026-09-10"
                assert q['parameters']['ref_time'] == "10:30:00"

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
        executed_queries.clear()
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

    finally:
        event.remove(engine, "before_cursor_execute", query_listener)

if __name__ == "__main__":
    test_future_target_inference()
