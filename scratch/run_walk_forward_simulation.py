"""
Walk-Forward Paper Trading Simulation Runner.

Simulates paper trading deployment:
1. Ingests reference timestamps on Sep 09 -> Sep 10 (which have resolved market data in DB).
2. Generates predictions strictly at ref time, saves as OPEN, then resolves against Sep 10 10:30 market data.
3. Ingests forward OPEN predictions from Sep 10 -> Sep 11 and Sep 15 (future targets beyond DB).
4. Verifies metrics, horizon tables, and persistence.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_trading import (
    create_paper_prediction,
    resolve_all_pending,
    list_predictions,
    get_legitimate_symbols
)
from src.performance import generate_paper_performance_summary

def run_simulation():
    print("=" * 60)
    print("STARTING PHASE 6 WALK-FORWARD PAPER TRADING SIMULATION")
    print("=" * 60)

    # 1. Select a diverse sample of legitimate liquid symbols
    symbols = ["360ONE", "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT"]
    legit = get_legitimate_symbols()
    valid_symbols = [s for s in symbols if s in legit]
    print(f"Running simulation on {len(valid_symbols)} legitimate symbols: {valid_symbols}")

    # 2. Generate H1 predictions (Ref: Sep 09 10:30 -> Target: Sep 10 10:30)
    print("\n--- STEP 1: Generating H1 Paper Predictions (Sep 09 -> Sep 10) ---")
    created_h1 = []
    for sym in valid_symbols:
        try:
            rec = create_paper_prediction(
                symbol=sym,
                reference_timestamp="2026-09-09 10:30 IST",
                target_timestamp="2026-09-10 10:30 IST",
                model_type="pooled"
            )
            created_h1.append(rec)
            print(f"  [OPEN] {rec['prediction_id']} | {sym:12} | Ref: {rec['reference_price']:8.2f} | Pred: {rec['predicted_class']} (p_down={rec['probability_down']:.2f}, p_st={rec['probability_stable']:.2f}, p_up={rec['probability_up']:.2f})")
        except Exception as e:
            print(f"  [SKIP] {sym}: {e}")

    # 3. Generate future OPEN predictions beyond database (Ref: Sep 10 10:30 -> Target: Sep 11 and Sep 15)
    print("\n--- STEP 2: Generating Future OPEN Paper Predictions (Sep 10 -> Sep 11 & Sep 15) ---")
    created_future = []
    for sym in valid_symbols[:5]:
        # H1 future target
        try:
            rec = create_paper_prediction(
                symbol=sym,
                reference_timestamp="2026-09-10 10:30 IST",
                target_timestamp="2026-09-11 10:30 IST",
                model_type="pooled"
            )
            created_future.append(rec)
            print(f"  [OPEN] {rec['prediction_id']} | {sym:12} | H=1 (Sep 11) | Pred: {rec['predicted_class']}")
        except Exception as e:
            print(f"  [SKIP] {sym} Sep 11: {e}")

        # H3 future multi-day target across weekend
        try:
            rec3 = create_paper_prediction(
                symbol=sym,
                reference_timestamp="2026-09-10 10:30 IST",
                target_timestamp="2026-09-15 10:30 IST",
                model_type="pooled"
            )
            created_future.append(rec3)
            print(f"  [OPEN] {rec3['prediction_id']} | {sym:12} | H=3 (Sep 15) | Pred: {rec3['predicted_class']}")
        except Exception as e:
            print(f"  [SKIP] {sym} Sep 15: {e}")

    # 4. Resolve predictions whose target is Sep 10 (in DB)
    print("\n--- STEP 3: Resolving Pending Predictions Against Realized Market Data ---")
    res_counts = resolve_all_pending(as_of_time="2026-09-10 15:30:00")
    print(f"Resolution results: {res_counts}")

    # 5. Generate performance summary
    print("\n--- STEP 4: Evaluating Forward Paper Performance ---")
    summary = generate_paper_performance_summary()
    print(f"Total Predictions  : {summary['total_predictions']}")
    print(f"Resolved           : {summary['resolved_predictions']}")
    print(f"Open               : {summary['open_predictions']}")
    print(f"Expired            : {summary['expired_predictions']}")

    fwd = summary['forward_paper_metrics']
    print(f"\nForward Metrics:")
    print(f"  Accuracy         : {fwd['accuracy']}")
    print(f"  Balanced Accuracy: {fwd['balanced_accuracy']}")
    print(f"  Macro F1         : {fwd['macro_f1']}")
    print(f"  Macro Precision  : {fwd['macro_precision']}")
    print(f"  Macro Recall     : {fwd['macro_recall']}")
    print(f"  Per-Class        : {fwd['per_class']}")

    print("\nPerformance by Horizon:")
    for h in summary['by_horizon']:
        print(f"  {h['horizon']}: count={h['sample_count']}, status={h['status']}, acc={h['accuracy']}, macro_f1={h['macro_f1']}")

    print("\nComparison with Phase 5 Historical Test Baseline:")
    hb = summary['historical_benchmark']
    print(f"  Historical Test Macro F1: {hb['macro_f1']} (Accuracy: {hb['accuracy']})")
    print(f"  Forward Paper Macro F1  : {fwd['macro_f1']} (Accuracy: {fwd['accuracy']})")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_simulation()
