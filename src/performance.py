"""
Performance Evaluation & Monitoring Engine for Paper Trading.

Calculates:
- Primary Metric: Macro F1
- Multi-Class Metrics: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall
- Per-Class Breakdown: DOWN, STABLE, UP (Precision, Recall, F1)
- Confusion Matrix
- Horizon-wise performance (H1 to H7) with explicit 'NOT ENOUGH DATA' markers
- Time-based monitoring (Daily counts, rolling accuracy, rolling recalls)
- Confidence calibration buckets (0.50-0.60, 0.60-0.70, etc.)
- Clear separation between Historical Test Performance and Forward Paper Performance
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)
from sqlalchemy import text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_trading import get_engine, list_predictions

CLASS_LABELS = ['DOWN', 'STABLE', 'UP']
CLASS_INDEX = {'DOWN': 0, 'STABLE': 1, 'UP': 2}

# Reference Phase 5 Historical Out-of-Sample Test Baseline (Sep 09 -> Sep 10, N=1,235)
HISTORICAL_TEST_BENCHMARK = {
    "split_name": "Phase 5 Historical Test Set (Candidate 2)",
    "sample_count": 1235,
    "accuracy": 0.4640,
    "balanced_accuracy": 0.3869,
    "macro_f1": 0.3563,
    "down_recall": 0.5144,
    "stable_recall": 0.6103,
    "up_recall": 0.0359,
    "notes": "Static chronological evaluation on historical NSE market data (Sep 09 to Sep 10, 2026)"
}

def compute_classification_metrics(df_resolved: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes primary Macro F1, accuracy, balanced accuracy, and per-class metrics.
    """
    if df_resolved.empty or len(df_resolved) == 0:
        return {
            "status": "NO_RESOLVED_DATA",
            "sample_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "per_class": {},
            "confusion_matrix": None
        }

    y_true = df_resolved['actual_class'].astype(str).values
    y_pred = df_resolved['predicted_class'].astype(str).values

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average='macro', zero_division=0
    )

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()

    per_class = {}
    for i, cls in enumerate(CLASS_LABELS):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    return {
        "status": "VALID",
        "sample_count": int(len(df_resolved)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": cm
        }
    }

def compute_metrics_by_horizon(df_resolved: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes metrics separately for Horizons 1 through 7.
    Marks horizons with 0 or insufficient observations as 'NOT ENOUGH DATA'.
    """
    horizon_results = []
    
    for h in range(1, 8):
        df_h = df_resolved[df_resolved['observed_sessions_ahead'] == h] if not df_resolved.empty else pd.DataFrame()
        count = len(df_h)
        
        if count == 0:
            horizon_results.append({
                "horizon": f"H{h}",
                "sample_count": 0,
                "status": "NOT ENOUGH DATA",
                "accuracy": "NOT ENOUGH DATA",
                "macro_f1": "NOT ENOUGH DATA",
                "down_recall": "NOT ENOUGH DATA",
                "stable_recall": "NOT ENOUGH DATA",
                "up_recall": "NOT ENOUGH DATA",
                "notes": "Zero resolved forward paper observations recorded yet."
            })
        elif count < 5:
            # Under 5 observations: metrics are unstable, report count and mark low sample warning
            y_true = df_h['actual_class'].values
            y_pred = df_h['predicted_class'].values
            acc = float(accuracy_score(y_true, y_pred))
            horizon_results.append({
                "horizon": f"H{h}",
                "sample_count": count,
                "status": "LOW SAMPLE (PRELIMINARY)",
                "accuracy": round(acc, 4),
                "macro_f1": "NOT ENOUGH DATA",
                "down_recall": "NOT ENOUGH DATA",
                "stable_recall": "NOT ENOUGH DATA",
                "up_recall": "NOT ENOUGH DATA",
                "notes": f"Sample size (N={count}) too small for reliable multi-class F1."
            })
        else:
            m = compute_classification_metrics(df_h)
            horizon_results.append({
                "horizon": f"H{h}",
                "sample_count": count,
                "status": "EVALUATED",
                "accuracy": m['accuracy'],
                "macro_f1": m['macro_f1'],
                "down_recall": m['per_class']['DOWN']['recall'],
                "stable_recall": m['per_class']['STABLE']['recall'],
                "up_recall": m['per_class']['UP']['recall'],
                "notes": f"Evaluated across {count} resolved forward observations."
            })
            
    return horizon_results

def compute_confidence_analysis(df_resolved: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Groups predictions by confidence probability:
    0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.00
    """
    if df_resolved.empty:
        return []

    # Calculate max predicted probability (confidence)
    probs = df_resolved[['probability_down', 'probability_stable', 'probability_up']].values
    max_probs = np.max(probs, axis=1)
    df_calc = df_resolved.copy()
    df_calc['confidence'] = max_probs
    df_calc['correct'] = (df_calc['predicted_class'] == df_calc['actual_class']).astype(int)

    bins = [0.33, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = ['0.33–0.50', '0.50–0.60', '0.60–0.70', '0.70–0.80', '0.80–0.90', '0.90–1.00']
    df_calc['conf_bucket'] = pd.cut(df_calc['confidence'], bins=bins, labels=labels, right=False)

    bucket_results = []
    for bucket in labels:
        df_b = df_calc[df_calc['conf_bucket'] == bucket]
        cnt = len(df_b)
        if cnt == 0:
            bucket_results.append({
                "confidence_tier": bucket,
                "prediction_count": 0,
                "accuracy": None,
                "macro_f1": None,
                "notes": "No predictions in this confidence tier"
            })
        else:
            acc = float(accuracy_score(df_b['actual_class'], df_b['predicted_class']))
            f1 = float(f1_score(df_b['actual_class'], df_b['predicted_class'], labels=CLASS_LABELS, average='macro', zero_division=0)) if cnt >= 5 else None
            bucket_results.append({
                "confidence_tier": bucket,
                "prediction_count": cnt,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4) if f1 is not None else "NOT ENOUGH DATA",
                "notes": f"N={cnt}" + (" (low sample caution)" if cnt < 10 else "")
            })

    return bucket_results

def compute_performance_over_time(df_resolved: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes time-based metrics:
    Daily prediction volume, daily accuracy, and rolling recall trends.
    """
    if df_resolved.empty:
        return []

    df_calc = df_resolved.copy()
    df_calc['date'] = pd.to_datetime(df_calc['target_timestamp']).dt.strftime('%Y-%m-%d')
    df_calc['correct'] = (df_calc['predicted_class'] == df_calc['actual_class']).astype(int)

    daily = []
    for date, group in df_calc.groupby('date', sort=True):
        cnt = len(group)
        acc = float(accuracy_score(group['actual_class'], group['predicted_class']))
        
        # Per-class recalls
        y_true = group['actual_class'].values
        y_pred = group['predicted_class'].values
        _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
        
        daily.append({
            "target_date": str(date),
            "prediction_count": cnt,
            "accuracy": round(acc, 4),
            "down_recall": round(float(rec[0]), 4),
            "stable_recall": round(float(rec[1]), 4),
            "up_recall": round(float(rec[2]), 4)
        })

    return daily

def generate_paper_performance_summary(engine_override: Optional[Engine] = None) -> Dict[str, Any]:
    """
    Assembles comprehensive forward paper performance evaluation summary.
    Strictly separates Forward Paper Performance from Historical Test Performance.
    """
    all_preds = list_predictions(limit=5000, engine_override=engine_override)
    df_all = pd.DataFrame(all_preds) if all_preds else pd.DataFrame()

    total_count = len(df_all)
    if total_count == 0:
        return {
            "total_predictions": 0,
            "open_predictions": 0,
            "resolved_predictions": 0,
            "expired_predictions": 0,
            "forward_paper_metrics": compute_classification_metrics(pd.DataFrame()),
            "by_horizon": compute_metrics_by_horizon(pd.DataFrame()),
            "confidence_tiers": [],
            "performance_over_time": [],
            "historical_benchmark": HISTORICAL_TEST_BENCHMARK
        }

    df_resolved = df_all[df_all['status'] == 'RESOLVED']
    df_open = df_all[df_all['status'] == 'OPEN']
    df_expired = df_all[df_all['status'] == 'EXPIRED']

    return {
        "total_predictions": int(total_count),
        "open_predictions": int(len(df_open)),
        "resolved_predictions": int(len(df_resolved)),
        "expired_predictions": int(len(df_expired)),
        "forward_paper_metrics": compute_classification_metrics(df_resolved),
        "by_horizon": compute_metrics_by_horizon(df_resolved),
        "confidence_tiers": compute_confidence_analysis(df_resolved),
        "performance_over_time": compute_performance_over_time(df_resolved),
        "historical_benchmark": HISTORICAL_TEST_BENCHMARK
    }
