"""
Data Quality and Leakage Assertion Module for NSE Intraday ML Project.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXCLUDED_SYNTHETIC_SYMBOLS

def assert_temporal_integrity(df: pd.DataFrame) -> None:
    """Asserts target timestamps are strictly in the future of reference timestamps."""
    assert (df['ts_tgt'] > df['ts_ref']).all(), "CRITICAL LEAKAGE: target_ts <= reference_ts"
    assert (df['calendar_days_ahead'] > 0).all(), "CRITICAL: calendar_days_ahead <= 0"
    assert (df['observed_sessions_ahead'] > 0).all(), "CRITICAL: observed_sessions_ahead <= 0"

def assert_no_target_leakage(feature_cols: list, forbidden_targets: list) -> None:
    """Asserts no target or label variables are present in the feature matrix."""
    for f in feature_cols:
        assert f not in forbidden_targets, f"CRITICAL LEAKAGE: Forbidden target column '{f}' in feature set"

def assert_session_bound_integrity(df: pd.DataFrame) -> None:
    """Asserts session-so-far statistics are valid and non-leaking."""
    assert (df['session_high_so_far'] >= df['session_low_so_far']).all(), "Session high < session low"
    assert (df['reference_close'] > 0).all(), "Non-positive reference close"
    assert (df['target_close'] > 0).all(), "Non-positive target close"

def assert_numeric_sanity(df: pd.DataFrame, feature_cols: list) -> None:
    """Asserts zero infinite values in engineered features."""
    for f in feature_cols:
        assert not np.isinf(df[f]).any(), f"CRITICAL: Infinite values detected in feature '{f}'"

def assert_symbol_cleanliness(df: pd.DataFrame) -> None:
    """Asserts no excluded synthetic/test symbols are present."""
    assert not df['symbol'].isin(EXCLUDED_SYNTHETIC_SYMBOLS).any(), "CRITICAL: Synthetic symbol present"

def run_all_data_quality_assertions(df: pd.DataFrame, feature_cols: list, forbidden_targets: list) -> None:
    """Executes complete suite of pipeline assertions."""
    print("Executing complete Data Quality & Leakage assertion suite...")
    assert_temporal_integrity(df)
    assert_no_target_leakage(feature_cols, forbidden_targets)
    assert_session_bound_integrity(df)
    assert_numeric_sanity(df, feature_cols)
    assert_symbol_cleanliness(df)
    print("All Data Quality & Leakage assertions PASSED (100% compliant).")
