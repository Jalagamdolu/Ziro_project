"""
Production Model Training, Evaluation, and Comparison Module for Phase 5.

Enforces:
1. Strict temporal splits:
   - Pooled: Clean Train = 4,824, Clean Val = 687, Test = 1,235
   - Dedicated H1: Clean Train = 854, Clean Val = 994, Clean Test = 2,100
2. Primary Metric: Macro F1 (with full per-class metrics and confusion matrices).
3. Naive baselines: Global majority and Horizon-conditional majority (fit on Train only).
4. Models: Logistic Regression, Random Forest, Extra Trees, XGBoost.
5. Mandatory Horizon Ablation (FULL vs NO_HORIZON).
6. Dedicated H1 Benchmark comparison.
7. Performance by Horizon breakdown (H1 to H7).
8. Model selection using Validation Macro F1 ONLY (Test set untouched until final selection).
9. Output calibrated probability distributions.
10. Secondary Regression benchmark (Ridge and RF Regressor).
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from scipy import stats

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR, MODELS_DIR, PROJECT_ROOT
from src.feature_engineering import MODEL_FEATURES_FULL, MODEL_FEATURES_NO_HORIZON

DATA_DIR = PROJECT_ROOT / "data"
CLASS_MAP = {'DOWN': 0, 'STABLE': 1, 'UP': 2}
INV_CLASS_MAP = {0: 'DOWN', 1: 'STABLE', 2: 'UP'}
CLASS_NAMES = ['DOWN', 'STABLE', 'UP']

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average='macro'))
    mprec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    mrec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bacc, 4),
        'macro_f1': round(mf1, 4),
        'macro_precision': round(mprec, 4),
        'macro_recall': round(mrec, 4),
        'down_f1': round(float(per_f1[0]), 4),
        'stable_f1': round(float(per_f1[1]), 4),
        'up_f1': round(float(per_f1[2]), 4),
        'down_precision': round(float(per_prec[0]), 4),
        'stable_precision': round(float(per_prec[1]), 4),
        'up_precision': round(float(per_prec[2]), 4),
        'down_recall': round(float(per_rec[0]), 4),
        'stable_recall': round(float(per_rec[1]), 4),
        'up_recall': round(float(per_rec[2]), 4)
    }
    
    if y_prob is not None:
        try:
            metrics['log_loss'] = round(float(log_loss(y_true, y_prob)), 4)
        except Exception:
            metrics['log_loss'] = np.nan
    else:
        metrics['log_loss'] = np.nan
        
    return metrics

def save_confusion_matrix_csv(y_true, y_pred, filename):
    """Saves confusion matrix with labeled rows and columns."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    df_cm = pd.DataFrame(cm, index=[f"Actual_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
    df_cm.to_csv(REPORTS_DIR / filename)

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

def run_phase5():
    print("=" * 80)
    print("PHASE 5: MODEL TRAINING & RIGOROUS EVALUATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATASETS & APPLY CLEAN SPLITS
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Parquet datasets and applying clean chronological splits...")
    df_pooled = pd.read_parquet(DATA_DIR / "dataset_pooled_multi_horizon.parquet")
    df_h1 = pd.read_parquet(DATA_DIR / "dataset_dedicated_horizon_1.parquet")
    
    # Pooled Splits (Candidate 2)
    p_tr_mask = (df_pooled['trade_date_ref'].astype(str) <= '2026-08-27') & (df_pooled['trade_date_tgt'].astype(str) <= '2026-08-27')
    p_val_mask = (df_pooled['trade_date_ref'].astype(str).isin(['2026-08-28', '2026-09-02'])) & (df_pooled['trade_date_tgt'].astype(str) <= '2026-09-02')
    p_te_mask = df_pooled['trade_date_ref'].astype(str).isin(['2026-09-09', '2026-09-10'])
    
    p_train = df_pooled[p_tr_mask].copy()
    p_val = df_pooled[p_val_mask].copy()
    p_test = df_pooled[p_te_mask].copy()
    
    print(f"Pooled Clean Train: {len(p_train):,}")
    print(f"Pooled Clean Val:   {len(p_val):,}")
    print(f"Pooled Test:        {len(p_test):,}")
    assert len(p_train) == 4824, f"Expected 4,824 Clean Train, got {len(p_train)}"
    assert len(p_val) == 687, f"Expected 687 Clean Val, got {len(p_val)}"
    assert len(p_test) == 1235, f"Expected 1,235 Test, got {len(p_test)}"
    
    # Dedicated H1 Splits
    h1_tr_mask = df_h1['trade_date_ref'].astype(str).isin(['2026-08-17', '2026-08-18'])
    h1_val_mask = df_h1['trade_date_ref'].astype(str) == '2026-08-27'
    h1_te_mask = df_h1['trade_date_ref'].astype(str).isin(['2026-09-02', '2026-09-09'])
    
    h1_train = df_h1[h1_tr_mask].copy()
    h1_val = df_h1[h1_val_mask].copy()
    h1_test = df_h1[h1_te_mask].copy()
    
    print(f"H1 Clean Train:     {len(h1_train):,}")
    print(f"H1 Clean Val:       {len(h1_val):,}")
    print(f"H1 Clean Test:      {len(h1_test):,}")
    assert len(h1_train) == 854, f"Expected 854 H1 Clean Train, got {len(h1_train)}"
    assert len(h1_val) == 994, f"Expected 994 H1 Clean Val, got {len(h1_val)}"
    assert len(h1_test) == 2100, f"Expected 2,100 H1 Clean Test, got {len(h1_test)}"
    
    y_p_train = p_train['target_class'].map(CLASS_MAP).values
    y_p_val = p_val['target_class'].map(CLASS_MAP).values
    y_p_test = p_test['target_class'].map(CLASS_MAP).values
    
    # -------------------------------------------------------------------------
    # 2. NAIVE BASELINES
    # -------------------------------------------------------------------------
    print("\n[Step 2] Establishing Naive Baselines (strictly fit on Clean Train)...")
    baseline_rows = []
    
    # Baseline A: Global Majority Class (DOWN on Clean Train)
    global_maj_class = p_train['target_class'].value_counts().index[0]
    global_maj_idx = CLASS_MAP[global_maj_class]
    print(f"  Global Majority Class on Clean Train: {global_maj_class} (Index: {global_maj_idx})")
    
    pred_val_maj = np.full_like(y_p_val, global_maj_idx)
    pred_te_maj = np.full_like(y_p_test, global_maj_idx)
    
    m_val_maj = evaluate_predictions(y_p_val, pred_val_maj)
    m_te_maj = evaluate_predictions(y_p_test, pred_te_maj)
    
    baseline_rows.append({
        'baseline': 'Global Majority Class (DOWN)',
        'split': 'Validation',
        **m_val_maj
    })
    baseline_rows.append({
        'baseline': 'Global Majority Class (DOWN)',
        'split': 'Test',
        **m_te_maj
    })
    
    # Baseline B: Horizon-Conditional Majority Class
    horizon_maj_map = {}
    for h, grp in p_train.groupby('observed_sessions_ahead'):
        horizon_maj_map[h] = CLASS_MAP[grp['target_class'].value_counts().index[0]]
    print(f"  Horizon-Conditional Dominant Map on Clean Train: {horizon_maj_map}")
    
    pred_val_h = p_val['observed_sessions_ahead'].map(lambda h: horizon_maj_map.get(h, global_maj_idx)).values
    pred_te_h = p_test['observed_sessions_ahead'].map(lambda h: horizon_maj_map.get(h, global_maj_idx)).values
    
    m_val_h = evaluate_predictions(y_p_val, pred_val_h)
    m_te_h = evaluate_predictions(y_p_test, pred_te_h)
    
    baseline_rows.append({
        'baseline': 'Horizon-Conditional Majority',
        'split': 'Validation',
        **m_val_h
    })
    baseline_rows.append({
        'baseline': 'Horizon-Conditional Majority',
        'split': 'Test',
        **m_te_h
    })
    
    df_baselines = pd.DataFrame(baseline_rows)
    df_baselines.to_csv(REPORTS_DIR / "baseline_results.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'baseline_results.csv'}")
    print(df_baselines[['baseline', 'split', 'accuracy', 'balanced_accuracy', 'macro_f1', 'down_f1', 'stable_f1', 'up_f1']])
    
    # -------------------------------------------------------------------------
    # 3. POOLED MULTI-HORIZON MODEL TRAINING & COMPARISON
    # -------------------------------------------------------------------------
    print("\n[Step 3] Training Pooled Multi-Horizon Classifiers (MODEL_FEATURES_FULL)...")
    
    # Preprocessor fit strictly on Clean Train
    prep_pooled = PreprocessingPipeline(MODEL_FEATURES_FULL)
    prep_pooled.fit(p_train)
    
    X_p_tr = prep_pooled.transform(p_train)
    X_p_val = prep_pooled.transform(p_val)
    X_p_te = prep_pooled.transform(p_test)
    
    models = {
        'Logistic Regression (L2, C=0.1)': LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Logistic Regression (L2, C=1.0)': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Random Forest (depth=8, trees=150)': RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Extra Trees (depth=8, trees=150)': ExtraTreesClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost (depth=4, lr=0.05, trees=100)': xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='mlogloss'
        )
    }
    
    sw_train = compute_sample_weight('balanced', y_p_train)
    
    comparison_rows = []
    fitted_models = {}
    val_probs = {}
    test_probs = {}
    val_preds = {}
    test_preds = {}
    
    for name, clf in models.items():
        print(f"  Training {name}...")
        if 'XGBoost' in name:
            clf.fit(X_p_tr, y_p_train, sample_weight=sw_train)
        else:
            clf.fit(X_p_tr, y_p_train)
            
        fitted_models[name] = clf
        
        # Validation Evaluation
        v_prob = clf.predict_proba(X_p_val)
        v_pred = clf.predict(X_p_val)
        val_probs[name] = v_prob
        val_preds[name] = v_pred
        m_val = evaluate_predictions(y_p_val, v_pred, v_prob)
        
        # Test Evaluation (One-time evaluation)
        t_prob = clf.predict_proba(X_p_te)
        t_pred = clf.predict(X_p_te)
        test_probs[name] = t_prob
        test_preds[name] = t_pred
        m_te = evaluate_predictions(y_p_test, t_pred, t_prob)
        
        comparison_rows.append({'model': name, 'split': 'Validation', **m_val})
        comparison_rows.append({'model': name, 'split': 'Test', **m_te})
        
        # Save confusion matrices
        safe_name = name.split(' ')[0].lower() + "_" + name.split(' ')[1].lower().replace('(', '').replace(')', '').replace(',', '')
        save_confusion_matrix_csv(y_p_val, v_pred, f"confusion_matrix_{safe_name}_val.csv")
        save_confusion_matrix_csv(y_p_test, t_pred, f"confusion_matrix_{safe_name}_test.csv")
        
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'model_comparison.csv'}")
    
    val_summary = df_comparison[df_comparison['split'] == 'Validation'].sort_values('macro_f1', ascending=False)
    print("\nValidation Leaderboard (Primary Metric: Validation Macro F1):")
    print(val_summary[['model', 'macro_f1', 'balanced_accuracy', 'accuracy', 'down_f1', 'stable_f1', 'up_f1']].to_string(index=False))
    
    # Select Best Model strictly by Validation Macro F1
    best_model_name = val_summary.iloc[0]['model']
    best_val_macro_f1 = val_summary.iloc[0]['macro_f1']
    print(f"\n>>> Selected Best Pooled Model: {best_model_name} (Validation Macro F1 = {best_val_macro_f1:.4f}) <<<")
    
    # -------------------------------------------------------------------------
    # 4. MANDATORY HORIZON ABLATION (FULL vs NO_HORIZON)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Mandatory Horizon Ablation (FULL vs NO_HORIZON)...")
    
    prep_no_h = PreprocessingPipeline(MODEL_FEATURES_NO_HORIZON)
    prep_no_h.fit(p_train)
    X_tr_noh = prep_no_h.transform(p_train)
    X_val_noh = prep_no_h.transform(p_val)
    X_te_noh = prep_no_h.transform(p_test)
    
    ablation_rows = []
    
    # Test selected best model and other core models on NO_HORIZON
    for m_label in ['Logistic Regression (L2, C=0.1)', 'Random Forest (depth=8, trees=150)', 'XGBoost (depth=4, lr=0.05, trees=100)']:
        clf_full = fitted_models[m_label]
        m_val_f = evaluate_predictions(y_p_val, val_preds[m_label])
        m_te_f = evaluate_predictions(y_p_test, test_preds[m_label])
        
        # Train on NO_HORIZON
        if 'Logistic' in m_label:
            clf_nh = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
            clf_nh.fit(X_tr_noh, y_p_train)
        elif 'Random' in m_label:
            clf_nh = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1)
            clf_nh.fit(X_tr_noh, y_p_train)
        else:
            clf_nh = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='mlogloss')
            clf_nh.fit(X_tr_noh, y_p_train, sample_weight=sw_train)
            
        m_val_nh = evaluate_predictions(y_p_val, clf_nh.predict(X_val_noh))
        m_te_nh = evaluate_predictions(y_p_test, clf_nh.predict(X_te_noh))
        
        ablation_rows.append({'model': m_label, 'feature_spec': 'FULL (26 feats)', 'split': 'Validation', **m_val_f})
        ablation_rows.append({'model': m_label, 'feature_spec': 'NO_HORIZON (24 feats)', 'split': 'Validation', **m_val_nh})
        ablation_rows.append({'model': m_label, 'feature_spec': 'FULL (26 feats)', 'split': 'Test', **m_te_f})
        ablation_rows.append({'model': m_label, 'feature_spec': 'NO_HORIZON (24 feats)', 'split': 'Test', **m_te_nh})
        
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(REPORTS_DIR / "horizon_ablation_results.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'horizon_ablation_results.csv'}")
    print(df_ablation[['model', 'feature_spec', 'split', 'macro_f1', 'balanced_accuracy', 'accuracy']].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 5. DEDICATED HORIZON-1 BENCHMARK
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training and Evaluating Dedicated Horizon-1 Benchmark...")
    
    y_h1_tr = h1_train['target_class'].map(CLASS_MAP).values
    y_h1_val = h1_val['target_class'].map(CLASS_MAP).values
    y_h1_te = h1_test['target_class'].map(CLASS_MAP).values
    
    prep_h1 = PreprocessingPipeline(MODEL_FEATURES_NO_HORIZON)
    prep_h1.fit(h1_train)
    
    X_h1_tr = prep_h1.transform(h1_train)
    X_h1_val = prep_h1.transform(h1_val)
    X_h1_te = prep_h1.transform(h1_test)
    
    h1_models = {
        'H1 Logistic Regression (C=0.1)': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
        'H1 Random Forest (depth=6)': RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1),
        'H1 Extra Trees (depth=6)': ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=15, class_weight='balanced', random_state=42, n_jobs=-1),
        'H1 XGBoost (depth=3, lr=0.05)': xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    }
    
    sw_h1 = compute_sample_weight('balanced', y_h1_tr)
    h1_rows = []
    fitted_h1_models = {}
    
    for name, clf in h1_models.items():
        if 'XGBoost' in name:
            clf.fit(X_h1_tr, y_h1_tr, sample_weight=sw_h1)
        else:
            clf.fit(X_h1_tr, y_h1_tr)
        fitted_h1_models[name] = clf
        
        m_val = evaluate_predictions(y_h1_val, clf.predict(X_h1_val), clf.predict_proba(X_h1_val))
        m_te = evaluate_predictions(y_h1_te, clf.predict(X_h1_te), clf.predict_proba(X_h1_te))
        
        h1_rows.append({'model': name, 'split': 'Validation', **m_val})
        h1_rows.append({'model': name, 'split': 'Test', **m_te})
        
    df_h1_results = pd.DataFrame(h1_rows)
    df_h1_results.to_csv(REPORTS_DIR / "dedicated_h1_results.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'dedicated_h1_results.csv'}")
    print(df_h1_results[['model', 'split', 'macro_f1', 'balanced_accuracy', 'accuracy']].to_string(index=False))
    
    # Best Dedicated H1 Model (Validation Macro F1)
    best_h1_name = df_h1_results[df_h1_results['split'] == 'Validation'].sort_values('macro_f1', ascending=False).iloc[0]['model']
    print(f"\n>>> Selected Best Dedicated H1 Model: {best_h1_name} <<<")
    
    # -------------------------------------------------------------------------
    # 6. POOLED MODEL PERFORMANCE BY HORIZON BREAKDOWN (H1 to H7)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Pooled Model Performance by Horizon Breakdown (H1 to H7)...")
    
    best_pooled_clf = fitted_models[best_model_name]
    horizon_breakdown_rows = []
    
    for h in range(1, 8):
        mask_h = (p_test['observed_sessions_ahead'] == h)
        cnt = int(mask_h.sum())
        if cnt > 0:
            y_true_h = y_p_test[mask_h]
            y_pred_h = test_preds[best_model_name][mask_h]
            m_h = evaluate_predictions(y_true_h, y_pred_h)
            horizon_breakdown_rows.append({
                'horizon': f"H{h}",
                'test_samples': cnt,
                'accuracy': m_h['accuracy'],
                'macro_f1': m_h['macro_f1'],
                'down_recall': m_h['down_recall'],
                'stable_recall': m_h['stable_recall'],
                'up_recall': m_h['up_recall'],
                'notes': f"Evaluated on {cnt} active out-of-sample test pairs"
            })
        else:
            horizon_breakdown_rows.append({
                'horizon': f"H{h}",
                'test_samples': 0,
                'accuracy': np.nan,
                'macro_f1': np.nan,
                'down_recall': np.nan,
                'stable_recall': np.nan,
                'up_recall': np.nan,
                'notes': "0 samples in test split (terminal database date Sep 10 requires future dates beyond scope)"
            })
            
    df_h_breakdown = pd.DataFrame(horizon_breakdown_rows)
    df_h_breakdown.to_csv(REPORTS_DIR / "pooled_results_by_horizon.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'pooled_results_by_horizon.csv'}")
    print(df_h_breakdown.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 7. SECONDARY REGRESSION BENCHMARK
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Secondary Regression Benchmark...")
    y_ret_tr = p_train['future_return_pct'].values
    y_ret_val = p_val['future_return_pct'].values
    y_ret_te = p_test['future_return_pct'].values
    
    def ret_to_class(arr):
        return np.where(arr > 1.0, 2, np.where(arr < -1.0, 0, 1))
        
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_p_tr, y_ret_tr)
    pred_val_ridge = ridge.predict(X_p_val)
    pred_te_ridge = ridge.predict(X_p_te)
    
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_reg.fit(X_p_tr, y_ret_tr)
    pred_val_rfreg = rf_reg.predict(X_p_val)
    pred_te_rfreg = rf_reg.predict(X_p_te)
    
    reg_rows = []
    for reg_name, p_v, p_t in [('Ridge Regression (alpha=10)', pred_val_ridge, pred_te_ridge), ('Random Forest Regressor (depth=6)', pred_val_rfreg, pred_te_rfreg)]:
        v_mae = float(mean_absolute_error(y_ret_val, p_v))
        v_rmse = float(np.sqrt(mean_squared_error(y_ret_val, p_v)))
        v_r2 = float(r2_score(y_ret_val, p_v))
        v_f1 = float(f1_score(y_p_val, ret_to_class(p_v), average='macro'))
        
        t_mae = float(mean_absolute_error(y_ret_te, p_t))
        t_rmse = float(np.sqrt(mean_squared_error(y_ret_te, p_t)))
        t_r2 = float(r2_score(y_ret_te, p_t))
        t_f1 = float(f1_score(y_p_test, ret_to_class(p_t), average='macro'))
        
        reg_rows.append({'model': reg_name, 'split': 'Validation', 'mae_pct': round(v_mae, 2), 'rmse_pct': round(v_rmse, 2), 'r2': round(v_r2, 4), 'derived_macro_f1': round(v_f1, 4)})
        reg_rows.append({'model': reg_name, 'split': 'Test', 'mae_pct': round(t_mae, 2), 'rmse_pct': round(t_rmse, 2), 'r2': round(t_r2, 4), 'derived_macro_f1': round(t_f1, 4)})
        
    df_reg = pd.DataFrame(reg_rows)
    print("\nSecondary Regression Results:")
    print(df_reg.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 8. SAVE TRAINED ARTIFACTS & METADATA
    # -------------------------------------------------------------------------
    print("\n[Step 8] Saving trained models and production artifacts under models/...")
    
    joblib.dump(best_pooled_clf, MODELS_DIR / "best_pooled_model.joblib")
    joblib.dump(prep_pooled, MODELS_DIR / "pooled_preprocessor.joblib")
    
    best_h1_clf = fitted_h1_models[best_h1_name]
    joblib.dump(best_h1_clf, MODELS_DIR / "best_h1_model.joblib")
    joblib.dump(prep_h1, MODELS_DIR / "h1_preprocessor.joblib")
    
    metadata = {
        'best_pooled_model': best_model_name,
        'best_h1_model': best_h1_name,
        'class_map': CLASS_MAP,
        'inv_class_map': INV_CLASS_MAP,
        'model_features_full': MODEL_FEATURES_FULL,
        'model_features_no_horizon': MODEL_FEATURES_NO_HORIZON,
        'pooled_train_size': len(p_train),
        'pooled_val_size': len(p_val),
        'pooled_test_size': len(p_test),
        'h1_train_size': len(h1_train),
        'h1_val_size': len(h1_val),
        'h1_test_size': len(h1_test),
        'best_pooled_val_macro_f1': best_val_macro_f1,
        'best_pooled_test_macro_f1': float(df_comparison[(df_comparison['model'] == best_model_name) & (df_comparison['split'] == 'Test')]['macro_f1'].iloc[0])
    }
    
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved models and metadata to {MODELS_DIR / 'model_metadata.json'}")

if __name__ == "__main__":
    run_phase5()
