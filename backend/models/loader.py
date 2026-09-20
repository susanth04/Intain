"""
loader.py
=========
Singleton model registry.
Loads ALL trained pkl artifacts once at module import time.
Every pipeline stage imports MODELS from this module — never from disk again.
"""

import os
import sys
import joblib
from pathlib import Path

# ── Locate the loan-perf-engine project root ──────────────────────────────────
# Priority: env var INTAIN_PROJECT_ROOT → auto-discover relative to this file
_env_root = os.environ.get("INTAIN_PROJECT_ROOT", "")
if _env_root and Path(_env_root).exists():
    PROJECT_ROOT = Path(_env_root).resolve()
else:
    # Walk up from this file to find the project by looking for config.yaml
    _here = Path(__file__).resolve()
    for _parent in _here.parents:
        if (_parent / "config.yaml").exists():
            PROJECT_ROOT = _parent
            break
    else:
        # Hardcoded fallback
        PROJECT_ROOT = Path(r"C:\Users\susan\OneDrive\Desktop\Intain\loan-perf-engine").resolve()

# Add project root to sys.path so we can import src.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR  = PROJECT_ROOT / "data" / "processed" / "models"
PROC_DIR    = PROJECT_ROOT / "data" / "processed"
RAW_DIR     = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

import yaml
with open(PROJECT_ROOT / "config.yaml") as _f:
    CFG = yaml.safe_load(_f)

# ── Load all artifacts ────────────────────────────────────────────────────────
print("[loader] Loading model artifacts from:", MODELS_DIR)

MODELS: dict = {}
_files = {
    # Calibrated binary classifiers (used for inference)
    "next_3m_delinquency_flag_cal":    "next_3m_delinquency_flag_lgbm_cal.pkl",
    "next_6m_delinquency_flag_cal":    "next_6m_delinquency_flag_lgbm_cal.pkl",
    "next_12m_default_flag_cal":       "next_12m_default_flag_lgbm_cal.pkl",
    "next_12m_prepayment_flag_cal":    "next_12m_prepayment_flag_lgbm_cal.pkl",
    # Raw LGBM models (used for SHAP — TreeExplainer needs the underlying estimator)
    "next_3m_delinquency_flag_raw":    "next_3m_delinquency_flag_lgbm_raw.pkl",
    "next_6m_delinquency_flag_raw":    "next_6m_delinquency_flag_lgbm_raw.pkl",
    "next_12m_default_flag_raw":       "next_12m_default_flag_lgbm_raw.pkl",
    "next_12m_prepayment_flag_raw":    "next_12m_prepayment_flag_lgbm_raw.pkl",
    # Logistic Regression baselines
    "next_3m_delinquency_flag_lr":     "next_3m_delinquency_flag_lr.pkl",
    "next_6m_delinquency_flag_lr":     "next_6m_delinquency_flag_lr.pkl",
    "next_12m_default_flag_lr":        "next_12m_default_flag_lr.pkl",
    "next_12m_prepayment_flag_lr":     "next_12m_prepayment_flag_lr.pkl",
    # Next-state multiclass
    "next_state_lgbm":                 "next_state_lgbm.pkl",
    "next_state_le":                   "next_state_le.pkl",
    "next_state_lr":                   "next_state_lr.pkl",
    # Anomaly detection
    "isolation_forest":                "isolation_forest.pkl",
    "anomaly_scaler":                  "anomaly_scaler.pkl",
    # Survival
    "cox_ph":                          "cox_ph.pkl",
}
_preprocessors = {
    "imputer":                         "imputer.pkl",
}

_missing = []
for key, fname in {**_files, **_preprocessors}.items():
    path = MODELS_DIR / fname if key != "imputer" else PROC_DIR / fname
    if path.exists():
        MODELS[key] = joblib.load(path)
        print(f"  ✓ {key} loaded ({path.name})")
    else:
        _missing.append(key)
        print(f"  ✗ MISSING: {fname}")

if _missing:
    print(f"[loader] WARNING: {len(_missing)} artifacts missing: {_missing}")
else:
    print(f"[loader] All {len(MODELS)} artifacts loaded successfully.")

# Pre-load static data for fast ingestion
import pandas as pd

print("[loader] Loading raw data into memory …")
_DATA: dict = {}
for key, fname in {
    "train":    "loan_monthly_performance_train.csv",
    "test":     "loan_monthly_performance_test.csv",
    "static":   "loan_static_attributes.csv",
}.items():
    path = RAW_DIR / fname
    if path.exists():
        _DATA[key] = pd.read_csv(path, low_memory=False)
        print(f"  ✓ {fname}: {len(_DATA[key]):,} rows")
    else:
        _DATA[key] = pd.DataFrame()
        print(f"  ✗ MISSING: {fname}")

# Combine train + test for full panel
if not _DATA["train"].empty and not _DATA["test"].empty:
    _DATA["panel"] = pd.concat([_DATA["train"], _DATA["test"]], ignore_index=True)
elif not _DATA["train"].empty:
    _DATA["panel"] = _DATA["train"].copy()
elif not _DATA["test"].empty:
    _DATA["panel"] = _DATA["test"].copy()
else:
    _DATA["panel"] = pd.DataFrame()

print(f"[loader] Full panel: {len(_DATA['panel']):,} rows")

# Pre-load pre-computed anomaly scores for reference
_anomaly_scores_path = PROC_DIR / "anomaly_scores.csv"
if _anomaly_scores_path.exists():
    _DATA["anomaly_scores"] = pd.read_csv(_anomaly_scores_path, low_memory=False)
    print(f"  ✓ anomaly_scores.csv: {len(_DATA['anomaly_scores']):,} rows")
else:
    _DATA["anomaly_scores"] = pd.DataFrame()

# Pre-load validation violations
_violations_path = PROC_DIR / "validation_violations.csv"
if _violations_path.exists():
    _DATA["violations"] = pd.read_csv(_violations_path, low_memory=False)
else:
    _DATA["violations"] = pd.DataFrame()
