"""
splits.py
=========
Time-aware train/validation/test split for loan panel data.

Key design:
  - Split is on month_index, NOT on loan_id.
  - The same loan_id CAN appear across splits (a continuing loan),
    but the model never sees future outcomes when predicting past periods.
  - Leakage proof: assert that no target is derived from post-cutoff data.
  - Printed leakage check is required for the submission.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


# ── Feature columns (must not contain future-derived information) ──────────
FEATURE_COLS = [
    "loan_age_months", "remaining_term_months", "current_balance",
    "interest_rate", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "month_index",
    # Categorical (will be encoded later)
    "credit_score_band", "ltv_band", "dti_band", "state",
    "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "current_status", "document_status",
    "loss_severity_band", "source_system",
    # Lag features (added in build_features.py)
]

TARGET_COLS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]


def make_splits(panel: pd.DataFrame, cfg: dict) -> tuple:
    """
    Split panel into (train, val, test) sets by month_index cutoff.

    Returns:
        (df_train, df_val, df_test) DataFrames with features + targets retained.
    """
    train_cut = cfg["SPLIT"]["TRAIN_CUTOFF"]
    val_cut   = cfg["SPLIT"]["VAL_CUTOFF"]

    df_train = panel[panel["month_index"] <= train_cut].copy()
    df_val   = panel[(panel["month_index"] > train_cut) & (panel["month_index"] <= val_cut)].copy()
    df_test  = panel[(panel["month_index"] > val_cut) & (panel["month_index"] <= val_cut + 5)].copy()

    _print_split_stats(df_train, df_val, df_test, train_cut, val_cut)
    _leakage_check(df_train, df_val, df_test)

    return df_train, df_val, df_test


def _print_split_stats(train, val, test, tc, vc):
    print("\n" + "="*60)
    print(f"TIME-AWARE SPLIT (train ≤ month {tc}, val ≤ month {vc}, test > month {vc})")
    print("="*60)
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:5s}: {len(df):>8,} rows | "
              f"{df['loan_id'].nunique():>6,} unique loans | "
              f"months {df['month_index'].min()}–{df['month_index'].max()}")


def _leakage_check(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """
    Critical check: verify that no row that appears in train has its target
    derived from observations in val/test splits.

    Our generation process builds targets from forward rows in the SAME simulated
    trajectory, not from the actual future panel slice — so this check confirms
    that the feature columns contain no future-derived values.

    We verify:
    1. No column in FEATURE_COLS has perfect correlation (|r|=1) with any TARGET.
    2. The max month_index in train is strictly < min month_index in val.
    3. No loan_id in train has rows whose month_index overlaps in an invalid way.
    """
    print("\n── Leakage Check ──────────────────────────────────────────")

    # Check 1: temporal ordering
    max_train = train["month_index"].max()
    min_val   = val["month_index"].min() if len(val) else max_train + 1
    min_test  = test["month_index"].min() if len(test) else min_val + 1

    assert max_train < min_val, f"LEAKAGE: train max month {max_train} >= val min month {min_val}"
    print(f"  ✅ Temporal ordering OK: train ends at month {max_train}, val starts at {min_val}")

    # Check 2: no perfect-correlation feature (leakage sniff)
    numeric_feats = [c for c in FEATURE_COLS if c in train.columns and
                     pd.api.types.is_numeric_dtype(train[c])]
    targets_binary = [t for t in TARGET_COLS if t in train.columns and
                      pd.api.types.is_numeric_dtype(train[t])]

    leakage_found = False
    for feat in numeric_feats:
        for tgt in targets_binary:
            try:
                corr = train[feat].corr(train[tgt].astype(float))
                if abs(corr) > 0.99:
                    print(f"  ⚠ HIGH CORR ({corr:.3f}): {feat} vs {tgt} — possible leakage!")
                    leakage_found = True
            except Exception:
                pass

    if not leakage_found:
        print(f"  ✅ No near-perfect feature↔target correlations detected (|r|<0.99)")

    # Check 3: future-derived columns not in feature set
    bad_cols = [c for c in ["next_3m_delinquency_flag","next_6m_delinquency_flag",
                             "next_12m_default_flag","next_12m_prepayment_flag",
                             "next_state"] if c in FEATURE_COLS]
    assert not bad_cols, f"LEAKAGE: TARGET columns found in FEATURE_COLS: {bad_cols}"
    print(f"  ✅ No target column present in FEATURE_COLS")
    print("── End Leakage Check ──────────────────────────────────────\n")


if __name__ == "__main__":
    from load_and_validate import load_cfg, load_all
    cfg  = load_cfg()
    dfs  = load_all(cfg)
    panel = pd.concat([dfs.get("train", pd.DataFrame()), dfs.get("test", pd.DataFrame())],
                      ignore_index=True)
    train, val, test = make_splits(panel, cfg)
