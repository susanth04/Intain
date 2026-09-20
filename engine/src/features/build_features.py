"""
build_features.py
=================
Task 2 — Feature Engineering

Builds the feature matrix from the panel DataFrame:
  - Lag / rolling delinquency history
  - Loan-age buckets (seasoning)
  - Rate-spread features
  - Static × dynamic interactions
  - Ordinal / one-hot encoding of categoricals
  - Preserves TARGET_COLS untouched

Returns:
  X_train, X_val, X_test, y_dict, feature_names
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


TARGET_COLS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]

CAT_COLS = [
    "credit_score_band", "ltv_band", "dti_band", "state",
    "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "current_status", "document_status",
    "loss_severity_band", "source_system",
]

NUMERIC_RAW = [
    "loan_age_months", "remaining_term_months", "current_balance",
    "interest_rate", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "month_index",
    "original_balance",
]

ORDINAL_MAPS = {
    "credit_score_band": ["Excellent", "Good", "Fair", "Poor"],
    "ltv_band":          ["<=60", "60-75", "75-90", ">90"],
    "dti_band":          ["<=28", "28-36", "36-43", ">43"],
    "current_status":    ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"],
    "loss_severity_band":["Low", "Medium", "High"],
}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-period and 3-period lags + rolling stats for key columns."""
    df = df.sort_values(["loan_id", "month_index"]).copy()
    grp = df.groupby("loan_id")

    for col in ["days_past_due", "current_balance", "current_status"]:
        if col not in df.columns:
            continue
        if col == "current_status":
            # Encode status to numeric for rolling
            status_map = {"Current":0,"30DPD":1,"60DPD":2,"90DPD":3,"Default":4,"Prepaid":5,"Closed":6}
            s = df[col].map(status_map).fillna(0)
            df["status_numeric"] = s
            df["status_lag1"] = grp["status_numeric"].shift(1)
            df["status_lag3"] = grp["status_numeric"].shift(3)
            df["status_roll3_max"] = grp["status_numeric"].transform(
                lambda x: x.rolling(3, min_periods=1).max())
            df.drop("status_numeric", axis=1, inplace=True)
        else:
            df[f"{col}_lag1"] = grp[col].shift(1)
            df[f"{col}_lag3"] = grp[col].shift(3)
            df[f"{col}_roll3_mean"] = grp[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            if col == "days_past_due":
                df[f"{col}_roll6_max"] = grp[col].transform(
                    lambda x: x.rolling(6, min_periods=1).max())
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rate-spread, seasoning buckets, static×dynamic interactions."""
    df = df.copy()

    # Seasoning buckets
    bins   = [0, 12, 24, 36, 60, 120, 999]
    labels = ["0-12m","13-24m","25-36m","37-60m","61-120m","120m+"]
    df["seasoning_bucket"] = pd.cut(df["loan_age_months"], bins=bins, labels=labels,
                                    right=True, include_lowest=True).astype(str)

    # Current balance as fraction of original
    df["balance_utilisation"] = (
        df["current_balance"] / df["original_balance"].replace(0, np.nan)
    ).clip(0, 3)

    # Rate spread proxy (interest_rate relative to vintage-level mean)
    if "vintage" in df.columns and "interest_rate" in df.columns:
        vintage_mean_rate = df.groupby("vintage")["interest_rate"].transform("mean")
        df["rate_spread"] = df["interest_rate"] - vintage_mean_rate
    elif "interest_rate" in df.columns:
        df["rate_spread"] = df["interest_rate"] - df["interest_rate"].mean()

    df["high_risk_combo"] = (
        df.get("credit_score_band", "").isin(["Fair", "Poor"]).astype(int) *
        df.get("ltv_band", "").isin(["75-90", ">90"]).astype(int)
    )

    return df


def encode_categoricals(df_train: pd.DataFrame,
                        df_val:   pd.DataFrame,
                        df_test:  pd.DataFrame,
                        proc_dir: Path) -> tuple:
    """
    Ordinal-encode known-order categoricals, frequency-encode the rest.
    Fits on train only; applies to val/test.
    """
    encoders = {}
    proc_dir.mkdir(parents=True, exist_ok=True)

    for df in [df_train, df_val, df_test]:
        for col, order in ORDINAL_MAPS.items():
            if col not in df.columns: continue
            enc_col = f"{col}_enc"
            mapping = {v: i for i, v in enumerate(order)}
            df[enc_col] = df[col].map(mapping).fillna(-1).astype(int)

        # Frequency-encode remaining cats
        for col in CAT_COLS:
            if col not in df.columns or col in ORDINAL_MAPS: continue
            enc_col = f"{col}_enc"
            if df is df_train:
                freq = df[col].value_counts(normalize=True)
                encoders[col] = freq
            else:
                freq = encoders.get(col, pd.Series(dtype=float))
            df[enc_col] = df[col].map(freq).fillna(0)

    return df_train, df_val, df_test, encoders


def get_feature_names(df: pd.DataFrame) -> list:
    """Return the final list of numeric feature columns (no targets, no raw cats)."""
    exclude = set(TARGET_COLS) | set(CAT_COLS) | {
        "loan_id", "reporting_month", "origination_month",
        "last_updated_at", "source_system", "vintage",
    }
    feats = [c for c in df.columns
             if c not in exclude
             and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def build_features(df_train: pd.DataFrame,
                   df_val:   pd.DataFrame,
                   df_test:  pd.DataFrame,
                   cfg:      dict) -> tuple:
    """
    End-to-end feature pipeline.

    Returns:
        X_train, X_val, X_test: DataFrames of features
        y_dict: {target_name: (y_train, y_val, y_test)}
        feature_names: list[str]
    """
    proc_dir = ROOT / cfg["PATHS"]["processed_data"]

    print("  [features] Adding lag features …")
    # Combine for consistent lag computation, then re-split
    full = pd.concat([df_train, df_val, df_test], ignore_index=True)
    full = add_lag_features(full)
    full = add_engineered_features(full)

    train_cut = cfg["SPLIT"]["TRAIN_CUTOFF"]
    val_cut   = cfg["SPLIT"]["VAL_CUTOFF"]

    df_train_ = full[full["month_index"] <= train_cut].copy()
    df_val_   = full[(full["month_index"] > train_cut) & (full["month_index"] <= val_cut)].copy()
    df_test_  = full[full["month_index"] > val_cut].copy()

    print("  [features] Encoding categoricals …")
    df_train_, df_val_, df_test_, _ = encode_categoricals(df_train_, df_val_, df_test_, proc_dir)

    feature_names = get_feature_names(df_train_)
    print(f"  [features] {len(feature_names)} features assembled")

    # Impute missing values (median for numeric)
    imp = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imp.fit_transform(df_train_[feature_names]),
                           columns=feature_names, index=df_train_.index)
    X_val   = pd.DataFrame(imp.transform(df_val_[feature_names]),
                           columns=feature_names, index=df_val_.index)
    X_test  = pd.DataFrame(imp.transform(df_test_[feature_names]),
                           columns=feature_names, index=df_test_.index)

    joblib.dump(imp, proc_dir / "imputer.pkl")

    # Build target dict
    y_dict = {}
    for tgt in TARGET_COLS:
        y_tr  = df_train_.get(tgt)
        y_v   = df_val_.get(tgt)
        y_te  = df_test_.get(tgt)
        if y_tr is not None:
            y_dict[tgt] = (y_tr, y_v, y_te)

    return X_train, X_val, X_test, y_dict, feature_names, df_train_, df_val_, df_test_


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.load_and_validate import load_cfg, load_all
    from src.data.splits import make_splits
    import pandas as pd

    cfg   = load_cfg()
    dfs   = load_all(cfg)
    panel = pd.concat([dfs["train"], dfs["test"]], ignore_index=True)
    tr, val, te = make_splits(panel, cfg)
    X_tr, X_val, X_te, y_dict, feats, *_ = build_features(tr, val, te, cfg)
    print(f"X_train: {X_tr.shape}, features: {feats[:5]}…")
