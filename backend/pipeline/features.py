"""
features.py — Stage 3: Feature Engineering
Runs the EXACT same feature pipeline used during model training.
Reuses src/features/build_features.py functions directly.
"""

import time
import numpy as np
import pandas as pd
from models.loader import _DATA, PROJECT_ROOT, MODELS

# Import the real feature engineering functions
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    add_lag_features,
    add_engineered_features,
    ORDINAL_MAPS,
    CAT_COLS,
    TARGET_COLS,
    get_feature_names,
)


def _safe(v):
    if hasattr(v, "item"): return v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    return v


def run(loan_id: str) -> dict:
    """
    Build features for the given loan using the full panel context
    (required for rolling/lag features to be accurate).
    Returns the feature vector ready for model inference.
    """
    t0 = time.perf_counter()

    panel: pd.DataFrame = _DATA.get("panel", pd.DataFrame())
    static: pd.DataFrame = _DATA.get("static", pd.DataFrame())

    if panel.empty:
        return {"status": "error", "error": "Panel data not available"}

    # Merge static attributes into panel if needed
    if not static.empty and "loan_id" in static.columns:
        static_cols = [c for c in static.columns
                       if c not in panel.columns or c == "loan_id"]
        panel_merged = panel.merge(static[static_cols], on="loan_id", how="left")
    else:
        panel_merged = panel.copy()

    raw_cols = panel_merged.columns.tolist()
    n_raw = len(raw_cols)

    # ── Add lag features (must process full panel for correct rolling windows) ──
    panel_lagged = add_lag_features(panel_merged)

    # ── Add engineered features ───────────────────────────────────────────────
    panel_eng = add_engineered_features(panel_lagged)

    # ── Encode categoricals using the SAME ordinal maps used during training ──
    for col, order in ORDINAL_MAPS.items():
        if col in panel_eng.columns:
            enc_col = f"{col}_enc"
            mapping = {v: i for i, v in enumerate(order)}
            panel_eng[enc_col] = panel_eng[col].map(mapping).fillna(-1).astype(int)

    # Frequency-encode remaining cats (fit on full panel, apply to loan)
    freq_encoders = {}
    for col in CAT_COLS:
        if col not in panel_eng.columns or col in ORDINAL_MAPS:
            continue
        enc_col = f"{col}_enc"
        freq = panel_eng[col].value_counts(normalize=True)
        freq_encoders[col] = freq
        panel_eng[enc_col] = panel_eng[col].map(freq).fillna(0)

    # ── Get feature names (same selection logic as training) ─────────────────
    feature_names = get_feature_names(panel_eng)
    n_engineered = len(panel_eng.columns)
    n_features = len(feature_names)

    # ── Extract this loan's latest record ────────────────────────────────────
    loan_panel = panel_eng[panel_eng["loan_id"] == loan_id].sort_values("month_index")
    if loan_panel.empty:
        return {"status": "error", "error": f"Loan {loan_id} not found after feature engineering"}

    latest = loan_panel.iloc[-1]

    # ── Apply imputer (fitted on train set) ─────────────────────────────────
    imputer = MODELS.get("imputer")
    loan_feat_row = latest[feature_names].copy()

    if imputer is not None:
        # The imputer was fitted on all features; apply it
        try:
            import pandas as _pd
        except:
            pass
        loan_arr = loan_feat_row.values.reshape(1, -1)
        loan_arr_imputed = imputer.transform(loan_arr)
        feature_vector = pd.Series(loan_arr_imputed[0], index=feature_names)
    else:
        feature_vector = loan_feat_row.fillna(loan_feat_row.median())

    # ── Build output ──────────────────────────────────────────────────────────
    feature_values = {}
    for fname in feature_names:
        v = feature_vector.get(fname)
        feature_values[fname] = _safe(v)

    # Key transformations applied
    transformations = [
        "status_lag1, status_lag3, status_roll3_max (DPD status rolling window)",
        "current_balance_lag1, current_balance_lag3, current_balance_roll3_mean",
        "days_past_due_lag1, days_past_due_lag3, days_past_due_roll6_max",
        "balance_utilisation = current_balance / original_balance (clipped 0-3)",
        "rate_spread = interest_rate - portfolio mean",
        "seasoning_bucket (loan age bins: 0-12m, 13-24m, 25-36m, 37-60m, 61-120m, 120m+)",
        "high_risk_combo = Fair/Poor credit × high LTV (interaction feature)",
        "Ordinal encoding: credit_score_band, ltv_band, dti_band, current_status, loss_severity_band",
        "Frequency encoding: state, loan_purpose, occupancy_type, property_type, servicer_name",
        "Median imputation (imputer.pkl fitted on train set)",
    ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "raw_feature_count": n_raw,
        "engineered_column_count": n_engineered,
        "final_model_feature_count": n_features,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "transformations_applied": transformations,
        "loan_month_index": int(latest.get("month_index", -1)),
        "imputer_used": imputer is not None,
        "execution_ms": elapsed_ms,
    }


def get_feature_vector(loan_id: str) -> tuple:
    """
    Returns (feature_vector_as_DataFrame_row, feature_names) for use by
    downstream stages (prediction, anomaly, scenario, SHAP).
    This avoids re-running feature engineering multiple times.
    """
    panel: pd.DataFrame = _DATA.get("panel", pd.DataFrame())
    static: pd.DataFrame = _DATA.get("static", pd.DataFrame())

    if panel.empty:
        return None, []

    if not static.empty and "loan_id" in static.columns:
        static_cols = [c for c in static.columns
                       if c not in panel.columns or c == "loan_id"]
        panel_merged = panel.merge(static[static_cols], on="loan_id", how="left")
    else:
        panel_merged = panel.copy()

    panel_lagged = add_lag_features(panel_merged)
    panel_eng = add_engineered_features(panel_lagged)

    for col, order in ORDINAL_MAPS.items():
        if col in panel_eng.columns:
            enc_col = f"{col}_enc"
            mapping = {v: i for i, v in enumerate(order)}
            panel_eng[enc_col] = panel_eng[col].map(mapping).fillna(-1).astype(int)

    for col in CAT_COLS:
        if col not in panel_eng.columns or col in ORDINAL_MAPS:
            continue
        enc_col = f"{col}_enc"
        freq = panel_eng[col].value_counts(normalize=True)
        panel_eng[enc_col] = panel_eng[col].map(freq).fillna(0)

    feature_names = get_feature_names(panel_eng)

    loan_panel = panel_eng[panel_eng["loan_id"] == loan_id].sort_values("month_index")
    if loan_panel.empty:
        return None, feature_names

    latest = loan_panel.iloc[-1]
    loan_feat_row = latest[feature_names].copy()

    imputer = MODELS.get("imputer")
    if imputer is not None:
        try:
            loan_arr = loan_feat_row.values.reshape(1, -1)
            loan_arr_imputed = imputer.transform(loan_arr)
            feature_vector = pd.DataFrame(loan_arr_imputed, columns=feature_names)
        except Exception:
            feature_vector = pd.DataFrame(
                [loan_feat_row.fillna(0).values], columns=feature_names
            )
    else:
        feature_vector = pd.DataFrame(
            [loan_feat_row.fillna(0).values], columns=feature_names
        )

    return feature_vector, feature_names
