"""
explainability.py — Stage 6: SHAP Feature Attribution
Uses the existing SHAP TreeExplainer implementation on the RAW LightGBM models.
Computes LOCAL SHAP values for the specific loan record.
"""

import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from models.loader import MODELS
from pipeline.features import get_feature_vector


BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]

TARGET_LABELS = {
    "next_3m_delinquency_flag":  "3-Month Delinquency",
    "next_6m_delinquency_flag":  "6-Month Delinquency",
    "next_12m_default_flag":     "12-Month Default",
    "next_12m_prepayment_flag":  "12-Month Prepayment",
}


def _safe_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except Exception:
        return None


def run(loan_id: str) -> dict:
    t0 = time.perf_counter()

    feat_df, feature_names = get_feature_vector(loan_id)
    if feat_df is None or feat_df.empty:
        return {"status": "error", "error": f"Feature vector not available for {loan_id}"}

    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False

    if not shap_available:
        return {
            "available": False,
            "reason": "shap library not installed. Run: pip install shap",
        }

    explanations = {}

    for tgt in BINARY_TARGETS:
        raw_model_key = f"{tgt}_raw"
        raw_model = MODELS.get(raw_model_key)

        if raw_model is None:
            explanations[tgt] = {
                "label": TARGET_LABELS[tgt],
                "available": False,
                "reason": f"Raw model artifact '{tgt}_lgbm_raw.pkl' not found.",
            }
            continue

        try:
            t_shap = time.perf_counter()

            # Align features to model's expected input
            try:
                if hasattr(raw_model, "feature_names_in_"):
                    model_feats = list(raw_model.feature_names_in_)
                    X_aligned = feat_df.reindex(columns=model_feats, fill_value=0)
                elif hasattr(raw_model, "feature_name_"):
                    model_feats = raw_model.feature_name_()
                    X_aligned = feat_df.reindex(columns=model_feats, fill_value=0)
                else:
                    X_aligned = feat_df
                    model_feats = list(feat_df.columns)
            except Exception:
                X_aligned = feat_df
                model_feats = list(feat_df.columns)

            explainer   = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(X_aligned)

            # Handle list output (binary classification may return [neg_class, pos_class])
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    sv = shap_values[1]  # positive class
                else:
                    sv = shap_values[0]
            else:
                sv = shap_values

            # sv shape: (1, n_features) for single sample
            if sv.ndim == 2:
                sv_loan = sv[0]
            else:
                sv_loan = sv

            shap_ms = round((time.perf_counter() - t_shap) * 1000, 1)

            # Build sorted feature attribution list
            feat_shap = sorted(
                zip(model_feats, sv_loan, X_aligned.values[0]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            top_features = []
            for fname, shap_val, feat_val in feat_shap[:15]:
                sv_safe  = _safe_float(shap_val)
                fv_safe  = _safe_float(feat_val)
                if sv_safe is None:
                    continue
                top_features.append({
                    "feature":   fname,
                    "value":     fv_safe,
                    "shap_value": sv_safe,
                    "direction": "positive" if sv_safe > 0 else "negative",
                    "impact":    "increases_risk" if sv_safe > 0 else "decreases_risk",
                })

            # Expected value (base rate)
            try:
                expected_value = _safe_float(explainer.expected_value)
                if isinstance(expected_value, list):
                    expected_value = _safe_float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
            except Exception:
                expected_value = None

            explanations[tgt] = {
                "label": TARGET_LABELS[tgt],
                "available": True,
                "model_used": "shap.TreeExplainer on LightGBM (raw, uncalibrated)",
                "model_file": f"{tgt}_lgbm_raw.pkl",
                "expected_value": expected_value,
                "top_features": top_features,
                "total_features_explained": len(model_feats),
                "shap_computation_ms": shap_ms,
            }

        except Exception as e:
            explanations[tgt] = {
                "label": TARGET_LABELS[tgt],
                "available": False,
                "reason": str(e),
            }

    # Next-state SHAP — known to fail due to (35, 7) multidimensional output
    explanations["next_state"] = {
        "available": False,
        "reason": (
            "SHAP for next_state multiclass predictor fails at runtime due to a "
            "multidimensional output array (n_features, n_classes). "
            "This is a known limitation documented in model_card.md."
        ),
    }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "loan_id": loan_id,
        "explanations": explanations,
        "execution_ms": elapsed_ms,
    }
