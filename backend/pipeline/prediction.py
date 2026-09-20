"""
prediction.py — Stage 4: Risk Prediction
Loads existing calibrated model artifacts. Runs REAL inference.
NO training. NO retraining. Models loaded once at startup via loader.py.
"""

import time
import numpy as np
import pandas as pd
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
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None



LR_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def _run_lr_predictions(feat_df):
    lr_results = {}
    for tgt in LR_TARGETS:
        model = MODELS.get(f"{tgt}_lr")
        if model is None:
            continue
        try:
            try:
                proba = model.predict_proba(feat_df)[:, 1]
            except Exception:
                if hasattr(model, "feature_names_in_"):
                    aligned = feat_df.reindex(columns=list(model.feature_names_in_), fill_value=0)
                    proba = model.predict_proba(aligned)[:, 1]
                else:
                    continue
            lr_results[tgt] = _safe_float(proba[0])
        except Exception:
            pass
    return lr_results

def run(loan_id: str) -> dict:
    t0 = time.perf_counter()

    # Get the feature vector for this loan (reuses real feature pipeline)
    feat_df, feature_names = get_feature_vector(loan_id)

    if feat_df is None or feat_df.empty:
        return {"status": "error", "error": f"Feature vector could not be built for {loan_id}"}

    predictions = {}

    # ── Binary classifiers ─────────────────────────────────────────────────────
    for tgt in BINARY_TARGETS:
        model_key = f"{tgt}_cal"
        model = MODELS.get(model_key)

        if model is None:
            predictions[tgt] = {
                "label": TARGET_LABELS[tgt],
                "status": "model_not_loaded",
                "probability": None,
            }
            continue

        try:
            t_infer = time.perf_counter()

            # Align features — model may have been trained on a slightly different feature set
            try:
                proba = model.predict_proba(feat_df)[:, 1]
            except Exception as align_err:
                # Try getting feature names from model and aligning
                if hasattr(model, "feature_names_in_"):
                    model_feats = list(model.feature_names_in_)
                    aligned = feat_df.reindex(columns=model_feats, fill_value=0)
                    proba = model.predict_proba(aligned)[:, 1]
                elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
                    model_feats = list(model.estimator.feature_names_in_)
                    aligned = feat_df.reindex(columns=model_feats, fill_value=0)
                    proba = model.predict_proba(aligned)[:, 1]
                else:
                    raise align_err

            infer_ms = round((time.perf_counter() - t_infer) * 1000, 1)
            prob_val  = _safe_float(proba[0])

            # Risk tier
            if prob_val is None:
                tier = "unknown"
            elif prob_val >= 0.6:
                tier = "high"
            elif prob_val >= 0.35:
                tier = "elevated"
            elif prob_val >= 0.15:
                tier = "moderate"
            else:
                tier = "low"

            predictions[tgt] = {
                "label": TARGET_LABELS[tgt],
                "probability": prob_val,
                "risk_tier": tier,
                "model": "LightGBM (calibrated, CalibratedClassifierCV sigmoid)",
                "model_file": f"{tgt}_lgbm_cal.pkl",
                "inference_ms": infer_ms,
            }

        except Exception as e:
            predictions[tgt] = {
                "label": TARGET_LABELS[tgt],
                "status": "inference_error",
                "error": str(e),
                "probability": None,
            }

    # ── Next-state multiclass ────────────────────────────────────────────────
    next_state_result = _predict_next_state(feat_df)

    # ── Recommended action ───────────────────────────────────────────────────
    default_prob  = _safe_float(predictions.get("next_12m_default_flag", {}).get("probability"))
    prepay_prob   = _safe_float(predictions.get("next_12m_prepayment_flag", {}).get("probability"))
    recommended_action = _recommend_action(default_prob, prepay_prob)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Run LR baseline for confidence comparison
    lr_predictions = _run_lr_predictions(feat_df)

    # Build ensemble confidence per target
    ensemble = {}
    for tgt in BINARY_TARGETS:
        lgbm_p = predictions.get(tgt, {}).get("probability")
        lr_p   = lr_predictions.get(tgt)
        if lgbm_p is not None and lr_p is not None:
            diff = abs(lgbm_p - lr_p)
            ensemble[tgt] = {
                "lgbm_probability": lgbm_p,
                "lr_probability":   lr_p,
                "agreement_gap":    round(diff, 4),
                "confidence":       "high" if diff < 0.10 else ("medium" if diff < 0.25 else "low"),
            }

    return {
        "loan_id": loan_id,
        "predictions": predictions,
        "ensemble_confidence": ensemble,
        "next_state": next_state_result,
        "recommended_action": recommended_action,
        "features_used": len(feature_names),
        "execution_ms": elapsed_ms,
    }


def _predict_next_state(feat_df: pd.DataFrame) -> dict:
    """Run next-state multiclass prediction if model is available."""
    model = MODELS.get("next_state_lgbm")
    le    = MODELS.get("next_state_le")

    if model is None:
        return {"status": "model_not_loaded"}

    try:
        # Align features
        try:
            proba = model.predict_proba(feat_df)
            pred_class = model.predict(feat_df)
        except Exception:
            if hasattr(model, "feature_names_in_"):
                feats = list(model.feature_names_in_)
                aligned = feat_df.reindex(columns=feats, fill_value=0)
                proba = model.predict_proba(aligned)
                pred_class = model.predict(aligned)
            else:
                raise

        if le is not None:
            try:
                predicted_label = le.inverse_transform(pred_class.astype(int))[0]
                class_labels = list(le.classes_)
            except Exception:
                predicted_label = str(pred_class[0])
                class_labels = list(range(proba.shape[1]))
        else:
            predicted_label = str(pred_class[0])
            class_labels = list(range(proba.shape[1]))

        class_probs = {
            str(label): round(float(p), 4)
            for label, p in zip(class_labels, proba[0])
        }

        return {
            "predicted_state": predicted_label,
            "class_probabilities": class_probs,
            "model": "LightGBM multiclass",
            "model_file": "next_state_lgbm.pkl",
        }

    except Exception as e:
        return {"status": "inference_error", "error": str(e)}


def _recommend_action(default_prob, prepay_prob) -> dict:
    """Derive recommended servicing action from predicted probabilities."""
    ANOMALY_THRESHOLD = 0.3  # will be overridden by actual anomaly score in stage 5

    if default_prob is None and prepay_prob is None:
        return {"action": "insufficient_data", "label": "Insufficient data", "color": "gray"}

    dp = default_prob or 0.0
    pp = prepay_prob or 0.0

    if dp >= 0.5:
        return {
            "action": "monitor_default",
            "label": "Monitor closely — elevated default risk",
            "color": "amber",
            "default_prob": dp,
            "prepay_prob": pp,
        }
    elif pp >= 0.55:
        return {
            "action": "watch_prepayment",
            "label": "Watch for prepayment — portfolio income risk",
            "color": "blue",
            "default_prob": dp,
            "prepay_prob": pp,
        }
    else:
        return {
            "action": "no_immediate_action",
            "label": "No immediate action",
            "color": "green",
            "default_prob": dp,
            "prepay_prob": pp,
        }

