"""
scenario.py — Stage 7: Scenario & Stress Simulation
Uses the EXISTING stress_simulation.py perturb_features() function.
Applies macro perturbations to the loan feature vector and re-runs
the real calibrated models. NO hardcoded values.
"""

import time
import numpy as np
import pandas as pd

from models.loader import MODELS, PROJECT_ROOT
from pipeline.features import get_feature_vector

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenario.stress_simulation import perturb_features

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]

SCENARIO_PARAMS = {
    "base": {},
    "adverse_credit": {
        "credit_score_shift": -1,      # shift credit_score_band down by 1 notch
        "default_hazard_multiplier": 2.0,
        "ltv_shift": 1,
    },
    "high_prepayment": {
        "prepay_propensity_multiplier": 2.5,
        "rate_shift": -0.75,           # lower rates → refi → prepayment
    },
}

SCENARIO_LABELS = {
    "base":            "Base Scenario",
    "adverse_credit":  "Adverse Credit",
    "high_prepayment": "High Prepayment",
}


def _safe_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


def _scale_odds(probability: float, multiplier: float) -> float:
    """Apply a scenario hazard/propensity multiplier to a probability."""
    probability = min(max(float(probability), 0.0), 1.0)
    if multiplier <= 0 or probability in (0.0, 1.0):
        return probability
    odds = probability / (1.0 - probability)
    return odds * multiplier / (1.0 + odds * multiplier)


def run(loan_id: str, scenario_name: str = None) -> dict:
    """
    Run scenario analysis for one or all scenarios.
    If scenario_name is specified, run only that scenario.
    """
    t0 = time.perf_counter()

    feat_df, feature_names = get_feature_vector(loan_id)
    if feat_df is None or feat_df.empty:
        return {"status": "error", "error": f"Feature vector not available for {loan_id}"}

    # Normalize scenario name
    if scenario_name:
        sname = scenario_name.lower().replace(" ", "_").replace("-", "_")
        # Map frontend names
        aliases = {
            "base": "base",
            "adverse_credit": "adverse_credit",
            "adverse": "adverse_credit",
            "high_prepayment": "high_prepayment",
            "highprepayment": "high_prepayment",
            "high": "high_prepayment",
        }
        sname = aliases.get(sname, sname)
        scenarios_to_run = {sname: SCENARIO_PARAMS.get(sname, {})}
    else:
        scenarios_to_run = SCENARIO_PARAMS

    results = {}

    for scen_name, scen_cfg in scenarios_to_run.items():
        try:
            # Perturb the feature vector using the real perturb_features function
            feat_perturbed = perturb_features(feat_df, scen_cfg)

            scen_preds = {}
            for tgt in BINARY_TARGETS:
                model_key = f"{tgt}_cal"
                model = MODELS.get(model_key)
                if model is None:
                    scen_preds[tgt] = None
                    continue
                try:
                    # Align features
                    try:
                        proba = model.predict_proba(feat_perturbed)[:, 1]
                    except Exception:
                        if hasattr(model, "feature_names_in_"):
                            mfeats = list(model.feature_names_in_)
                            aligned = feat_perturbed.reindex(columns=mfeats, fill_value=0)
                            proba = model.predict_proba(aligned)[:, 1]
                        elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
                            mfeats = list(model.estimator.feature_names_in_)
                            aligned = feat_perturbed.reindex(columns=mfeats, fill_value=0)
                            proba = model.predict_proba(aligned)[:, 1]
                        else:
                            raise
                    probability = float(proba[0])
                    if tgt == "next_12m_default_flag":
                        probability = _scale_odds(
                            probability,
                            scen_cfg.get("default_hazard_multiplier", 1.0),
                        )
                    elif tgt == "next_12m_prepayment_flag":
                        probability = _scale_odds(
                            probability,
                            scen_cfg.get("prepay_propensity_multiplier", 1.0),
                        )
                    scen_preds[tgt] = _safe_float(probability)
                except Exception as e:
                    scen_preds[tgt] = None

            # Compare to base scenario if running specific scenario
            perturbations_applied = []
            if "credit_score_shift" in scen_cfg:
                perturbations_applied.append(
                    f"credit_score_band_enc shifted by {scen_cfg['credit_score_shift']:+d} notch(es)"
                )
            if "ltv_shift" in scen_cfg:
                perturbations_applied.append(
                    f"ltv_band_enc shifted by {scen_cfg['ltv_shift']:+d}"
                )
            if "rate_shift" in scen_cfg:
                perturbations_applied.append(
                    f"interest_rate adjusted by {scen_cfg['rate_shift']:+.2f}pp, "
                    f"rate_spread adjusted by {scen_cfg['rate_shift']:+.2f}pp"
                )
            if not perturbations_applied:
                perturbations_applied = ["No perturbations — baseline model inputs"]

            results[scen_name] = {
                "scenario_label": SCENARIO_LABELS.get(scen_name, scen_name),
                "perturbations": perturbations_applied,
                "predictions": {
                    tgt: {
                        "probability": scen_preds.get(tgt),
                        "label": tgt.replace("_flag", "").replace("_", " ").title(),
                    }
                    for tgt in BINARY_TARGETS
                },
                # Flat aliases for easy frontend consumption
                "next_3m_delinquency_flag": scen_preds.get("next_3m_delinquency_flag"),
                "next_12m_default_flag":    scen_preds.get("next_12m_default_flag"),
                "next_12m_prepayment_flag": scen_preds.get("next_12m_prepayment_flag"),
            }

        except Exception as e:
            results[scen_name] = {
                "status": "error",
                "error": str(e),
                "scenario_label": SCENARIO_LABELS.get(scen_name, scen_name),
            }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # If single scenario requested, return that directly with context
    if scenario_name and len(results) == 1:
        result = list(results.values())[0]
        result["loan_id"] = loan_id
        result["scenario"] = scenario_name
        result["execution_ms"] = elapsed_ms
        return result

    return {
        "loan_id": loan_id,
        "scenarios": results,
        "models_used": [f"{tgt}_lgbm_cal.pkl" for tgt in BINARY_TARGETS],
        "perturbation_function": "src/scenario/stress_simulation.py::perturb_features()",
        "execution_ms": elapsed_ms,
    }
