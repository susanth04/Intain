"""
anomaly.py — Stage 5: Anomaly Detection
Uses the EXISTING trained Isolation Forest + StandardScaler artifacts.
Combines with deterministic rule-violation scoring (same weights as training).
"""

import time
import numpy as np
import pandas as pd
from models.loader import MODELS, _DATA, PROJECT_ROOT

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly.exception_detection import rule_violation_score, top_drivers, ANOMALY_FEAT_COLS


def _safe(v):
    if hasattr(v, "item"): return v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    return v


def run(loan_id: str) -> dict:
    t0 = time.perf_counter()

    ifo    = MODELS.get("isolation_forest")
    scaler = MODELS.get("anomaly_scaler")
    panel: pd.DataFrame = _DATA.get("panel", pd.DataFrame())

    if panel.empty:
        return {"status": "error", "error": "Panel data not available"}

    loan_records = panel[panel["loan_id"] == loan_id].copy()
    if loan_records.empty:
        return {"status": "error", "error": f"Loan {loan_id} not found in panel"}

    # Add balance_utilisation (same as in exception_detection.py run())
    loan_records["balance_utilisation"] = (
        loan_records["current_balance"] / loan_records["original_balance"].replace(0, np.nan)
    ).clip(0, 5).fillna(0)

    # ── Rule violation score ──────────────────────────────────────────────────
    from pathlib import Path
    rules_path = PROJECT_ROOT / "data" / "raw" / "validation_rules.json"
    if rules_path.exists():
        rule_scores = rule_violation_score(loan_records, rules_path)
    else:
        rule_scores = pd.Series(0.0, index=loan_records.index)

    # ── ML anomaly score (Isolation Forest) ──────────────────────────────────
    feat_cols = [c for c in ANOMALY_FEAT_COLS if c in loan_records.columns]
    X_loan = loan_records[feat_cols].fillna(loan_records[feat_cols].median())

    if ifo is not None and scaler is not None:
        try:
            Xs_loan = scaler.transform(X_loan)
            raw_scores = -ifo.decision_function(Xs_loan)

            # Normalize using full panel stats for consistency
            panel_feat_cols = [c for c in ANOMALY_FEAT_COLS if c in panel.columns]
            panel_with_util = panel.copy()
            panel_with_util["balance_utilisation"] = (
                panel_with_util["current_balance"] / panel_with_util["original_balance"].replace(0, np.nan)
            ).clip(0, 5).fillna(0)
            X_panel = panel_with_util[[c for c in ANOMALY_FEAT_COLS if c in panel_with_util.columns]].fillna(0)
            Xs_panel = scaler.transform(X_panel)
            panel_raw = -ifo.decision_function(Xs_panel)
            panel_min, panel_max = panel_raw.min(), panel_raw.max()
            norm_scores = (raw_scores - panel_min) / (panel_max - panel_min + 1e-10)
            ml_scores = np.clip(norm_scores, 0, 1)
        except Exception as e:
            ml_scores = np.zeros(len(loan_records))
    else:
        ml_scores = np.zeros(len(loan_records))

    # ── Combine (same weights as training: 0.4 rule, 0.6 ML) ─────────────────
    rule_w = 0.4
    ml_w   = 0.6
    combined = (rule_w * rule_scores.values + ml_w * ml_scores).clip(0, 1)
    loan_records["anomaly_score"] = combined
    loan_records["rule_score"]    = rule_scores.values
    loan_records["ml_score"]      = ml_scores

    # ── Exception type ────────────────────────────────────────────────────────
    # Threshold from full panel (use pre-computed if available)
    anomaly_df: pd.DataFrame = _DATA.get("anomaly_scores", pd.DataFrame())
    if not anomaly_df.empty and "anomaly_score" in anomaly_df.columns:
        threshold = float(anomaly_df["anomaly_score"].quantile(0.95))
    else:
        threshold = 0.38  # approximate from training run

    latest_score   = float(combined[-1]) if len(combined) > 0 else 0.0
    latest_rule_sc = float(rule_scores.values[-1]) if len(rule_scores) else 0.0
    is_anomaly     = latest_score >= threshold

    if is_anomaly:
        # Check what kind
        latest_row = loan_records.iloc[-1]
        if latest_row.get("rule_score", 0) > 0.3:
            exc_type = "RuleViolation"
        else:
            exc_type = "Anomaly:IsolationForest"
        # Refine by specific rule violations
        if "default_flag" in latest_row and latest_row["default_flag"] == 1:
            if "document_status" in latest_row and latest_row["document_status"] == "Incomplete":
                exc_type = "DocumentGap"
        if "current_balance" in latest_row and "original_balance" in latest_row:
            if latest_row["current_balance"] > latest_row["original_balance"] * 1.5:
                exc_type = "BalanceAnomaly"
    else:
        exc_type = ""

    # ── Top drivers ───────────────────────────────────────────────────────────
    panel_all = _DATA.get("panel", pd.DataFrame())
    if not panel_all.empty:
        panel_all2 = panel_all.copy()
        panel_all2["balance_utilisation"] = (
            panel_all2["current_balance"] / panel_all2["original_balance"].replace(0, np.nan)
        ).clip(0, 5).fillna(0)
        global_mean = panel_all2[[c for c in feat_cols if c in panel_all2.columns]].mean()
        global_std  = panel_all2[[c for c in feat_cols if c in panel_all2.columns]].std()

        latest_for_drivers = loan_records.iloc[-1]
        top_drv_str = top_drivers(latest_for_drivers, feat_cols, global_mean, global_std)
        top_drv_list = []
        for part in top_drv_str.split("; "):
            part = part.strip()
            if not part: continue
            # Parse "feature(z=3.45)" format
            if "(z=" in part:
                fname = part.split("(z=")[0]
                z_val = float(part.split("(z=")[1].rstrip(")"))
                fval  = _safe(latest_for_drivers.get(fname))
                top_drv_list.append({
                    "feature": fname,
                    "value": fval,
                    "z_score": round(z_val, 2),
                })
            else:
                top_drv_list.append({"feature": part, "value": None, "z_score": None})
    else:
        top_drv_list = []

    # Build plain-language reason
    reason_parts = []
    latest_row2 = loan_records.iloc[-1]
    if latest_rule_sc > 0.2:
        reason_parts.append("rule violation(s) detected")
    if float(ml_scores[-1]) > 0.7:
        reason_parts.append("statistical outlier (Isolation Forest)")
    if "current_balance" in latest_row2 and "original_balance" in latest_row2:
        if _safe(latest_row2["current_balance"]) and _safe(latest_row2["original_balance"]):
            if latest_row2["current_balance"] > latest_row2["original_balance"] * 1.5:
                reason_parts.append("balance far exceeds original")
    if "default_flag" in latest_row2 and latest_row2.get("default_flag") == 1:
        if "document_status" in latest_row2 and latest_row2.get("document_status") == "Incomplete":
            reason_parts.append("defaulted with incomplete documentation")
    if not reason_parts:
        if is_anomaly:
            reason_parts.append(f"anomaly score ({latest_score:.3f}) exceeds threshold ({threshold:.3f})")
        else:
            reason_parts.append("within normal range")

    # Pre-computed reference (for transparency — not used as the score)
    precomputed_ref = {}
    anomaly_df2: pd.DataFrame = _DATA.get("anomaly_scores", pd.DataFrame())
    if not anomaly_df2.empty and "loan_id" in anomaly_df2.columns:
        ref_rows = anomaly_df2[anomaly_df2["loan_id"] == loan_id]
        if not ref_rows.empty:
            ref_latest = ref_rows.sort_values("month_index").iloc[-1]
            precomputed_ref = {
                "precomputed_anomaly_score": _safe(ref_latest.get("anomaly_score")),
                "precomputed_exception_flag": _safe(ref_latest.get("exception_flag")),
                "note": "Pre-computed from full pipeline run — shown for reference only. Real-time score computed above.",
            }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "loan_id": loan_id,
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(latest_score, 4),
        "rule_score": round(latest_rule_sc, 4),
        "ml_score": round(float(ml_scores[-1]), 4),
        "threshold": round(threshold, 4),
        "exception_type": exc_type,
        "reason": "; ".join(reason_parts),
        "top_drivers": top_drv_list,
        "model_used": "Isolation Forest (isolation_forest.pkl) + StandardScaler (anomaly_scaler.pkl)",
        "weights": {"rule_weight": rule_w, "ml_weight": ml_w},
        "precomputed_reference": precomputed_ref,
        "execution_ms": elapsed_ms,
    }
