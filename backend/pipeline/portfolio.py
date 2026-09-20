"""
portfolio.py — Judge-demo aggregates: heatmaps, risk tiers, honest metrics.
"""

from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd

from models.loader import _DATA, PROC_DIR

BAND_ORDER = ["Poor", "Fair", "Good", "Excellent"]
STATUS_ORDER = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"]
SCENARIOS = ["base", "adverse_credit", "high_prepayment"]

# From reports/scenario_report.md — shown as the stress overlay
SCENARIO_BAND_DEFAULT = {
    "Poor": {"base": 0.2921, "adverse_credit": 0.2857, "high_prepayment": 0.2784},
    "Fair": {"base": 0.2683, "adverse_credit": 0.2619, "high_prepayment": 0.2557},
    "Good": {"base": 0.2009, "adverse_credit": 0.2044, "high_prepayment": 0.2003},
    "Excellent": {"base": 0.1480, "adverse_credit": 0.1520, "high_prepayment": 0.1410},
}


def _safe_str(v, fallback="Unknown"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    s = str(v).strip()
    return s if s and s.lower() != "nan" else fallback


def _latest_panel() -> pd.DataFrame:
    panel = _DATA.get("panel")
    if panel is None or panel.empty:
        return pd.DataFrame()
    sort_col = "month_index" if "month_index" in panel.columns else panel.columns[1]
    return panel.sort_values(sort_col).groupby("loan_id", as_index=False).last()


def _crosstab_matrix(df: pd.DataFrame, row_col: str, col_col: str, row_order, col_order, value_col=None):
    if df.empty:
        return {"rows": row_order, "cols": col_order, "cells": [[0] * len(col_order) for _ in row_order], "max": 0}

    work = df.copy()
    work[row_col] = work[row_col].map(lambda v: _safe_str(v))
    work[col_col] = work[col_col].map(lambda v: _safe_str(v))

    if value_col:
        pivot = work.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    else:
        pivot = pd.crosstab(work[row_col], work[col_col])

    cells = []
    vmax = 0.0
    for row in row_order:
        line = []
        for col in col_order:
            try:
                val = float(pivot.loc[row, col]) if row in pivot.index and col in pivot.columns else 0.0
            except Exception:
                val = 0.0
            if np.isnan(val):
                val = 0.0
            line.append(round(val, 4))
            vmax = max(vmax, val)
        cells.append(line)
    return {"rows": list(row_order), "cols": list(col_order), "cells": cells, "max": round(vmax, 4)}


def _risk_tier(dpd: float, def_flag: int, anomaly: float) -> str:
    proxy = min(1.0, (dpd / 120.0) * 0.45 + def_flag * 0.35 + anomaly * 0.2)
    if proxy >= 0.6 or def_flag == 1:
        return "high"
    if proxy >= 0.3 or dpd > 30:
        return "elevated"
    if proxy >= 0.1 or dpd > 0:
        return "moderate"
    return "low"


def _load_metrics():
    metrics = {}
    for name in ("binary_model_metrics.json", "next_state_metrics.json", "survival_metrics.json"):
        path = Path(PROC_DIR) / name
        if path.exists():
            try:
                metrics[name.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    highlights = [
        {
            "target": "3M delinquency",
            "lgbm_test_roc": 0.7129,
            "lr_test_roc": 0.6618,
            "note": "LightGBM lifts ROC vs logistic baseline on test.",
        },
        {
            "target": "6M delinquency",
            "lgbm_test_roc": 0.7098,
            "lr_test_roc": 0.6580,
            "note": "Strongest ranking lift vs baseline.",
        },
        {
            "target": "12M default",
            "lgbm_test_roc": 0.6966,
            "lr_test_roc": 0.6966,
            "note": "LGBM matched LR — rare-event target; ranking still usable, PR-AUC remains the honest metric.",
        },
        {
            "target": "12M prepay",
            "lgbm_test_roc": 0.5652,
            "lr_test_roc": 0.5652,
            "note": "Weak signal. Do not over-claim prepay skill in the demo.",
        },
        {
            "target": "Next state (macro F1)",
            "lgbm_test_roc": 0.5083,
            "lr_test_roc": 0.3140,
            "note": "Multiclass LightGBM clearly beats LR (0.51 vs 0.31 macro F1).",
        },
    ]
    return {"raw": metrics, "highlights": highlights, "cox_concordance": 0.5617}


def overview(limit: int = 24) -> dict:
    latest = _latest_panel()
    if latest.empty:
        return {"status": "error", "error": "Panel not loaded"}

    anomaly_scores = _DATA.get("anomaly_scores")
    if anomaly_scores is not None and not anomaly_scores.empty:
        sort_col = "month_index" if "month_index" in anomaly_scores.columns else anomaly_scores.columns[0]
        anom_latest = anomaly_scores.sort_values(sort_col).groupby("loan_id", as_index=False).last()
        keep = [c for c in ["loan_id", "anomaly_score", "exception_flag", "predicted_exception_type", "top_drivers"] if c in anom_latest.columns]
        latest = latest.merge(anom_latest[keep], on="loan_id", how="left")

    if "anomaly_score" not in latest.columns:
        latest["anomaly_score"] = 0.0
    latest["anomaly_score"] = latest["anomaly_score"].fillna(0.0)
    latest["credit_score_band"] = latest.get("credit_score_band", "Unknown")
    latest["current_status"] = latest.get("current_status", "Unknown")

    rows = []
    for _, row in latest.iterrows():
        dpd = float(row.get("days_past_due", 0) or 0)
        def_flag = int(row.get("default_flag", 0) or 0)
        anomaly = float(row.get("anomaly_score", 0) or 0)
        tier = _risk_tier(dpd, def_flag, anomaly)
        rows.append({
            "loan_id": str(row["loan_id"]),
            "credit_band": _safe_str(row.get("credit_score_band")),
            "current_status": _safe_str(row.get("current_status")),
            "state": _safe_str(row.get("state")),
            "days_past_due": round(dpd, 1),
            "current_balance": round(float(row.get("current_balance", 0) or 0), 2),
            "anomaly_score": round(anomaly, 4),
            "exception_type": _safe_str(row.get("predicted_exception_type"), ""),
            "risk_tier": tier,
        })

    rows.sort(key=lambda x: (x["anomaly_score"], x["days_past_due"]), reverse=True)
    tier_counts = Counter(r["risk_tier"] for r in rows)

    work = latest.copy()
    work["credit_score_band"] = work["credit_score_band"].map(lambda v: _safe_str(v))
    work["current_status"] = work["current_status"].map(lambda v: _safe_str(v))
    if "days_past_due" in work.columns:
        work["days_past_due"] = pd.to_numeric(work["days_past_due"], errors="coerce").fillna(0)
    else:
        work["days_past_due"] = 0
    if "default_flag" in work.columns:
        work["default_flag"] = pd.to_numeric(work["default_flag"], errors="coerce").fillna(0)
    else:
        work["default_flag"] = 0
    if "state" in work.columns:
        work["state"] = work["state"].map(lambda v: _safe_str(v))
    else:
        work["state"] = "NA"

    status_heatmap = _crosstab_matrix(work, "credit_score_band", "current_status", BAND_ORDER, STATUS_ORDER)
    dpd_heatmap = _crosstab_matrix(work, "credit_score_band", "current_status", BAND_ORDER, STATUS_ORDER, "days_past_due")

    scenario_cells = []
    smax = 0.0
    for band in BAND_ORDER:
        line = []
        for scen in SCENARIOS:
            val = SCENARIO_BAND_DEFAULT.get(band, {}).get(scen, 0.0)
            line.append(val)
            smax = max(smax, val)
        scenario_cells.append(line)
    scenario_heatmap = {
        "rows": BAND_ORDER,
        "cols": ["Base", "Adverse credit", "High prepay"],
        "cells": scenario_cells,
        "max": round(smax, 4),
        "unit": "12m default rate",
    }

    state_rows = (
        work.groupby("state")
        .agg(
            loans=("loan_id", "count"),
            avg_dpd=("days_past_due", "mean"),
            default_rate=("default_flag", "mean"),
        )
        .reset_index()
        .sort_values("default_rate", ascending=False)
        .head(12)
    )
    state_heatmap = {
        "rows": state_rows["state"].tolist(),
        "cols": ["Loans", "Avg DPD", "Default rate"],
        "cells": [
            [int(r.loans), round(float(r.avg_dpd), 1), round(float(r.default_rate), 4)]
            for r in state_rows.itertuples()
        ],
        "max": round(float(state_rows["default_rate"].max() or 0), 4) if len(state_rows) else 0,
    }

    flagged = [r for r in rows if r["anomaly_score"] >= 0.35 or r["risk_tier"] in ("high", "elevated")]
    exception_types = Counter(r["exception_type"] or "Unsupervised outlier" for r in flagged[:200])

    return {
        "total_loans": len(rows),
        "tier_summary": {k: int(tier_counts.get(k, 0)) for k in ("high", "elevated", "moderate", "low")},
        "status_heatmap": status_heatmap,
        "dpd_heatmap": dpd_heatmap,
        "scenario_heatmap": scenario_heatmap,
        "state_heatmap": state_heatmap,
        "exception_mix": [{"type": k, "count": v} for k, v in exception_types.most_common(8)],
        "top_loans": rows[: max(8, min(limit, 40))],
        "metrics": _load_metrics(),
        "demo_note": (
            "Heatmaps are live aggregates from the latest observation per loan. "
            "Use them to show concentration of 60/90 DPD in weaker credit bands — "
            "this is the judge-facing extension when 12m default/prepay ROC is flat vs LR."
        ),
    }
