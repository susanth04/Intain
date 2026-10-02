"""
survival.py - Survival time estimate for a single loan using Cox PH + KM models.
"""
import json
import time
import pandas as pd
from models.loader import MODELS, _DATA, PROC_DIR

CREDIT_MAP = {"Excellent": 0, "Good": 1, "Fair": 2, "Poor": 3}
LTV_MAP = {"<=60": 0, "60-75": 1, "75-90": 2, ">90": 3}
_FALLBACK_MEDIANS = {"Excellent": 16.0, "Good": 16.0, "Fair": 13.0, "Poor": 10.0}


def _load_metrics() -> dict:
    try:
        with open(PROC_DIR / "survival_metrics.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_METRICS = _load_metrics()
CREDIT_BAND_MEDIANS = {
    **_FALLBACK_MEDIANS,
    **{k: float(v) for k, v in (_METRICS.get("km_median_survival") or {}).items() if v is not None},
}
COX_CONCORDANCE = _METRICS.get("cox_concordance")


def run(loan_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        panel = _DATA.get("panel")
        if panel is None or panel.empty:
            return {"status": "error", "error": "Panel not loaded"}

        rows = panel[panel["loan_id"] == loan_id]
        if rows.empty:
            return {"status": "error", "error": f"Loan {loan_id} not found"}

        sort_col = "month_index" if "month_index" in rows.columns else rows.columns[1]
        row = rows.sort_values(sort_col).iloc[-1]

        loan_age = float(row.get("loan_age_months", 0) or 0)
        credit_band = str(row.get("credit_score_band", "Fair"))
        ltv_band = str(row.get("ltv_band", ""))
        current_status = str(row.get("current_status", "Unknown"))

        km_median = CREDIT_BAND_MEDIANS.get(credit_band, 13.0)
        months_remaining = max(0.0, round(km_median - loan_age, 1))

        cox = MODELS.get("cox_ph")
        cox_hazard_ratio = None
        cox_interpretation = None
        if cox is not None:
            try:
                # Same encoding the Cox model was trained with (1.5 = unknown band)
                credit_enc = CREDIT_MAP.get(credit_band, 1.5)
                ltv_enc = LTV_MAP.get(ltv_band, 1.5)
                cox_df = pd.DataFrame([{"credit_enc": credit_enc, "ltv_enc": ltv_enc}])
                hr = float(cox.predict_partial_hazard(cox_df).iloc[0])
                cox_hazard_ratio = round(hr, 4)
                if hr > 1.5:
                    cox_interpretation = "Elevated hazard — significantly above baseline"
                elif hr > 1.0:
                    cox_interpretation = "Moderate hazard — above baseline"
                else:
                    cox_interpretation = "Low hazard — below baseline"
            except Exception as e:
                cox_interpretation = f"Cox model unavailable: {e}"

        return {
            "loan_id": loan_id,
            "loan_age_months": loan_age,
            "credit_band": credit_band,
            "ltv_band": ltv_band,
            "current_status": current_status,
            "km_median_survival_months": km_median,
            "estimated_months_remaining": months_remaining,
            "survival_label": (
                f"KM median for '{credit_band}' credit band: {km_median:.0f} months. "
                f"Loan is {loan_age:.0f} months old — ~{months_remaining:.0f} months to expected exit."
            ),
            "cox_hazard_ratio": cox_hazard_ratio,
            "cox_interpretation": cox_interpretation,
            "cox_concordance": COX_CONCORDANCE,
            "execution_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}