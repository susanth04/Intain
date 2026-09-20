"""
survival.py - Survival time estimate for a single loan using Cox PH + KM models.
"""
import time
from models.loader import MODELS, _DATA, PROJECT_ROOT
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CREDIT_BAND_MEDIANS = {
    "Excellent": 16.0,
    "Good":      16.0,
    "Fair":      13.0,
    "Poor":      10.0,
}

def run(loan_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        panel = _DATA.get("panel")
        if panel is None or panel.empty:
            return {"status": "error", "error": "Panel not loaded"}

        rows = panel[panel["loan_id"] == loan_id]
        if rows.empty:
            return {"status": "error", "error": f"Loan {loan_id} not found"}

        # Sort by month_index (correct column name)
        sort_col = "month_index" if "month_index" in rows.columns else rows.columns[1]
        row = rows.sort_values(sort_col).iloc[-1]

        loan_age = float(row.get("loan_age_months", 0) or 0)
        credit_band = str(row.get("credit_score_band", "Fair"))
        current_status = str(row.get("current_status", "Unknown"))

        km_median = CREDIT_BAND_MEDIANS.get(credit_band, 13.0)
        months_remaining = max(0.0, round(km_median - loan_age, 1))

        # Cox PH model
        cox = MODELS.get("cox_ph")
        cox_hazard_ratio = None
        cox_interpretation = None
        if cox is not None:
            try:
                import pandas as pd
                credit_enc = float(row.get("credit_score_band_enc", 2) or 2)
                ltv_enc    = float(row.get("ltv_band_enc", 2) or 2)
                cox_df = pd.DataFrame([{"credit_enc": credit_enc, "ltv_enc": ltv_enc}])
                hr = cox.predict_partial_hazard(cox_df).iloc[0]
                cox_hazard_ratio = round(float(hr), 4)
                if hr > 1.5:
                    cox_interpretation = "Elevated hazard — significantly above baseline"
                elif hr > 1.0:
                    cox_interpretation = "Moderate hazard — above baseline"
                else:
                    cox_interpretation = "Low hazard — below baseline"
            except Exception as e:
                cox_interpretation = f"Cox model unavailable: {e}"

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "loan_id": loan_id,
            "loan_age_months": loan_age,
            "credit_band": credit_band,
            "current_status": current_status,
            "km_median_survival_months": km_median,
            "estimated_months_remaining": months_remaining,
            "survival_label": (
                f"KM median for '{credit_band}' credit band: {km_median:.0f} months. "
                f"Loan is {loan_age:.0f} months old — ~{months_remaining:.0f} months to expected exit."
            ),
            "cox_hazard_ratio": cox_hazard_ratio,
            "cox_interpretation": cox_interpretation,
            "execution_ms": elapsed_ms,
        }
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
