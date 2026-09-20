"""
ingestion.py — Stage 1: Data Ingestion
Retrieves actual loan records from the real dataset CSV files.
"""

import time
import pandas as pd
from models.loader import _DATA, RAW_DIR


def run(loan_id: str) -> dict:
    t0 = time.perf_counter()

    panel: pd.DataFrame = _DATA.get("panel", pd.DataFrame())
    static: pd.DataFrame = _DATA.get("static", pd.DataFrame())

    if panel.empty:
        return {"status": "error", "error": "Panel data not loaded"}

    # Filter to the specific loan
    loan_records = panel[panel["loan_id"] == loan_id].copy()

    if loan_records.empty:
        return {
            "status": "error",
            "error": f"Loan ID '{loan_id}' not found in dataset. "
                     f"Total unique loans: {panel['loan_id'].nunique():,}"
        }

    # Merge static attributes if available
    if not static.empty and "loan_id" in static.columns:
        static_row = static[static["loan_id"] == loan_id]
        if not static_row.empty:
            static_dict = static_row.iloc[0].to_dict()
        else:
            static_dict = {}
    else:
        static_dict = {}

    # Determine the latest record
    loan_records_sorted = loan_records.sort_values("month_index")
    latest = loan_records_sorted.iloc[-1]

    # Build a clean record of current state
    def safe(v):
        if pd.isna(v): return None
        if hasattr(v, "item"): return v.item()
        return v

    record_fields = [
        "loan_id", "month_index", "reporting_month", "current_status",
        "days_past_due", "current_balance", "original_balance",
        "interest_rate", "loan_age_months", "remaining_term_months",
        "modification_flag", "prepayment_flag", "default_flag",
        "document_status", "origination_month",
    ]

    current_record = {}
    for f in record_fields:
        if f in latest.index:
            current_record[f] = safe(latest[f])

    # Add static fields (credit_score_band, ltv_band, dti_band, state, etc.)
    for f in ["credit_score_band", "ltv_band", "dti_band", "state",
              "loan_purpose", "servicer_name", "occupancy_type", "property_type"]:
        if f in static_dict:
            current_record[f] = safe(static_dict[f])
        elif f in latest.index:
            current_record[f] = safe(latest[f])

    # Summary statistics for this loan
    n_records = len(loan_records)
    min_month = int(loan_records["month_index"].min()) if "month_index" in loan_records.columns else None
    max_month = int(loan_records["month_index"].max()) if "month_index" in loan_records.columns else None
    source_set = []
    if not _DATA.get("train", pd.DataFrame()).empty:
        in_train = loan_id in _DATA["train"]["loan_id"].values
        if in_train: source_set.append("train")
    if not _DATA.get("test", pd.DataFrame()).empty:
        in_test = loan_id in _DATA["test"]["loan_id"].values
        if in_test: source_set.append("test")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "loan_id": loan_id,
        "records_retrieved": n_records,
        "source_datasets": source_set if source_set else ["panel"],
        "source_file": "loan_monthly_performance_train/test.csv + loan_static_attributes.csv",
        "month_range": {"first": min_month, "last": max_month},
        "reporting_period": f"Month {min_month} – Month {max_month}",
        "current_record": current_record,
        "all_months": loan_records_sorted[["month_index", "reporting_month", "current_status",
                                           "days_past_due", "current_balance"]].fillna(0).to_dict("records"),
        "execution_ms": elapsed_ms,
    }
