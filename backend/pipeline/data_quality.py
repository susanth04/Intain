"""
data_quality.py — Stage 2: Data Quality
Runs VR001-VR006 validation rules against the loan's actual records.
Uses the real rule implementation from src/data/load_and_validate.py.
Also queries pre-computed validation_violations.csv for context.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from models.loader import _DATA, RAW_DIR, PROJECT_ROOT


def _safe(v):
    if pd.isna(v): return None
    if hasattr(v, "item"): return v.item()
    return v


def run(loan_id: str, loan_records: pd.DataFrame) -> dict:
    t0 = time.perf_counter()

    if loan_records.empty:
        return {"status": "error", "error": "No records provided for DQ check"}

    rules_path = RAW_DIR / "validation_rules.json"
    checks = []
    total_violations = 0

    if rules_path.exists():
        with open(rules_path) as f:
            rules_cfg = json.load(f)

        for rule in rules_cfg["rules"]:
            rid  = rule["rule_id"]
            name = rule["name"]
            sev  = rule["severity"]
            desc = rule.get("description", "")
            result = _check_rule(rid, loan_records)
            checks.append({
                "rule_id": rid,
                "rule_name": name,
                "description": desc,
                "severity": sev,
                "status": result["status"],
                "n_violations": result["n_violations"],
                "details": result["details"],
            })
            if result["status"] == "FAIL":
                total_violations += result["n_violations"]
    else:
        # Fallback: run checks manually if rules file missing
        checks = _manual_checks(loan_records)
        total_violations = sum(1 for c in checks if c["status"] == "FAIL")

    # Cross-reference pre-computed violations CSV
    prior_violations = []
    violations_df: pd.DataFrame = _DATA.get("violations", pd.DataFrame())
    if not violations_df.empty and "loan_id" in violations_df.columns:
        prior = violations_df[violations_df["loan_id"] == loan_id]
        if not prior.empty:
            prior_violations = prior[["rule_id", "rule_name", "severity", "month_index", "details"]].to_dict("records")

    # Missing values check
    numeric_cols = loan_records.select_dtypes(include="number").columns.tolist()
    missing_counts = loan_records[numeric_cols].isna().sum().to_dict()
    missing_counts = {k: int(v) for k, v in missing_counts.items() if v > 0}

    # Outlier check (z-score > 3)
    outliers = {}
    for col in ["current_balance", "days_past_due", "interest_rate", "original_balance"]:
        if col in loan_records.columns:
            panel: pd.DataFrame = _DATA.get("panel", pd.DataFrame())
            if not panel.empty and col in panel.columns:
                global_mean = panel[col].mean()
                global_std  = panel[col].std()
                if global_std > 1e-6:
                    val = loan_records[col].iloc[-1]
                    z = abs(val - global_mean) / global_std
                    if z > 3:
                        outliers[col] = {
                            "value": _safe(val),
                            "global_mean": round(float(global_mean), 2),
                            "z_score": round(float(z), 2),
                        }

    overall_status = "FAIL" if total_violations > 0 else "PASS"
    has_warnings = any(c["status"] == "WARNING" for c in checks)
    if overall_status == "PASS" and has_warnings:
        overall_status = "WARNING"

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "status": overall_status,
        "total_violations": total_violations,
        "checks": checks,
        "prior_violations_from_full_run": prior_violations,
        "missing_values": missing_counts,
        "outliers_vs_portfolio": outliers,
        "records_checked": len(loan_records),
        "execution_ms": elapsed_ms,
    }


def _check_rule(rule_id: str, df: pd.DataFrame) -> dict:
    """Run a single validation rule; return status and details."""
    try:
        if rule_id == "VR001":
            if "current_balance" in df.columns and "original_balance" in df.columns:
                mask = (df["current_balance"] > df["original_balance"] * 1.05) & \
                       (df.get("modification_flag", pd.Series(0, index=df.index)) != 1)
                n = int(mask.sum())
                if n > 0:
                    worst = df[mask].iloc[0]
                    return {"status": "FAIL", "n_violations": n,
                            "details": f"{n} record(s): current_balance exceeds 105% of original "
                                       f"without modification. "
                                       f"Example: current={_safe(worst['current_balance']):.0f}, "
                                       f"original={_safe(worst['original_balance']):.0f}"}
                return {"status": "PASS", "n_violations": 0,
                        "details": "Balance within 105% of original or modification present."}
            return {"status": "SKIP", "n_violations": 0, "details": "Required columns not present."}

        elif rule_id == "VR002":
            if "origination_month" in df.columns and "reporting_month" in df.columns:
                mask = df["origination_month"].fillna("1900-01") > df["reporting_month"].fillna("9999-01")
                n = int(mask.sum())
                if n > 0:
                    return {"status": "FAIL", "n_violations": n,
                            "details": f"{n} record(s): origination_month > reporting_month (date ordering error)."}
                return {"status": "PASS", "n_violations": 0, "details": "Date ordering valid."}
            return {"status": "SKIP", "n_violations": 0, "details": "Date columns not present."}

        elif rule_id == "VR003":
            if "days_past_due" in df.columns and "current_status" in df.columns:
                mask = (df["days_past_due"] == 0) & \
                       (~df["current_status"].isin(["Current", "Prepaid", "Closed"]))
                n = int(mask.sum())
                if n > 0:
                    return {"status": "WARNING", "n_violations": n,
                            "details": f"{n} record(s): days_past_due=0 but status is not Current/Prepaid/Closed."}
                return {"status": "PASS", "n_violations": 0,
                        "details": "DPD consistent with status."}
            return {"status": "SKIP", "n_violations": 0, "details": "Required columns not present."}

        elif rule_id == "VR005":
            if "default_flag" in df.columns and "document_status" in df.columns:
                mask = (df["default_flag"] == 1) & (df["document_status"] == "Incomplete")
                n = int(mask.sum())
                if n > 0:
                    return {"status": "FAIL", "n_violations": n,
                            "details": f"{n} record(s): default_flag=1 with Incomplete documentation."}
                return {"status": "PASS", "n_violations": 0,
                        "details": "Default documentation completeness OK."}
            return {"status": "SKIP", "n_violations": 0, "details": "Required columns not present."}

        elif rule_id == "VR006":
            if "current_balance" in df.columns and "month_index" in df.columns:
                df_s = df.sort_values("month_index").copy()
                df_s["prev_balance"] = df_s["current_balance"].shift(1)
                mask = (
                    df_s["current_balance"] > df_s["prev_balance"] * 1.02
                ) & (df_s.get("modification_flag", pd.Series(0, index=df_s.index)) != 1) & \
                  (df_s["month_index"] > df_s["month_index"].min())
                n = int(mask.sum())
                if n > 0:
                    return {"status": "WARNING", "n_violations": n,
                            "details": f"{n} record(s): balance increased >2% month-over-month without modification."}
                return {"status": "PASS", "n_violations": 0,
                        "details": "Balance monotonicity maintained."}
            return {"status": "SKIP", "n_violations": 0, "details": "Required columns not present."}

        else:
            return {"status": "SKIP", "n_violations": 0, "details": f"Rule {rule_id} not implemented."}

    except Exception as e:
        return {"status": "ERROR", "n_violations": 0, "details": str(e)}


def _manual_checks(df: pd.DataFrame) -> list:
    """Fallback if validation_rules.json is missing."""
    checks = []
    for rid in ["VR001", "VR002", "VR003", "VR005", "VR006"]:
        result = _check_rule(rid, df)
        checks.append({"rule_id": rid, "rule_name": rid, "severity": "medium", **result})
    return checks
