"""
load_and_validate.py
====================
Loads all raw data files, runs deterministic validation rules, and outputs
a validation summary. Returns loaded DataFrames for downstream use.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def load_all(cfg: dict) -> dict:
    """Load all raw CSVs. Returns dict of DataFrames."""
    raw = ROOT / cfg["PATHS"]["raw_data"]
    dfs = {}
    files = {
        "static":       "loan_static_attributes.csv",
        "train":        "loan_monthly_performance_train.csv",
        "test":         "loan_monthly_performance_test.csv",
        "servicer":     "servicer_updates.csv",
        "scenarios":    "macro_scenarios.csv",
        "template":     "submission_template.csv",
    }
    for key, fname in files.items():
        path = raw / fname
        if path.exists():
            dfs[key] = pd.read_csv(path, low_memory=False)
            print(f"  ✓ Loaded {fname}: {len(dfs[key]):,} rows")
        else:
            print(f"  ✗ Missing: {fname}")
    return dfs


def run_validation(df: pd.DataFrame, rules_path: Path) -> pd.DataFrame:
    """
    Execute validation_rules.json checks against the panel DataFrame.
    Returns a DataFrame of violations with columns:
      loan_id, month_index, rule_id, rule_name, severity, details
    """
    with open(rules_path) as f:
        rules_cfg = json.load(f)

    violations = []

    for rule in rules_cfg["rules"]:
        rid  = rule["rule_id"]
        name = rule["name"]
        sev  = rule["severity"]

        if rid == "VR001":
            mask = (df["current_balance"] > df["original_balance"] * 1.05) & (df["modification_flag"] != 1)
            _add(violations, df[mask], rid, name, sev, "current_balance > 105% of original")

        elif rid == "VR002":
            # Compare YYYY-MM strings lexicographically
            mask = df["origination_month"] > df["reporting_month"]
            _add(violations, df[mask], rid, name, sev, "origination_month > reporting_month")

        elif rid == "VR003":
            mask = (df["days_past_due"] == 0) & (~df["current_status"].isin(["Current", "Prepaid", "Closed"]))
            _add(violations, df[mask], rid, name, sev, "dpd=0 but status not Current/Prepaid/Closed")

        elif rid == "VR005":
            mask = (df["default_flag"] == 1) & (df["document_status"] == "Incomplete")
            _add(violations, df[mask], rid, name, sev, "default=1 but doc status Incomplete")

        elif rid == "VR006":
            df_sorted = df.sort_values(["loan_id", "month_index"])
            df_sorted["prev_balance"] = df_sorted.groupby("loan_id")["current_balance"].shift(1)
            mask = (
                df_sorted["current_balance"] > df_sorted["prev_balance"] * 1.02
            ) & (df_sorted["modification_flag"] != 1) & (df_sorted["month_index"] > 1)
            _add(violations, df_sorted[mask], rid, name, sev, "balance increased without modification")

    vdf = pd.DataFrame(violations) if violations else pd.DataFrame(
        columns=["loan_id","month_index","rule_id","rule_name","severity","details"])
    return vdf


def _add(lst, sub_df, rid, name, sev, details):
    for _, row in sub_df.iterrows():
        lst.append({
            "loan_id":     row.get("loan_id", ""),
            "month_index": row.get("month_index", -1),
            "rule_id":     rid,
            "rule_name":   name,
            "severity":    sev,
            "details":     details,
        })


def validate_and_save(cfg: dict, dfs: dict) -> pd.DataFrame:
    raw = ROOT / cfg["PATHS"]["raw_data"]
    proc = ROOT / cfg["PATHS"]["processed_data"]
    proc.mkdir(parents=True, exist_ok=True)

    panel = pd.concat([dfs.get("train", pd.DataFrame()), dfs.get("test", pd.DataFrame())],
                      ignore_index=True)

    rules_path = raw / "validation_rules.json"
    if not rules_path.exists():
        print("  ⚠ validation_rules.json not found, skipping rule checks")
        return pd.DataFrame()

    violations = run_validation(panel, rules_path)
    violations.to_csv(proc / "validation_violations.csv", index=False)
    print(f"  ✓ Validation complete: {len(violations):,} rule violations found")
    summary = violations.groupby(["rule_id","rule_name","severity"]).size().reset_index(name="count")
    print(summary.to_string(index=False))
    return violations


if __name__ == "__main__":
    cfg = load_cfg()
    dfs = load_all(cfg)
    validate_and_save(cfg, dfs)
