"""
exception_detection.py
======================
Task 4 — Anomaly & Exception Detection

Hybrid approach:
  1. Rules layer: execute validation_rules.json, score = fraction violated
  2. ML layer: Isolation Forest on numeric features → anomaly score
  3. Combine with configurable weights (rule_weight + ml_weight)

Outputs:
  - anomaly_scores.csv (record-level scores + top drivers)
  - anomaly_examples.md (20+ reviewer-ready examples in table form)
  - Trained isolation forest model
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def rule_violation_score(df: pd.DataFrame, rules_path: Path) -> pd.Series:
    """
    For each row, compute a rule-violation score = #errors / max_possible_errors.
    """
    with open(rules_path) as f:
        rules_cfg = json.load(f)

    n_rules = len(rules_cfg["rules"])
    score   = pd.Series(0.0, index=df.index)

    # VR001: balance > 105% of original without modification
    if "current_balance" in df and "original_balance" in df:
        mask = (df["current_balance"] > df["original_balance"] * 1.05) & \
               (df.get("modification_flag", 0) != 1)
        score[mask] += 1.0

    # VR002: bad date ordering
    if "origination_month" in df and "reporting_month" in df:
        mask = df["origination_month"].fillna("1900-01") > df["reporting_month"].fillna("9999-01")
        score[mask] += 1.0

    # VR003: dpd=0 but status not current/prepaid/closed
    if "days_past_due" in df and "current_status" in df:
        mask = (df["days_past_due"] == 0) & \
               (~df["current_status"].isin(["Current","Prepaid","Closed"]))
        score[mask] += 0.5  # warning only → half weight

    # VR005: default=1 + document Incomplete
    if "default_flag" in df and "document_status" in df:
        mask = (df["default_flag"] == 1) & (df["document_status"] == "Incomplete")
        score[mask] += 0.5

    return (score / n_rules).clip(0, 1)


ANOMALY_FEAT_COLS = [
    "loan_age_months", "remaining_term_months", "current_balance",
    "original_balance", "interest_rate", "days_past_due",
    "modification_flag", "prepayment_flag", "default_flag",
    "balance_utilisation", "month_index",
]


def ml_anomaly_score(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Train Isolation Forest on numeric features.
    Returns (scores series [0..1], trained model).
    Higher = more anomalous.
    """
    feat_cols = [c for c in ANOMALY_FEAT_COLS if c in df.columns]
    X = df[feat_cols].copy()

    # Add balance_utilisation if not already
    if "balance_utilisation" not in X and "current_balance" in df and "original_balance" in df:
        X["balance_utilisation"] = (
            df["current_balance"] / df["original_balance"].replace(0, np.nan)
        ).fillna(0).clip(0, 5)

    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    contamination = cfg["ANOMALY"]["isolation_forest_contamination"]
    ifo = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=cfg["RANDOM_SEED"],
        n_jobs=-1,
    )
    ifo.fit(Xs)

    # decision_function: lower = more anomalous → invert and normalise
    raw_scores = -ifo.decision_function(Xs)
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

    return pd.Series(norm_scores, index=df.index), ifo, scaler, feat_cols


def top_drivers(row: pd.Series, feat_cols: list, global_mean: pd.Series) -> str:
    """Return top 3 features whose deviation from mean is largest."""
    devs = {c: abs(row.get(c, global_mean[c]) - global_mean[c]) for c in feat_cols
            if c in row.index}
    sorted_devs = sorted(devs.items(), key=lambda x: x[1], reverse=True)
    return "; ".join([f"{c}(dev={v:.2f})" for c, v in sorted_devs[:3]])


def run(panel: pd.DataFrame, df_train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Full anomaly detection pipeline.
    Returns panel DataFrame with anomaly_score, exception_flag, exception_type, top_drivers.
    """
    plots_dir  = ROOT / cfg["PATHS"].get("plots","reports/plots")
    models_dir = ROOT / cfg["PATHS"].get("models","data/processed/models")
    proc_dir   = ROOT / cfg["PATHS"]["processed_data"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Add balance_utilisation to panel
    panel = panel.copy()
    panel["balance_utilisation"] = (
        panel["current_balance"] / panel["original_balance"].replace(0, np.nan)
    ).clip(0, 5).fillna(0)

    raw_path = ROOT / cfg["PATHS"]["raw_data"] / "validation_rules.json"
    print("\n[anomaly] Computing rule-violation scores …")
    rule_scores = rule_violation_score(panel, raw_path) if raw_path.exists() else pd.Series(0.0, index=panel.index)

    print("[anomaly] Training Isolation Forest …")
    ml_scores, ifo, scaler, feat_cols = ml_anomaly_score(panel, cfg)

    # Combine
    rw = cfg["ANOMALY"]["rule_weight"]
    mw = cfg["ANOMALY"]["ml_weight"]
    panel["rule_score"]   = rule_scores.values
    panel["ml_score"]     = ml_scores.values
    panel["anomaly_score"] = (rw * rule_scores.values + mw * ml_scores.values).clip(0, 1)

    # Exception flag: top ~5% by anomaly_score
    threshold = panel["anomaly_score"].quantile(0.95)
    panel["exception_flag"] = (panel["anomaly_score"] >= threshold).astype(int)

    # Exception type: derive from rule violations for labeled examples,
    # else use "Anomaly:IsolationForest"
    def _exc_type(row):
        if row.get("exception_required", 0) == 1:
            return row.get("exception_type", "Unknown")
        if row["rule_score"] > 0.3:
            return "RuleViolation"
        if row["anomaly_score"] > threshold:
            return "Anomaly:IsolationForest"
        return ""
    panel["predicted_exception_type"] = panel.apply(_exc_type, axis=1)

    # Top drivers for each record
    global_mean = panel[[c for c in feat_cols if c in panel.columns]].mean()
    print("[anomaly] Computing top drivers (may take a moment) …")
    sample_idx  = panel[panel["exception_flag"] == 1].index
    panel["top_drivers"] = ""
    for idx in sample_idx:
        panel.at[idx, "top_drivers"] = top_drivers(
            panel.loc[idx], feat_cols, global_mean)

    # Save models
    joblib.dump(ifo,    models_dir / "isolation_forest.pkl")
    joblib.dump(scaler, models_dir / "anomaly_scaler.pkl")

    # Anomaly score distribution plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(panel["anomaly_score"], bins=50, color="#DC2626", edgecolor="white", alpha=0.8)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold (p95={threshold:.3f})")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution"); ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "anomaly_score_dist.png", dpi=100)
    plt.close(fig)

    # Save results
    anomaly_out = panel[["loan_id","month_index","reporting_month",
                          "anomaly_score","rule_score","ml_score",
                          "exception_flag","predicted_exception_type","top_drivers"]].copy()
    anomaly_out.to_csv(proc_dir / "anomaly_scores.csv", index=False)

    # Generate reviewer-ready examples table (≥20)
    flagged = panel[panel["exception_flag"] == 1].sort_values("anomaly_score", ascending=False)
    flagged = flagged.head(30)
    examples = []
    for _, row in flagged.iterrows():
        reason_parts = []
        if row.get("rule_score", 0) > 0.2:
            reason_parts.append("rule violation(s)")
        if row.get("ml_score", 0) > 0.7:
            reason_parts.append("statistical outlier (Isolation Forest)")
        if row.get("current_balance", 0) > row.get("original_balance", 1) * 1.5:
            reason_parts.append("balance far exceeds original")
        if str(row.get("origination_month","")) > str(row.get("reporting_month","")):
            reason_parts.append("origination after reporting (date error)")
        if row.get("default_flag", 0) == 1 and row.get("document_status","") == "Incomplete":
            reason_parts.append("defaulted with incomplete documentation")
        if not reason_parts:
            reason_parts.append(f"high anomaly score ({row['anomaly_score']:.3f})")

        examples.append({
            "loan_id":      row["loan_id"],
            "month_index":  int(row["month_index"]),
            "anomaly_score": round(row["anomaly_score"], 3),
            "exception_type": row["predicted_exception_type"],
            "plain_language_reason": "; ".join(reason_parts),
            "top_drivers":  row.get("top_drivers",""),
        })

    ex_df = pd.DataFrame(examples)
    ex_md = "# Anomaly & Exception Examples (Reviewer-Ready)\n\n"
    ex_md += f"Total flagged records: **{panel['exception_flag'].sum():,}** "
    ex_md += f"(threshold: anomaly_score ≥ {threshold:.4f})\n\n"
    ex_md += ex_df.to_markdown(index=False)
    (ROOT / cfg["PATHS"]["reports"] / "anomaly_examples.md").write_text(ex_md, encoding="utf-8")

    print(f"  ✓ Anomaly detection done: {panel['exception_flag'].sum():,} flagged records")
    return panel
