"""
hazard_transition_model.py
===========================
Task 3 — Time-to-Event / Survival Modeling

Implements:
  1. Kaplan-Meier survival curves by segment
  2. Cox Proportional Hazards model (lifelines)
  3. Competing-risks framing (default vs. prepayment as competing exits)
  4. Monthly discrete-time hazard model
  5. Naive baseline comparison (constant hazard rate)

Outputs:
  - KM survival curves plot (by credit_score_band, vintage)
  - Cumulative incidence curves (competing risks)
  - Hazard model summary
  - reports/survival_metrics.json
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json
from pathlib import Path
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def prepare_survival_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build a loan-level survival dataset:
      - T     : observed months until event or censoring
      - E     : event indicator (1 = default or prepaid, 0 = censored)
      - event_type : 'default' | 'prepaid' | 'censored'

    Censoring: loans still in 'Current' or 'DPD' at data cutoff.
    """
    # Get last observation per loan
    last = panel.sort_values(["loan_id","month_index"]).groupby("loan_id").last().reset_index()

    def _classify(row):
        s = row["current_status"]
        if s == "Default":
            return row["loan_age_months"], 1, "default"
        elif s == "Prepaid":
            return row["loan_age_months"], 1, "prepaid"
        else:
            return row["loan_age_months"], 0, "censored"

    surv = last.apply(lambda r: pd.Series(_classify(r),
                                           index=["T","E","event_type"]), axis=1)
    result = pd.concat([last[["loan_id","credit_score_band",
                              "ltv_band","dti_band","state","loan_purpose"]],
                        surv], axis=1)
    result["T"] = result["T"].clip(lower=1)
    return result


def run(panel: pd.DataFrame, cfg: dict) -> dict:
    plots_dir = ROOT / cfg["PATHS"].get("plots","reports/plots")
    proc_dir  = ROOT / cfg["PATHS"]["processed_data"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[survival] Preparing survival dataset …")
    surv_df = prepare_survival_data(panel)
    surv_df.to_csv(proc_dir / "survival_data.csv", index=False)

    outcomes = {}

    # ── 1. Kaplan-Meier by credit band ───────────────────────────────────
    try:
        from lifelines import KaplanMeierFitter
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#2563EB","#16A34A","#D97706","#DC2626"]

        for i, band in enumerate(["Excellent","Good","Fair","Poor"]):
            sub = surv_df[surv_df["credit_score_band"] == band]
            if len(sub) < 10: continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["T"], event_observed=sub["E"], label=band)
            kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)], ci_show=True)

        ax.set_xlabel("Months since Origination")
        ax.set_ylabel("Survival Probability (no default / prepay)")
        ax.set_title("Kaplan-Meier Survival Curves by Credit Band")
        ax.legend(title="Credit Band")
        plt.tight_layout()
        fig.savefig(plots_dir / "km_survival_credit_band.png", dpi=100)
        plt.close(fig)
        print("  ✓ KM curves by credit band")

        # Naive baseline: constant hazard = overall event rate / mean time
        overall_rate = surv_df["E"].mean()
        mean_T       = surv_df["T"].mean()
        naive_hazard = overall_rate / mean_T
        naive_surv   = np.exp(-naive_hazard * np.arange(0, int(surv_df["T"].max())+1))
        outcomes["naive_hazard_rate"]    = round(naive_hazard, 5)
        outcomes["km_median_survival"]   = {}

        for band in ["Excellent","Good","Fair","Poor"]:
            sub = surv_df[surv_df["credit_score_band"] == band]
            if len(sub) < 10: continue
            kmf = KaplanMeierFitter()
            kmf.fit(sub["T"], event_observed=sub["E"])
            med = kmf.median_survival_time_
            outcomes["km_median_survival"][band] = float(med) if not np.isinf(med) else None

    except ImportError:
        print("  ⚠ lifelines not available, skipping KM")

    # ── 2. Cox Proportional Hazards ───────────────────────────────────────
    try:
        from lifelines import CoxPHFitter
        cox_df = surv_df.copy()

        # Encode categoricals
        credit_map = {"Excellent":0,"Good":1,"Fair":2,"Poor":3}
        ltv_map    = {"<=60":0,"60-75":1,"75-90":2,">90":3}
        cox_df["credit_enc"] = cox_df["credit_score_band"].map(credit_map).fillna(1.5)
        cox_df["ltv_enc"]    = cox_df["ltv_band"].map(ltv_map).fillna(1.5)

        cox_features = ["T","E","credit_enc","ltv_enc"]
        cox_clean = cox_df[cox_features].dropna()

        cph = CoxPHFitter()
        cph.fit(cox_clean, duration_col="T", event_col="E")

        fig, ax = plt.subplots(figsize=(6, 3))
        cph.plot(ax=ax)
        ax.set_title("Cox PH — Hazard Ratios")
        plt.tight_layout()
        fig.savefig(plots_dir / "cox_hazard_ratios.png", dpi=100)
        plt.close(fig)

        outcomes["cox_concordance"] = round(cph.concordance_index_, 4)
        print(f"  ✓ Cox PH: C-index = {cph.concordance_index_:.4f}")

        joblib.dump(cph, ROOT / cfg["PATHS"]["models"] / "cox_ph.pkl")

    except Exception as e:
        print(f"  ⚠ Cox PH failed: {e}")

    # ── 3. Competing Risks (default vs. prepaid) ──────────────────────────
    try:
        from lifelines import AalenJohansenFitter

        fig, ax = plt.subplots(figsize=(9, 5))
        for event_label, event_code, color in [
            ("Default", 1, "#DC2626"),
            ("Prepaid", 2, "#2563EB"),
        ]:
            # code events: 0=censored, 1=default, 2=prepaid
            e_coded = surv_df["event_type"].map(
                {"default": 1, "prepaid": 2, "censored": 0}).fillna(0).astype(int)
            ajf = AalenJohansenFitter(calculate_variance=True)
            ajf.fit(surv_df["T"], e_coded, event_of_interest=event_code,
                    label=event_label)
            ajf.plot_cumulative_density(ax=ax, color=color)

        ax.set_xlabel("Months since Origination")
        ax.set_ylabel("Cumulative Incidence")
        ax.set_title("Competing Risks — Cumulative Incidence (Default vs Prepay)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plots_dir / "competing_risks_cif.png", dpi=100)
        plt.close(fig)
        print("  ✓ Competing risks CIF plot")

    except Exception as e:
        print(f"  ⚠ Competing risks skipped: {e}")

    # ── 4. Discrete-time hazard model ─────────────────────────────────────
    # Use all panel rows as person-period records
    try:
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.metrics import roc_auc_score

        pp = panel.copy()
        pp["event"] = (pp["current_status"].isin(["Default","Prepaid"])).astype(int)
        pp = pp[~pp["current_status"].isin(["Default","Prepaid"])].copy()  # exclude already-exited

        pp_feats = ["loan_age_months","days_past_due","modification_flag","month_index"]
        pp_feats = [f for f in pp_feats if f in pp.columns]
        pp_clean = pp[pp_feats + ["event"]].dropna()

        split = int(len(pp_clean) * 0.8)
        X_pp  = pp_clean[pp_feats].values
        y_pp  = pp_clean["event"].values

        dth_model = LR(class_weight="balanced", max_iter=500, C=0.1, random_state=42)
        dth_model.fit(X_pp[:split], y_pp[:split])
        prob_pp = dth_model.predict_proba(X_pp[split:])[:, 1]

        auc_dth = roc_auc_score(y_pp[split:], prob_pp)
        outcomes["discrete_time_hazard_auc"] = round(auc_dth, 4)

        # Compare to naive
        naive_prob = np.full(len(y_pp[split:]), y_pp[:split].mean())
        naive_auc  = roc_auc_score(y_pp[split:], naive_prob)
        outcomes["naive_constant_hazard_auc"] = round(naive_auc, 4)

        print(f"  ✓ Discrete-time hazard AUC: {auc_dth:.4f} (vs. naive {naive_auc:.4f})")
        joblib.dump(dth_model, ROOT / cfg["PATHS"]["models"] / "discrete_hazard.pkl")

    except Exception as e:
        print(f"  ⚠ Discrete-time hazard failed: {e}")

    with open(proc_dir / "survival_metrics.json", "w") as f:
        json.dump(outcomes, f, indent=2)

    return outcomes
