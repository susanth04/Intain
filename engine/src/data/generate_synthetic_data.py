"""
generate_synthetic_data.py
==========================
Generates a realistic synthetic loan-performance dataset for the
Loan Performance Intelligence Engine (Intain Campus FinTech Challenge 2026).

Outputs (all to data/raw/):
  loan_static_attributes.csv
  loan_monthly_performance_train.csv
  loan_monthly_performance_test.csv (held-out period)
  servicer_updates.csv
  data_dictionary.md
  validation_rules.json
  macro_scenarios.csv
  submission_template.csv

Design principles:
  - Markov-chain state machine per loan with latent risk factors
  - Forward-looking targets derived from simulated future trajectory
  - Realistic missingness (MCAR + MNAR), outliers, conflicts
  - All randomness seeded via config for reproducibility
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

def load_cfg(cfg_path=None):
    path = cfg_path or ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

# ── State machine constants ─────────────────────────────────────────────────
STATES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"]
TERMINAL = {"Default", "Prepaid", "Closed"}

CREDIT_BANDS   = ["Excellent", "Good", "Fair", "Poor"]
LTV_BANDS      = ["<=60", "60-75", "75-90", ">90"]
DTI_BANDS      = ["<=28", "28-36", "36-43", ">43"]
STATES_CODE    = ["Current", "30DPD", "60DPD", "90DPD"]  # non-terminal
PURPOSES       = ["Purchase", "Refinance", "CashOut"]
OCCUPANCY      = ["PrimaryResidence", "SecondHome", "Investment"]
PROP_TYPES     = ["SingleFamily", "Condo", "MultiUnit", "Townhouse"]
SERVICERS      = ["ServicerA", "ServicerB", "ServicerC", "ServicerD"]
DOC_STATUS     = ["Complete", "Incomplete", "Pending", "Waived"]
EXCEPTION_TYPES = [
    "BalanceAnomaly", "StatusConflict", "DateViolation",
    "DocumentGap", "DuplicateRecord", "StaleUpdate"
]

# ── Latent risk scoring ─────────────────────────────────────────────────────
CREDIT_RISK  = {"Excellent": 0.02, "Good": 0.06, "Fair": 0.12, "Poor": 0.22}
LTV_RISK     = {"<=60": 0.01, "60-75": 0.04, "75-90": 0.09, ">90": 0.16}
DTI_RISK     = {"<=28": 0.01, "28-36": 0.04, "36-43": 0.09, ">43": 0.15}

def latent_risk(credit, ltv, dti, rate_spread=0.0):
    """Composite risk score [0,1] driving state transitions."""
    base = CREDIT_RISK[credit] + LTV_RISK[ltv] + DTI_RISK[dti]
    return min(base + rate_spread * 0.02, 0.90)

# ── Base transition matrix factory ─────────────────────────────────────────
def transition_matrix(risk: float) -> dict:
    """
    Returns a dict of {from_state: {to_state: prob}} conditional on risk.
    Absorbing states return to themselves.
    """
    r = risk
    tm = {
        "Current": {
            "Current": max(0.92 - r, 0.50),
            "30DPD":   r * 0.5,
            "60DPD":   r * 0.1,
            "Prepaid": max(0.04 - r * 0.03, 0.005),
            "Closed":  0.005,
        },
        "30DPD": {
            "Current": max(0.60 - r * 0.5, 0.10),
            "30DPD":   0.15,
            "60DPD":   r * 0.8,
            "Default": r * 0.1,
            "Prepaid": 0.02,
        },
        "60DPD": {
            "Current": max(0.25 - r * 0.3, 0.03),
            "30DPD":   0.10,
            "60DPD":   0.15,
            "90DPD":   r * 0.6,
            "Default": r * 0.3,
        },
        "90DPD": {
            "90DPD":   0.20,
            "Default": r * 0.8,
            "Closed":  0.05,
            "Current": 0.05,
        },
        "Default": {"Default": 1.0},
        "Prepaid": {"Prepaid": 1.0},
        "Closed":  {"Closed":  1.0},
    }
    # Normalise rows
    for state, row in tm.items():
        total = sum(row.values())
        tm[state] = {k: v / total for k, v in row.items()}
    return tm


def sample_next_state(current: str, tm: dict, rng: np.random.Generator) -> str:
    row = tm[current]
    states = list(row.keys())
    probs  = list(row.values())
    return rng.choice(states, p=probs)


def dpd_from_state(state: str) -> int:
    mapping = {"Current": 0, "30DPD": 35, "60DPD": 65, "90DPD": 95,
               "Default": 120, "Prepaid": 0, "Closed": 0}
    return mapping.get(state, 0)


# ── Origination date helpers ────────────────────────────────────────────────
ORIGIN_START = datetime(2018, 1, 1)

def month_offset(dt: datetime, offset: int) -> str:
    m = dt.month + offset
    y = dt.year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return f"{y:04d}-{m:02d}"

def add_months(dt: datetime, n: int) -> datetime:
    m = dt.month + n
    y = dt.year + (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return dt.replace(year=y, month=m, day=1)


# ── Main generator ──────────────────────────────────────────────────────────
def generate(cfg: dict, out_dir: Path):
    rng_np  = np.random.default_rng(cfg["RANDOM_SEED"])
    rng_py  = random.Random(cfg["RANDOM_SEED"])
    N       = cfg["N_LOANS"]
    M       = cfg["N_MONTHS"]
    SPLIT   = cfg["SPLIT"]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] Synthesising {N:,} loans × up to {M} months …")

    # ── 1. Static attributes ──────────────────────────────────────────────
    credit_bands = rng_np.choice(CREDIT_BANDS, size=N, p=[0.25,0.35,0.25,0.15])
    ltv_bands    = rng_np.choice(LTV_BANDS,    size=N, p=[0.20,0.35,0.30,0.15])
    dti_bands    = rng_np.choice(DTI_BANDS,    size=N, p=[0.25,0.35,0.25,0.15])

    orig_balances = rng_np.lognormal(mean=12.2, sigma=0.5, size=N).clip(50_000, 2_000_000).astype(int)
    orig_dates    = [ORIGIN_START + timedelta(days=int(d))
                     for d in rng_np.integers(0, 365*5, size=N)]
    vintages      = [str(d.year) for d in orig_dates]

    static = pd.DataFrame({
        "loan_id":           [f"LN{i:07d}" for i in range(1, N+1)],
        "original_balance":  orig_balances,
        "credit_score_band": credit_bands,
        "ltv_band":          ltv_bands,
        "dti_band":          dti_bands,
        "state":             rng_np.choice(["CA","TX","FL","NY","IL","WA","GA","OH","AZ","NV"], size=N),
        "loan_purpose":      rng_np.choice(PURPOSES,  size=N, p=[0.50,0.35,0.15]),
        "occupancy_type":    rng_np.choice(OCCUPANCY, size=N, p=[0.75,0.12,0.13]),
        "property_type":     rng_np.choice(PROP_TYPES,size=N, p=[0.60,0.20,0.10,0.10]),
        "vintage":           vintages,
    })
    static.to_csv(out_dir / "loan_static_attributes.csv", index=False)
    print(f"  ✓ loan_static_attributes.csv  ({len(static):,} rows)")

    # ── 2. Simulate monthly trajectories ─────────────────────────────────
    panel_rows = []
    servicer_rows = []

    for i, row_s in static.iterrows():
        loan_id  = row_s["loan_id"]
        credit   = row_s["credit_score_band"]
        ltv      = row_s["ltv_band"]
        dti      = row_s["dti_band"]
        orig_bal = row_s["original_balance"]
        orig_dt  = orig_dates[i]

        # interest rate: base + spread linked to credit
        rate_spread = {"Excellent": 0.0, "Good": 0.3, "Fair": 0.8, "Poor": 1.5}[credit]
        ir = round(rng_np.uniform(3.5, 5.0) + rate_spread, 2)

        risk = latent_risk(credit, ltv, dti, rate_spread)
        tm   = transition_matrix(risk)

        state   = "Current"
        bal     = float(orig_bal)
        monthly_pay = bal * (ir/100/12) / (1 - (1+ir/100/12)**-360)
        principal_frac = 0.3  # fraction of payment that's principal (simplified)

        # Store full trajectory first (for forward label derivation)
        trajectory = []
        for m in range(1, M + 1):
            reporting_dt = add_months(orig_dt, m)
            reporting_month = f"{reporting_dt.year:04d}-{reporting_dt.month:02d}"
            loan_age = m

            # Inject macro shock at month 15-18 (adverse period)
            macro_shock = 1.5 if 15 <= m <= 18 else 1.0

            # Absorbing check
            if state not in TERMINAL:
                effective_tm = transition_matrix(min(risk * macro_shock, 0.90))
                next_s = sample_next_state(state, effective_tm, rng_np)
            else:
                next_s = state

            # Balance evolution
            if state == "Current":
                bal = max(bal - monthly_pay * principal_frac, 0)
            elif state == "Prepaid":
                bal = 0.0

            trajectory.append({
                "loan_id":            loan_id,
                "month_index":        m,
                "reporting_month":    reporting_month,
                "origination_month":  f"{orig_dt.year:04d}-{orig_dt.month:02d}",
                "loan_age_months":    loan_age,
                "remaining_term_months": max(360 - loan_age, 0),
                "original_balance":   orig_bal,
                "current_balance":    round(bal, 2),
                "interest_rate":      ir,
                "credit_score_band":  credit,
                "ltv_band":           ltv,
                "dti_band":           dti,
                "state":              row_s["state"],
                "loan_purpose":       row_s["loan_purpose"],
                "occupancy_type":     row_s["occupancy_type"],
                "property_type":      row_s["property_type"],
                "servicer_name":      rng_py.choice(SERVICERS),
                "current_status":     state,
                "days_past_due":      dpd_from_state(state),
                "modification_flag":  int(state in ["60DPD","90DPD"] and rng_np.random() < 0.10),
                "prepayment_flag":    int(state == "Prepaid"),
                "default_flag":       int(state == "Default"),
                "loss_severity_band": "High" if state == "Default" else (
                                      "Medium" if state in ["60DPD","90DPD"] else "Low"),
                "last_updated_at":    reporting_month + "-01",
                "source_system":      "PRIMARY",
                "document_status":    rng_py.choices(DOC_STATUS, weights=[60,15,15,10], k=1)[0],
                "_next_state_actual": next_s,
            })
            if state in TERMINAL and m > 3:
                break  # loan exits
            state = next_s

        # Derive forward-looking targets from trajectory
        traj_df = pd.DataFrame(trajectory)
        n_traj  = len(traj_df)

        for j, rec in enumerate(trajectory):
            # next_3m_delinquency_flag
            future_3  = [trajectory[k]["current_status"] for k in range(j+1, min(j+4,  n_traj))]
            future_6  = [trajectory[k]["current_status"] for k in range(j+1, min(j+7,  n_traj))]
            future_12 = [trajectory[k]["current_status"] for k in range(j+1, min(j+13, n_traj))]

            rec["next_3m_delinquency_flag"]  = int(any(s in ["30DPD","60DPD","90DPD","Default"] for s in future_3))
            rec["next_6m_delinquency_flag"]  = int(any(s in ["30DPD","60DPD","90DPD","Default"] for s in future_6))
            rec["next_12m_default_flag"]     = int("Default" in future_12)
            rec["next_12m_prepayment_flag"]  = int("Prepaid" in future_12)
            rec["next_state"]                = rec.pop("_next_state_actual")

            # Exception / anomaly labels
            is_exception = 0
            exc_type     = ""
            if rec["current_balance"] > rec["original_balance"] * 1.05:
                is_exception, exc_type = 1, "BalanceAnomaly"
            elif rec["days_past_due"] > 0 and rec["current_status"] == "Current":
                is_exception, exc_type = 1, "StatusConflict"
            elif rec["document_status"] in ["Incomplete","Pending"] and rec["default_flag"] == 1:
                is_exception, exc_type = 1, "DocumentGap"
            rec["exception_required"] = is_exception
            rec["exception_type"]     = exc_type

            panel_rows.append(rec)

        # ── Servicer conflict rows ────────────────────────────────────────
        conflict_months = rng_py.sample(range(n_traj), k=max(1, int(n_traj * 0.05)))
        for j in conflict_months:
            orig_rec = trajectory[j]
            # Deliberately conflict status or timestamp
            conflict_status = rng_py.choice([s for s in STATES if s != orig_rec["current_status"]])
            servicer_rows.append({
                "loan_id":         loan_id,
                "month_index":     orig_rec["month_index"],
                "reporting_month": orig_rec["reporting_month"],
                "current_status":  conflict_status,
                "days_past_due":   dpd_from_state(conflict_status),
                "last_updated_at": orig_rec["last_updated_at"][:-3],  # stale by 1 month
                "servicer_name":   rng_py.choice(SERVICERS),
                "source_system":   "SERVICER_FEED",
                "conflict_flag":   1,
            })

    panel = pd.DataFrame(panel_rows)
    print(f"  ✓ Panel generated: {len(panel):,} rows across {N:,} loans")

    # ── 3. Inject messiness ───────────────────────────────────────────────
    # MCAR missingness on several columns (~3-8%)
    for col in ["credit_score_band","ltv_band","document_status","loss_severity_band"]:
        mask = rng_np.random(size=len(panel)) < rng_np.uniform(0.03, 0.08)
        panel.loc[mask, col] = np.nan

    # MNAR: document_status missing more often when default_flag=1
    mnar_mask = (panel["default_flag"] == 1) & (rng_np.random(size=len(panel)) < 0.25)
    panel.loc[mnar_mask, "document_status"] = np.nan

    # Outlier balances (0.5% of rows)
    outlier_mask = rng_np.random(size=len(panel)) < 0.005
    panel.loc[outlier_mask, "current_balance"] *= rng_np.uniform(5, 20, size=outlier_mask.sum())

    # Implausible date orderings (0.3% of rows)
    bad_date_mask = rng_np.random(size=len(panel)) < 0.003
    panel.loc[bad_date_mask, "origination_month"] = "2030-01"

    # Inject ~1% exact duplicate rows
    n_dups = max(1, int(len(panel) * 0.01))
    dup_idx = rng_np.choice(panel.index, size=n_dups, replace=False)
    panel = pd.concat([panel, panel.loc[dup_idx]], ignore_index=True)
    panel = panel.sample(frac=1, random_state=cfg["RANDOM_SEED"]).reset_index(drop=True)

    # ── 4. Train / test split (by month_index cutoff) ────────────────────
    train_cutoff = SPLIT["VAL_CUTOFF"]
    train = panel[panel["month_index"] <= train_cutoff].copy()
    test  = panel[panel["month_index"] >  train_cutoff].copy()

    train.to_csv(out_dir / "loan_monthly_performance_train.csv", index=False)
    test.to_csv( out_dir / "loan_monthly_performance_test.csv",  index=False)
    print(f"  ✓ train: {len(train):,} rows | test: {len(test):,} rows")

    # ── 5. Servicer updates ───────────────────────────────────────────────
    svc = pd.DataFrame(servicer_rows)
    svc.to_csv(out_dir / "servicer_updates.csv", index=False)
    print(f"  ✓ servicer_updates.csv  ({len(svc):,} rows, ~5% conflict rate)")

    # ── 6. Macro scenarios ────────────────────────────────────────────────
    months = list(range(1, M+1))
    scenarios = []
    for m in months:
        for scenario, (def_mult, prep_mult, rate_shift, hpi_shock) in {
            "base":             (1.0,  1.0,  0.0,  0.0),
            "adverse_credit":   (2.0,  0.8, +0.5, -0.10),
            "high_prepayment":  (0.8,  2.5, -0.75, +0.05),
        }.items():
            scenarios.append({
                "month_index":               m,
                "scenario":                  scenario,
                "default_hazard_multiplier": def_mult,
                "prepay_propensity_mult":    prep_mult,
                "rate_shift_pp":             rate_shift,
                "hpi_shock_pct":             hpi_shock,
                "unemployment_shock":        0.02 if scenario == "adverse_credit" else 0.0,
            })
    pd.DataFrame(scenarios).to_csv(out_dir / "macro_scenarios.csv", index=False)
    print(f"  ✓ macro_scenarios.csv")

    # ── 7. Submission template ────────────────────────────────────────────
    # Use test loans as the template
    template_loans = test[["loan_id","reporting_month"]].drop_duplicates().head(min(5000, len(test)))
    template_loans = template_loans.rename(columns={"reporting_month": "as_of_month"})
    for col in ["prob_delinquency_3m","prob_delinquency_6m","prob_default_12m",
                "prob_prepayment_12m"]:
        template_loans[col] = 0.0
    template_loans["predicted_next_state"] = ""
    template_loans["exception_flag"]       = 0
    template_loans["exception_type"]       = ""
    template_loans["anomaly_score"]         = 0.0
    template_loans["top_drivers"]          = ""
    template_loans["recommended_action"]   = ""
    template_loans["confidence"]           = 0.0
    template_loans.to_csv(out_dir / "submission_template.csv", index=False)
    print(f"  ✓ submission_template.csv  ({len(template_loans):,} rows)")

    # ── 8. Data dictionary ────────────────────────────────────────────────
    dd = """# Data Dictionary — Loan Performance Intelligence Engine

## loan_static_attributes.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Unique loan identifier (format: LN0000001) |
| original_balance | integer | Original loan balance in USD at origination |
| credit_score_band | categorical | Borrower credit quality: Excellent/Good/Fair/Poor |
| ltv_band | categorical | Loan-to-value ratio band: <=60/60-75/75-90/>90 |
| dti_band | categorical | Debt-to-income ratio band: <=28/28-36/36-43/>43 |
| state | categorical | US state abbreviation |
| loan_purpose | categorical | Purchase/Refinance/CashOut |
| occupancy_type | categorical | PrimaryResidence/SecondHome/Investment |
| property_type | categorical | SingleFamily/Condo/MultiUnit/Townhouse |
| vintage | string | Year of origination |

## loan_monthly_performance_train.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Foreign key to loan_static_attributes |
| month_index | integer | 1-based month sequence since origination |
| reporting_month | string | YYYY-MM reporting period |
| origination_month | string | YYYY-MM origination period |
| loan_age_months | integer | Months since origination |
| remaining_term_months | integer | Months remaining on 360-month term |
| original_balance | integer | Original balance (repeated for join convenience) |
| current_balance | float | Ending balance this month |
| interest_rate | float | Annual interest rate (%) |
| credit_score_band | categorical | May change if modification occurs |
| ltv_band | categorical | May change with balance / HPI changes |
| dti_band | categorical | Borrower DTI band |
| state | categorical | Loan state |
| loan_purpose | categorical | Purchase/Refinance/CashOut |
| occupancy_type | categorical | Occupancy type |
| property_type | categorical | Property type |
| servicer_name | categorical | Loan servicer |
| current_status | categorical | Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed |
| days_past_due | integer | Days past due (0 if current) |
| modification_flag | binary | 1 if loan was modified this period |
| prepayment_flag | binary | 1 if prepaid this period |
| default_flag | binary | 1 if defaulted this period |
| loss_severity_band | categorical | Low/Medium/High expected loss severity |
| last_updated_at | date | Date of last record update (YYYY-MM-DD) |
| source_system | string | Source system identifier |
| document_status | categorical | Complete/Incomplete/Pending/Waived |
| next_3m_delinquency_flag | binary | TARGET: delinquent within next 3 months |
| next_6m_delinquency_flag | binary | TARGET: delinquent within next 6 months |
| next_12m_default_flag | binary | TARGET: defaults within next 12 months |
| next_12m_prepayment_flag | binary | TARGET: prepays within next 12 months |
| next_state | categorical | TARGET: predicted state next month |
| exception_required | binary | TARGET: record requires exception review |
| exception_type | categorical | TARGET: type of exception |

## servicer_updates.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Loan identifier |
| month_index | integer | Month sequence |
| reporting_month | string | YYYY-MM |
| current_status | categorical | Status per servicer (may conflict with primary) |
| days_past_due | integer | DPD per servicer |
| last_updated_at | date | Servicer timestamp (may be stale) |
| servicer_name | categorical | Servicer name |
| source_system | string | SERVICER_FEED |
| conflict_flag | binary | 1 if this record conflicts with primary |

## macro_scenarios.csv
| Field | Type | Description |
|---|---|---|
| month_index | integer | Month index |
| scenario | categorical | base/adverse_credit/high_prepayment |
| default_hazard_multiplier | float | Multiplier on base default hazard rate |
| prepay_propensity_mult | float | Multiplier on base prepayment propensity |
| rate_shift_pp | float | Interest rate shift in percentage points |
| hpi_shock_pct | float | Home price index shock (fraction) |
| unemployment_shock | float | Unemployment rate shock (pp) |
"""
    (out_dir / "data_dictionary.md").write_text(dd)
    print(f"  ✓ data_dictionary.md")

    # ── 9. Validation rules ───────────────────────────────────────────────
    rules = {
        "rules": [
            {
                "rule_id": "VR001",
                "name": "balance_consistency",
                "description": "current_balance must not exceed original_balance by more than 5% absent modification",
                "sql_equivalent": "current_balance <= original_balance * 1.05 OR modification_flag = 1",
                "severity": "error"
            },
            {
                "rule_id": "VR002",
                "name": "date_validity",
                "description": "origination_month must be <= reporting_month",
                "sql_equivalent": "origination_month <= reporting_month",
                "severity": "error"
            },
            {
                "rule_id": "VR003",
                "name": "dpd_status_consistency",
                "description": "days_past_due=0 implies current_status=Current or Prepaid or Closed",
                "sql_equivalent": "NOT (days_past_due = 0 AND current_status NOT IN ('Current','Prepaid','Closed'))",
                "severity": "warning"
            },
            {
                "rule_id": "VR004",
                "name": "terminal_state_immutability",
                "description": "Loans in Default/Prepaid/Closed should not transition to other states",
                "sql_equivalent": "LAG(current_status) NOT IN ('Default','Prepaid','Closed') OR current_status = LAG(current_status)",
                "severity": "error"
            },
            {
                "rule_id": "VR005",
                "name": "document_completeness",
                "description": "Defaulted loans must not have document_status=Incomplete",
                "sql_equivalent": "NOT (default_flag = 1 AND document_status = 'Incomplete')",
                "severity": "warning"
            },
            {
                "rule_id": "VR006",
                "name": "balance_monotonic",
                "description": "current_balance should be non-increasing month-over-month absent modification",
                "sql_equivalent": "current_balance <= LAG(current_balance) OR modification_flag = 1 OR month_index = 1",
                "severity": "warning"
            }
        ]
    }
    with open(out_dir / "validation_rules.json", "w") as f:
        json.dump(rules, f, indent=2)
    print(f"  ✓ validation_rules.json")
    print(f"[generate] ✅ All synthetic files written to {out_dir}")
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    out_dir = ROOT / cfg["PATHS"]["raw_data"]
    generate(cfg, out_dir)
