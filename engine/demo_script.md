# Demo Script — Loan Performance Intelligence Engine
**Target duration: 5 minutes**

---

### 1. Dataset and Targets (0:00 - 0:30)
- **Visual:** Show `config.yaml` and `data/raw/loan_monthly_performance_train.csv`
- **Talk track:** "Our engine begins by processing the loan panel data. Since we're demonstrating locally, we've synthesized a compliant dataset of 5,000 loans over 30 months using a Markov-chain state machine. We extract five targets: 3M/6M delinquency, 12M default/prepay, and next state."

### 2. Data Profiling Report (0:30 - 1:00)
- **Visual:** Open `reports/data_intelligence_report.md`
- **Talk track:** "Before modeling, our Data Intelligence module scans the panel. It automatically identifies missingness patterns—for example, it detected an MNAR pattern where `document_status` is missing more often for defaulted loans. We also compute a PSI drift score between train and test sets to flag distribution shifts."

### 3. Top Data-Quality Issues (1:00 - 1:20)
- **Visual:** Scroll to the "Validation Rule Violations" section in the data intelligence report.
- **Talk track:** "We apply deterministic checks from `validation_rules.json`. Here we see flags for balances exceeding origination amounts and invalid date orderings. These flow into a composite DQ score for every record."

### 4. Feature Engineering (1:20 - 1:40)
- **Visual:** Briefly show `src/features/build_features.py`
- **Talk track:** "We enrich the raw data with rolling delinquency histories, loan seasoning buckets, and rate-spread proxies. Categoricals are encoded appropriately. Crucially, we drop all target-derived columns before the split."

### 5. Time-Aware Split (1:40 - 2:00)
- **Visual:** Show terminal output of `run_all.py` highlighting the "TIME-AWARE SPLIT" section.
- **Talk track:** "As required, we split strictly by time—not randomly by row. We fit on months 1–20, validate on 21–25, and test on 26–30. Our automated leakage check confirms no feature has a perfect correlation with any target, preventing look-ahead bias."

### 6. Baseline & Improved Models (2:00 - 2:40)
- **Visual:** Open `reports/model_metrics_report.md` and `reports/plots/next_12m_default_flag_calibration.png`
- **Talk track:** "We benchmarked a LightGBM model against a balanced Logistic Regression baseline. For the 12-month default target, LightGBM significantly outperformed the baseline on PR-AUC. We also applied isotonic calibration to ensure our predicted probabilities are reliable, as seen in this calibration diagram."

### 7. Survival/Transition Model (2:40 - 3:10)
- **Visual:** Open `reports/plots/competing_risks_cif.png` and `reports/plots/km_survival_credit_band.png`
- **Talk track:** "Time-to-event is modeled using a competing-risks Aalen-Johansen estimator, treating default and prepayment as competing exits. Our Kaplan-Meier curves clearly separate survival probabilities by credit band, providing a hazard baseline that outperforms naive estimates."

### 8. Anomaly Examples (3:10 - 3:40)
- **Visual:** Open `reports/anomaly_examples.md`
- **Talk track:** "Our exception detector fuses rule violations with Isolation Forest outliers. Here are reviewer-ready examples. For instance, this loan was flagged due to both a validation rule failure and a high statistical anomaly score, with the top drivers clearly listed."

### 9. Scenario Simulation (3:40 - 4:10)
- **Visual:** Open `reports/scenario_report.md` and `reports/plots/scenario_comparison.png`
- **Talk track:** "We subjected the portfolio to base, adverse credit, and high prepayment stress scenarios. The adverse scenario, driven by a downward shift in credit scores, nearly doubles the 12-month default rate, while the high prepayment scenario spikes early exits due to modeled rate drops."

### 10. Local Explanation (4:10 - 4:30)
- **Visual:** Open `reports/plots/shap_local_next_12m_default_flag_TP.png`
- **Talk track:** "For explainability, we generate global and local SHAP values. This waterfall plot shows a True Positive default prediction: you can see exactly how the `days_past_due` lag and `rate_spread` pushed the model's risk score higher for this specific borrower."

### 11. LLM-Assisted Reviewer Copilot & Error Logs (4:30 - 4:50)
- **Visual:** Open `reports/copilot_demo.md`
- **Talk track:** "Our LLM copilot generates plain-language summaries grounded entirely via TF-IDF RAG over our data dictionary and model outputs. Every note is strictly labeled as AI-generated. We’ve also documented three explicit failure modes—like the LLM hallucinating dynamic credit scores—to demonstrate our validation process and guardrails."

### 12. Final Deliverables (4:50 - 5:00)
- **Visual:** Show `ai_dev_log/AI_DEVELOPMENT_LOG.md` and `submission/submission.csv`
- **Talk track:** "The entire pipeline is wrapped in a single reproducible script. Our `submission.csv` exactly matches the required schema, and our AI Development Log transparently documents our agentic coding process, including rejected AI suggestions. Thank you."
