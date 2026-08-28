# AI Development Log
## Loan Performance Intelligence Engine — Intain Campus FinTech Challenge 2026

---

## Overview

This log documents how AI tools were used throughout the development of this solution.
It is honest and complete — including cases where AI suggestions were rejected or corrected.

**Primary AI Tool**: Antigravity (Google DeepMind) — agentic coding assistant  
**Session dates**: August 2026  
**Project**: Loan Performance Intelligence Engine (AI Track)

---

## Phase-by-Phase AI Usage

### Phase 0–1: Repo Scaffold & Config
**AI role**: Generated the complete directory structure, `config.yaml`, and `requirements.txt`.  
**Human review**: Reviewed and adjusted N_LOANS (bumped from 2000 to 5000 for richer validation output), modified the split cutoffs to better reflect a realistic 80/10/10 train/val/test regime.  
**AI code used**: ~90% — scaffolding is formulaic.

**Representative prompt used**:
> "Create a config.yaml for a loan performance ML pipeline. It needs to control N_LOANS, N_MONTHS, a random seed, file paths, train/val/test cutoffs, and model hyperparameter toggles for both LightGBM and logistic regression."

---

### Phase 2: Synthetic Data Generation
**AI role**: Drafted the Markov-chain state machine, transition matrix, and the full data generation loop including all 8 output files.  
**Human review**: The initial AI draft used `random.choice` with uniform probabilities for state transitions, which produced unrealistically low default rates (~0.5%). Manually injected credit-band-weighted risk scores and macro shock multipliers.  
**AI code accepted**: State machine structure, MCAR/MNAR injection, date helpers.  
**AI code rejected/modified**: Transition probability defaults — replaced with risk-adjusted version.

**Example of rejected AI output**:
> AI generated: `transition_matrix = {"Current": {"Current": 0.9, "30DPD": 0.05, ...}}`  
> Problem: This produced a uniform 5% monthly DPD rate regardless of credit quality, which isn't realistic. Replaced with the `latent_risk()` function that scales transitions by credit_band × ltv_band × dti_band.

---

### Phase 3: Data Intelligence & Profiling
**AI role**: Generated the full profiling module including PSI, KS-test drift detection, MNAR heuristic, and association rule mining.  
**Human review**: Verified the PSI implementation against the standard formula. Added the MNAR check for `document_status` vs. `default_flag` correlation — AI had only included MCAR analysis.  
**AI code used**: ~75%.

**Representative prompt**:
> "Write a Python function that computes Population Stability Index between two numeric arrays using 10 equal-frequency buckets. Handle the case where some buckets have zero counts."

---

### Phase 4: Feature Engineering
**AI role**: Generated lag feature computation, rolling statistics, ordinal encoding, and the imputer pipeline.  
**Human review**: AI initially computed lag features AFTER the train/val/test split, which would have introduced data leakage (val lags would be computed using train data). Fixed by computing lags on the FULL panel first, then splitting.  
**This was a critical bug catch by human review.**

**Example of rejected AI approach**:
> AI initially suggested: "Fit the OrdinalEncoder on training data, then apply to val/test."  
> Problem with the specific implementation: The encoder was being fitted inside the split loop, causing inconsistent category mappings between splits. Replaced with a fit-once-apply-everywhere pattern.

---

### Phase 5: Predictive Models
**AI role**: Generated the full LightGBM + Logistic Regression training loop, evaluation metrics, ROC/PR curves, and calibration wrapper.  
**Human review**: Verified the `recall_at_precision()` implementation (AI had an off-by-one in the precision-recall curve threshold indexing). Also added `early_stopping` callback which AI's initial draft omitted.  
**AI code used**: ~80%.

**Representative prompt**:
> "Write a Python function recall_at_precision(y_true, y_score, min_precision=0.80) that returns the maximum recall achievable while maintaining precision >= min_precision, using sklearn's precision_recall_curve."

---

### Phase 6: Survival Modeling
**AI role**: Generated the KM + Cox PH + competing risks + discrete-time hazard framework.  
**Human review**: AI initially used `SurvivalAnalysisMixin` from scikit-survival, which wasn't installed. Switched to `lifelines` which was already in requirements. AI was correct that both work; `lifelines` was simpler here.

---

### Phase 7: Anomaly Detection
**AI role**: Drafted the hybrid rules+Isolation Forest pipeline.  
**Human review**: AI's initial top-drivers implementation was O(N²) — for each flagged record it computed all pairwise deviations. Replaced with a vectorised approach using global mean comparison.  
**AI code rejected**: Quadratic top-drivers loop.

---

### Phase 8: Scenario Simulation
**AI role**: Generated the perturbation logic and segment breakdown.  
**Human review**: Initial scenario perturbations only modified `interest_rate` for the high_prepayment scenario. Human added `rate_spread` modification as well, since that's a derived feature the models actually use.

---

### Phase 9: Explainability
**AI role**: Generated SHAP integration, reliability diagrams, FP/FN analysis.  
**Human review**: AI used `shap.force_plot()` which requires JavaScript and doesn't save cleanly to PNG. Replaced with manual waterfall-style bar chart using matplotlib.

---

### Phase 10: LLM Copilot
**AI role**: Drafted the RAGGrounder (TF-IDF retrieval) and ReviewerCopilot classes.  
**Human review**: AI initially had the RAG context being passed AFTER the LLM was already called — completely defeating the purpose. Fixed the prompt construction to retrieve context first, then build the prompt.

---

## Approximate AI/Human Code Split

| Module | AI-generated | Human-modified |
|--------|-------------|----------------|
| Repo scaffold, config | 90% | 10% |
| Data generation | 60% | 40% |
| Data profiling | 75% | 25% |
| Feature engineering | 80% | 20% |
| Predictive models | 80% | 20% |
| Survival models | 70% | 30% |
| Anomaly detection | 75% | 25% |
| Scenario simulation | 75% | 25% |
| Explainability | 70% | 30% |
| LLM copilot | 65% | 35% |
| **Overall** | **~74%** | **~26%** |

---

## Human Review Process

Every AI-generated module was reviewed against:
1. **Schema correctness**: Does the code match the documented column names in the data dictionary?
2. **Leakage safety**: Are any future-derived columns leaking into training features?
3. **Mathematical correctness**: PSI formula, calibration, SHAP interpretation.
4. **Performance**: Would this scale to 250k loans? Quadratic loops were replaced.
5. **Output format**: Does `submission.csv` match the template exactly?

---

## Key Lessons Learned

1. **Leakage is subtle**: The lag feature bug (computing lags after splitting) would not have been caught by standard test failures. Manual inspection of the pipeline order is essential.
2. **SHAP + matplotlib**: `shap.force_plot()` produces interactive HTML; for static reports, use raw SHAP values with matplotlib bars.
3. **Lifelines vs. scikit-survival**: Both are valid; the choice depends on what's already installed. Document the dependency.
4. **LLM copilot context must be injected BEFORE the prompt**: Obvious in hindsight, but the AI draft got it backwards.
5. **Synthetic data quality matters**: Uniform transition probabilities produce unrealistic datasets. The Markov model needs risk stratification to generate meaningful class imbalance.

---

## Rejected AI Suggestions Summary

| Suggestion | Reason Rejected |
|---|---|
| Use `shap.force_plot()` for local explanations | Requires JS/HTML; can't save as PNG |
| Compute lag features after train/val split | Data leakage |
| Uniform transition probabilities for state machine | Produces unrealistic 5% DPD rate across all loans |
| O(N²) top-drivers computation | Performance — too slow for 250k rows |
| `SurvivalAnalysisMixin` from scikit-survival | Not installed; lifelines already available |
| Build RAG context inside LLM call | Defeats grounding purpose; context must be retrieved first |
