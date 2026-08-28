# Loan Performance Intelligence Engine
**Intain Campus FinTech Challenge 2026 — AI Track**

This repository contains an end-to-end Loan Performance Intelligence Engine. It addresses all 8 tasks of the challenge prompt without relying on LLMs for prediction, enforcing strict time-aware data splits, and ensuring fully reproducible results.

---

## 🚀 Quickstart

**1. Install dependencies**
```bash
pip install pandas numpy scikit-learn lightgbm xgboost lifelines shap matplotlib seaborn mlxtend imbalanced-learn scipy pyyaml tqdm joblib openai
```

**2. Run the pipeline**
```bash
python run_all.py
```
This single entry point runs every phase in order:
1. Generates 5,000 synthetic loans (30 months) to mock the organizer data.
2. Profiles data, checking missingness and validation rules.
3. Builds features and splits strictly by time (month_index).
4. Trains binary models (LightGBM/LR) and a multiclass next-state model.
5. Trains survival and hazard models.
6. Detects anomalies via rules + Isolation Forest.
7. Simulates macro stress scenarios.
8. Generates global and local SHAP explanations.
9. Runs the TF-IDF RAG-grounded LLM reviewer copilot.
10. Builds the final `submission/submission.csv`.

*(You can use `--reset` to clear checkpoints and force a full run).*

---

## 📂 Repository Structure

- `config.yaml` — Central configuration for parameters and hyper-parameters.
- `run_all.py` — Single entry point pipeline.
- `src/` — All source code:
  - `data/` — Synthetic generator, loader, validator, and time-aware splits.
  - `profiling/` — Data intelligence (Task 1).
  - `features/` — Feature engineering (Task 2).
  - `models/` — Binary classification and multiclass next-state (Task 2).
  - `survival/` — Competing risks and hazard models (Task 3).
  - `anomaly/` — Hybrid rule+ML exception detection (Task 4).
  - `scenario/` — Macro stress simulation (Task 5).
  - `explain/` — Global/Local SHAP and uncertainty (Task 6).
  - `copilot/` — Grounded RAG LLM reviewer (Task 7).
- `reports/` — Generated markdown reports and `plots/`.
- `ai_dev_log/` — Honest log of AI tool usage and human review.
- `model_card.md` — Detailed model documentation and failure modes.
- `demo_script.md` — 5-minute presentation script.
- `notebooks/` — End-to-end walkthrough notebook.

---

## 🛡️ Critical Compliance Controls

1. **No LLM Prediction**: Every score in `submission.csv` is derived from trained statistical models (LightGBM, Logistic Regression, Isolation Forest).
2. **Leakage Prevention**: We split by time cutoff (e.g. train on months 1-20, val on 21-25, test on 26-30). `src/data/splits.py` prints an automated proof asserting that no target column is leaked into the feature set and no overlapping time periods exist.
3. **Grounded Copilot**: The Reviewer Copilot relies strictly on TF-IDF retrieval over the `data_dictionary.md` and model metric JSONs. 

*(See `reports/self_assessment.md` for a full grading matrix).*
