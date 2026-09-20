# Loan Performance Intelligence Engine (Prototype Phase)

## Overview
This repository contains a full end-to-end pipeline for generating synthetic mortgage panel data, profiling it for anomalies and drift, engineering temporal features, and training predictive models for delinquency, default, and prepayment risk. It also features isolation forest anomaly detection and a SHAP-based explainability layer.

## Instructions to Run

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Reproduce Pipeline and `submission.csv`
You can run the entire pipeline from scratch, which will:
1. Generate the 42-month synthetic data panel.
2. Build all features and perform time-aware splits.
3. Train the baseline Logistic Regression and LightGBM models.
4. Generate reliability diagrams, explainability SHAP plots, and DQ reports.
5. Compile the final `submission.csv` on the held-out test window (months 26-30).

Run the pipeline:
```bash
python run_all.py --reset
```
*Note: This takes about 2 minutes. The final submission is placed in `submission/submission.csv`.*

### 3. Validate Submission
To run the automated audit on the final submission file:
```bash
python validate_submission.py submission/submission.csv 5
```
You should see: `ALL CHECKS PASSED`.

## Project Structure
- `config.yaml`: Central hyperparameters and file paths.
- `src/`: Data generation, feature engineering, modeling, and anomaly detection code.
- `reports/`: Markdown reports on data intelligence, model performance, explainability, and scenarios.
- `ai_dev_log/`: Log of AI-assisted development (including honest notes on resolving the 'No Signal' Beta-rescaling detour).
- `notebooks/`: Contains `end_to_end_walkthrough.ipynb` demonstrating the pipeline interactively.
- `submission/`: Contains the final output CSV.
