# Model Card: Intain Loan Performance Engine

## 1. Model Objective
The model suite predicts the 12-month forward probability of **Default** and **Prepayment** for a synthetic panel of 5,000 mortgages over a 42-month period, as well as 3-month and 6-month delinquency risk. The models serve as the intelligence engine to recommend optimal servicing actions (e.g., forbearance vs. modification) before severe losses materialize.

## 2. Data and Features
**Dataset**: Synthetic mortgage performance panel (81,189 rows, 5,000 unique loans).
**Observation Window**: Up to 42 months per loan.
**Time-Aware Split**:
- **Train**: Months 1–20 (64,177 rows)
- **Validation**: Months 21–25 (6,096 rows)
- **Test**: Months 26–30 (4,424 rows)
- This strict out-of-time split prevents temporal leakage.

**Key Features (35 total)**:
- **Core**: `interest_rate`, `original_balance`, `current_balance`, `rate_spread`
- **Borrower Risk**: `credit_score_band_enc`, `dti_band_enc`, `ltv_band_enc`
- **Temporal/Rolling**: `current_balance_roll3_mean`, `status_roll3_max`, `days_past_due`

## 3. Model Architecture
- **Baseline**: Logistic Regression with `class_weight='balanced'` and L2 penalty (C=0.1).
- **Champion**: LightGBM Classifier with `class_weight='balanced'`, optimized with early stopping on the validation set, and calibrated via `CalibratedClassifierCV` (Sigmoid method).

## 4. Performance Metrics (Real Evaluation)
*Metrics reported on the held-out Test Set (months 26-30).*

### Target: `next_12m_default_flag` (Test Pos Rate: 14.6%)
- **Test ROC-AUC**: 0.7968
- **Test PR-AUC**: 0.5374
- **Top Drivers**: `interest_rate`, `original_balance`, `state_enc`, `current_balance`

### Target: `next_12m_prepayment_flag` (Test Pos Rate: 28.2%)
- **Test ROC-AUC**: 0.7359
- **Test PR-AUC**: 0.6195
- **Top Drivers**: `interest_rate`, `original_balance`, `state_enc`, `current_balance`

## 5. Leakage Controls
- **Target Derivation**: Labels are strictly derived from the `[j+1 : j+13]` forward-looking window in the underlying simulated trajectory.
- **Strict Split Validation**: Validation script checks that `train` max month (20) strictly precedes `validation` min month (21).
- **Leakage Scanning**: Computed Pearson correlation between all numeric features and binary targets. Any `|r| > 0.99` is flagged to prevent target leakage. No such leaks were detected.

## 6. Known Limitations and Failure Modes
- **Macro-Shock Stationarity**: The synthetic data incorporates an adverse macro-shock in months 15-18. Because this falls entirely within the training set, the model's calibration in the test period (months 26-30) may overestimate risk if the macro environment normalizes, leading to false positives.
- **Tree Split Failures on Extreme Imbalance**: In earlier iterations, when forward targets collapsed to 0% due to simulation truncation (a bug in generating terminal states after month 3), LightGBM failed to find splits. It requires robust baseline event rates to train effectively.
- **Explainability Constraints**: The current SHAP implementation for the multiclass `next_state` predictor fails due to multidimensional arrays (`(35, 7)`). Explainability is currently restricted to the binary classifiers.
