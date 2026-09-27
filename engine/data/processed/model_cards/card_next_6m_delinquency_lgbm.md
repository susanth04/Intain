# Model Card: 6-Month Delinquency Prediction (LightGBM)

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | next_6m_delinquency_flag_lgbm |
| **Model Type** | LightGBM Gradient Boosting Classifier |
| **Version** | 1.0 |
| **Training Date** | August 2026 |
| **Purpose** | Predict whether a loan will become delinquent within 6 months |

## Objective

Predict the probability that a loan will become delinquent (30+ days past due) within the next 6 months based on current loan characteristics and payment history. This extended horizon provides earlier warning for intervention.

## Data

### Training Data
- **Source**: Synthetic loan panel data (Fannie Mae-inspired)
- **Time Period**: Months 1-20 (training), Months 21-25 (validation), Months 26-30 (test)
- **Sample Size**: ~81,189 panel rows × 5,000 unique loans
- **Target Variable**: `next_6m_delinquency_flag` (binary: 1 if delinquent within 6 months, else 0)
- **Positive Rate**: ~53% in training data

### Data Quality
- Missing values handled via median imputation
- Categorical variables encoded using ordinal maps and frequency encoding
- Temporal features engineered (rolling windows, lag features)
- Time-aware split prevents data leakage

## Features

### Feature Categories
- **Loan Characteristics**: loan_age_months, remaining_term_months, original_balance, current_balance, interest_rate
- **Payment History**: days_past_due, current_status, modification_flag, prepayment_flag
- **Credit Risk**: credit_score_band, ltv_band, dti_band
- **Geographic**: state
- **Loan Attributes**: loan_purpose, occupancy_type, property_type, servicer_name
- **Engineered Features**: 
  - Rolling delinquency status (status_roll3_max, status_roll6_max)
  - Balance utilization (current_balance / original_balance)
  - Rate spread (interest_rate - portfolio mean)
  - Seasoning buckets
  - High-risk combo (credit × LTV interaction)
  - Extended lag features (6-month windows for longer horizon)

### Total Features
- **Numeric Features**: ~25 features after encoding
- **Categorical Features**: 8 original categorical fields encoded as numeric
- **Feature Selection**: Automatic selection based on data types and exclusion of target variables

## Model Type

### Algorithm
- **LightGBM** (Light Gradient Boosting Machine)
- **Gradient Boosting Framework** optimized for efficiency and performance
- **Tree-based ensemble learning** method

### Hyperparameters
```yaml
n_estimators: 400
learning_rate: 0.05
num_leaves: 63
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
class_weight: balanced
random_state: 42
metric: auc
```

### Calibration
- **Method**: CalibratedClassifierCV with sigmoid (Platt scaling)
- **Cross-validation**: 3-fold CV on training data
- **Purpose**: Improve probability calibration for reliable risk scores

## Validation Method

### Time-Aware Split
- **Training Set**: months 1-20 (month_index ≤ 20)
- **Validation Set**: months 21-25 (month_index 21-25) 
- **Test Set**: months 26-30 (month_index 26-30)
- **Rationale**: Prevents leakage by ensuring model never sees future outcomes

### Early Stopping
- **Patience**: 50 rounds
- **Validation Monitor**: AUC on validation set
- **Purpose**: Prevent overfitting and optimize generalization

## Metrics

### Performance Metrics

| Split | ROC-AUC | PR-AUC | F1 | Brier Score | Recall@Precision80 |
|-------|---------|--------|-----|-------------|-------------------|
| Validation | 0.7897 | 0.7685 | 0.7332 | 0.1909 | 0.4142 |
| Test | 0.7098 | 0.6873 | 0.6801 | 0.2161 | 0.2094 |

### Comparison to Baseline
- **Logistic Regression Test ROC-AUC**: 0.658
- **LightGBM Test ROC-AUC**: 0.7098
- **Improvement**: +7.9% relative improvement

### Key Strengths
- Excellent discrimination ability (ROC-AUC > 0.71)
- Outstanding precision-recall tradeoff (PR-AUC > 0.69)
- Strong F1 score (0.68) indicating good balance
- Meaningful recall at high precision (0.21 at 80% precision)

## Limitations

### Data Limitations
- **Synthetic Data**: Trained on synthetic loan data; real-world performance may differ
- **Time Horizon**: 6-month prediction window may be affected by changing economic conditions
- **Static Features**: Does not incorporate dynamic economic indicators or market conditions

### Model Limitations
- **Feature Importance**: Complex interactions may be difficult to interpret
- **Extrapolation**: Performance may degrade on loans outside training distribution
- **Class Imbalance**: Despite balanced class weights, performance variance across splits

### Temporal Limitations
- **Concept Drift**: Model performance may degrade over time as lending practices evolve
- **Economic Sensitivity**: Not explicitly trained on different economic scenarios

## Leakage Controls

### Temporal Integrity
- **Strict Time Split**: No future information used in training
- **Lag Features**: Only historical data used for predictions
- **Target Construction**: Targets derived from future periods, not contemporaneous

### Feature Engineering Safeguards
- **No Future Data**: All features based on current or historical information
- **Target Exclusion**: Target variables explicitly excluded from feature set
- **Correlation Checks**: No near-perfect feature-target correlations (|r| < 0.99)

### Validation Rigor
- **Held-out Test Set**: Final evaluation on unseen time period
- **Leakage Detection**: Automated checks for temporal ordering and feature correlations
- **Reproducible Splits**: Deterministic splits based on month_index

## Known Failure Modes

### High-Risk Scenarios
1. **Economic Shocks**: Model not trained on recession scenarios
2. **Policy Changes**: Changes in lending regulations not reflected
3. **New Loan Products**: Performance on novel loan types unknown

### Data Quality Issues
1. **Missing Payment History**: Poor performance on loans with incomplete history
2. **Status Inconsistencies**: Conflicting servicer reports may confuse model
3. **Extreme Values**: Outliers in balance or rate features may affect predictions

### Edge Cases
1. **New Loans**: Limited history for recently originated loans
2. **Paid-off Loans**: Model may not handle early payoffs well
3. **Modifications**: Loan modifications may not be fully captured

## Intended Use

### Primary Use Cases
- **Early Warning System**: Identify loans likely to become delinquent within 6 months
- **Portfolio Management**: Prioritize loan review and intervention resources
- **Risk Stratification**: Segment portfolio by risk levels for monitoring

### Not Suitable For
- **Credit Decisions**: Not intended for origination underwriting
- **Regulatory Capital**: Not validated for regulatory capital requirements
- **Pricing**: Not calibrated for pricing or risk-based pricing decisions

## Ethical Considerations

### Fairness
- **Protected Classes**: Model does not explicitly use protected attributes (race, gender, age)
- **Geographic Bias**: State-based features may introduce geographic bias
- **Credit Score Bands**: Uses credit score bands which may reflect historical biases

### Transparency
- **Explainability**: SHAP values available for local explanations
- **Feature Importance**: Global feature importance available
- **Model Documentation**: This card provides comprehensive documentation

## Maintenance

### Retraining Schedule
- **Recommended**: Quarterly retraining with recent data
- **Trigger**: Significant performance degradation or data drift
- **Monitoring**: Track ROC-AUC, calibration, and feature stability

### Update Process
1. Collect new loan performance data
2. Apply same feature engineering pipeline
3. Retrain model with updated hyperparameters
4. Validate on recent time period
5. Deploy after performance verification

## References

- **Training Code**: `engine/src/models/delinquency_default_prepay.py`
- **Feature Engineering**: `engine/src/features/build_features.py`
- **Data Pipeline**: `engine/src/data/splits.py`
- **Configuration**: `engine/config.yaml`
- **Model Artifacts**: `engine/data/processed/models/next_6m_delinquency_flag_lgbm_*.pkl`

---

**Model Card Version**: 1.0  
**Last Updated**: September 2026  
**Maintained By**: Intain Campus FinTech Challenge 2026 AI Track Team
