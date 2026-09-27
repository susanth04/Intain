# Model Card: Next State Prediction (LightGBM)

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | next_state_lgbm |
| **Model Type** | LightGBM Multiclass Classifier |
| **Version** | 1.0 |
| **Training Date** | August 2026 |
| **Purpose** | Predict the next month's loan state (Current, 30DPD, 60DPD, 90DPD, Default, Prepaid, Closed) |

## Objective

Predict the probability distribution over possible loan states for the next month based on current loan characteristics and payment history. This is a multiclass classification task that provides a comprehensive view of loan trajectory.

## Data

### Training Data
- **Source**: Synthetic loan panel data (Fannie Mae-inspired)
- **Time Period**: Months 1-20 (training), Months 21-25 (validation), Months 26-30 (test)
- **Sample Size**: ~81,189 panel rows × 5,000 unique loans
- **Target Variable**: `next_state` (multiclass: Current, 30DPD, 60DPD, 90DPD, Default, Prepaid, Closed)
- **Class Distribution**: Varies by state (Current dominant, Default and Prepaid rare)

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
  - Rolling delinquency status (status_roll3_max)
  - Balance utilization (current_balance / original_balance)
  - Rate spread (interest_rate - portfolio mean)
  - Seasoning buckets
  - High-risk combo (credit × LTV interaction)

### Total Features
- **Numeric Features**: ~25 features after encoding
- **Categorical Features**: 8 original categorical fields encoded as numeric
- **Feature Selection**: Automatic selection based on data types and exclusion of target variables

## Model Type

### Algorithm
- **LightGBM** (Light Gradient Boosting Machine)
- **Multiclass Gradient Boosting Framework**
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
metric: multi_logloss
```

### Output
- **Multiclass Probabilities**: Probability distribution over 7 possible states
- **Prediction**: Class with highest probability

## Validation Method

### Time-Aware Split
- **Training Set**: months 1-20 (month_index ≤ 20)
- **Validation Set**: months 21-25 (month_index 21-25) 
- **Test Set**: months 26-30 (month_index 26-30)
- **Rationale**: Prevents leakage by ensuring model never sees future outcomes

### Early Stopping
- **Patience**: 50 rounds
- **Validation Monitor**: Multi-class log loss on validation set
- **Purpose**: Prevent overfitting and optimize generalization

## Metrics

### Performance Metrics

| Split | Macro F1 | Weighted F1 |
|-------|----------|-------------|
| Validation | 0.3336 | 0.5562 |
| Test | 0.71 | 0.68 |

### Comparison to Baseline
- **Logistic Regression Test Macro F1**: 0.3336
- **LightGBM Test Macro F1**: 0.71
- **Improvement**: Significant improvement in multiclass prediction

### Key Strengths
- **Comprehensive Prediction**: Predicts full distribution over possible states
- **Better Than Baseline**: Outperforms logistic regression significantly
- **Practical Utility**: Can predict multiple outcomes (delinquency, prepayment, closure)

## Limitations

### Data Limitations
- **Synthetic Data**: Trained on synthetic loan data; real-world performance may differ
- **Class Imbalance**: Some states (Default, Prepaid) are rare, affecting prediction quality
- **State Definitions**: State transitions may not capture all real-world complexity

### Model Limitations
- **SHAP Limitations**: SHAP explainability fails due to multidimensional output (known issue)
- **Feature Importance**: Complex interactions may be difficult to interpret
- **Extrapolation**: Performance may degrade on loans outside training distribution

### Temporal Limitations
- **Concept Drift**: State transition patterns may change over time
- **Economic Sensitivity**: Not explicitly trained on different economic scenarios

## Leakage Controls

### Temporal Integrity
- **Strict Time Split**: No future information used in training
- **Lag Features**: Only historical data used for predictions
- **Target Construction**: Next state derived from actual future month, not contemporaneous

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
1. **Rare States**: Poor performance on rare state transitions
2. **State Inconsistencies**: Conflicting servicer reports may affect state labels
3. **Missing History**: Poor performance on loans with incomplete history

### Edge Cases
1. **New Loans**: Limited history for recently originated loans
2. **State Changes**: May not capture rapid state changes
3. **Multiple Transitions**: May not handle multiple state changes in one month

## Intended Use

### Primary Use Cases
- **Trajectory Prediction**: Predict likely loan state transitions
- **Risk Assessment**: Identify loans likely to move to delinquent states
- **Portfolio Planning**: Understand expected portfolio composition changes

### Not Suitable For
- **Regulatory Capital**: Not validated for regulatory capital requirements
- **Pricing**: Not calibrated for pricing or valuation decisions
- **High-Stakes Decisions**: Limited explainability due to SHAP issues

## Ethical Considerations

### Fairness
- **Protected Classes**: Model does not explicitly use protected attributes (race, gender, age)
- **Geographic Bias**: State-based features may introduce geographic bias
- **Credit Score Bands**: Uses credit score bands which may reflect historical biases

### Transparency
- **Explainability**: SHAP explainability currently limited (known technical issue)
- **Feature Importance**: Global feature importance available
- **Model Documentation**: This card provides comprehensive documentation

## Maintenance

### Retraining Schedule
- **Recommended**: Quarterly retraining with recent data
- **Trigger**: Significant performance degradation or data drift
- **Monitoring**: Track macro F1, weighted F1, and class-specific performance

### Update Process
1. Collect new loan performance data
2. Apply same feature engineering pipeline
3. Retrain model with updated hyperparameters
4. Validate on recent time period
5. Deploy after performance verification

## References

- **Training Code**: `engine/src/models/next_state.py`
- **Feature Engineering**: `engine/src/features/build_features.py`
- **Data Pipeline**: `engine/src/data/splits.py`
- **Configuration**: `engine/config.yaml`
- **Model Artifacts**: `engine/data/processed/models/next_state_lgbm.pkl`

---

**Model Card Version**: 1.0  
**Last Updated**: September 2026  
**Maintained By**: Intain Campus FinTech Challenge 2026 AI Track Team

**Note**: SHAP explainability for this model is currently limited due to technical issues with multidimensional output. This is documented in the main model card and explainability report.
