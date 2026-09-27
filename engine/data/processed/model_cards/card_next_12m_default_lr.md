# Model Card: 12-Month Default Prediction (Logistic Regression)

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | next_12m_default_flag_lr |
| **Model Type** | Logistic Regression |
| **Version** | 1.0 |
| **Training Date** | August 2026 |
| **Purpose** | Predict whether a loan will default within 12 months |

## Objective

Predict the probability that a loan will default (severe credit event) within the next 12 months based on current loan characteristics and payment history. This is a fallback model due to limited signal in the target.

## Data

### Training Data
- **Source**: Synthetic loan panel data (Fannie Mae-inspired)
- **Time Period**: Months 1-20 (training), Months 21-25 (validation), Months 26-30 (test)
- **Sample Size**: ~81,189 panel rows × 5,000 unique loans
- **Target Variable**: `next_12m_default_flag` (binary: 1 if default within 12 months, else 0)
- **Positive Rate**: ~19% in training data

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
- **Logistic Regression** with L2 regularization
- **Linear classifier** with probabilistic interpretation
- **Baseline model** for comparison with more complex methods

### Hyperparameters
```yaml
max_iter: 1000
class_weight: balanced
C: 0.1
solver: lbfgs
random_state: 42
```

### Calibration
- **Method**: None (Logistic Regression provides well-calibrated probabilities by default)
- **Purpose**: Simplicity and interpretability

## Validation Method

### Time-Aware Split
- **Training Set**: months 1-20 (month_index ≤ 20)
- **Validation Set**: months 21-25 (month_index 21-25) 
- **Test Set**: months 26-30 (month_index 26-30)
- **Rationale**: Prevents leakage by ensuring model never sees future outcomes

## Metrics

### Performance Metrics

| Split | ROC-AUC | PR-AUC | F1 | Brier Score | Recall@Precision80 |
|-------|---------|--------|-----|-------------|-------------------|
| Validation | 0.7247 | 0.3017 | 0.3407 | 0.1373 | 0.0 |
| Test | 0.6966 | 0.2743 | 0.1665 | 0.1222 | 0.0 |

### Comparison to LightGBM
- **LightGBM Test ROC-AUC**: 0.6966
- **Logistic Regression Test ROC-AUC**: 0.6966
- **Improvement**: No improvement (identical performance)

### Key Limitations
- **Weak Signal**: PR-AUC of 0.27 indicates limited discrimination
- **Zero Recall at High Precision**: Cannot achieve 80% precision with meaningful recall
- **Identical Performance**: LightGBM shows no improvement, suggesting fundamental data limitations

## Limitations

### Data Limitations
- **Synthetic Data**: Trained on synthetic loan data; real-world performance may differ
- **Weak Target**: Default events may be too rare or poorly defined in synthetic data
- **Long Horizon**: 12-month prediction may be beyond current feature predictive power

### Model Limitations
- **Linear Assumptions**: Cannot capture non-linear relationships
- **Feature Interactions**: Limited ability to model complex feature interactions
- **Weak Performance**: Model shows limited predictive ability

### Temporal Limitations
- **Concept Drift**: Long-term predictions especially susceptible to drift
- **Economic Sensitivity**: Not trained on different economic scenarios

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
1. **Rare Events**: Default events may be too rare for reliable modeling
2. **Definition Issues**: Default definition may be unclear in synthetic data
3. **Insufficient History**: 12-month predictions may require longer loan history

### Edge Cases
1. **New Loans**: Very poor performance on recently originated loans
2. **Early Defaults**: May not capture very early default patterns
3. **Modifications**: Loan modifications may affect default patterns

## Intended Use

### Primary Use Cases
- **Baseline Comparison**: Serve as baseline for more complex models
- **Interpretability**: Provide interpretable risk scores
- **Limited Production Use**: Only if no better model available

### Not Suitable For
- **Credit Decisions**: Not intended for origination underwriting
- **Regulatory Capital**: Not validated for regulatory capital requirements
- **High-Stakes Decisions**: Weak performance makes it unsuitable for critical decisions

## Recommendations

### Immediate Actions
1. **Investigate Target**: Re-examine default definition and data quality
2. **Feature Engineering**: Develop features specifically for default prediction
3. **Data Collection**: Consider additional data sources for default prediction

### Future Improvements
1. **Alternative Targets**: Consider different default definitions or time horizons
2. **Survival Models**: Use survival analysis for time-to-default modeling
3. **Ensemble Methods**: Combine with other risk indicators

## Ethical Considerations

### Fairness
- **Protected Classes**: Model does not explicitly use protected attributes (race, gender, age)
- **Geographic Bias**: State-based features may introduce geographic bias
- **Credit Score Bands**: Uses credit score bands which may reflect historical biases

### Transparency
- **Explainability**: Linear coefficients provide direct interpretability
- **Feature Importance**: Coefficient magnitudes indicate feature importance
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
- **Model Artifacts**: `engine/data/processed/models/next_12m_default_flag_lr.pkl`

---

**Model Card Version**: 1.0  
**Last Updated**: September 2026  
**Maintained By**: Intain Campus FinTech Challenge 2026 AI Track Team

**Note**: This model shows weak performance and is primarily used as a baseline. For production use, investigate alternative approaches or feature engineering for default prediction.
