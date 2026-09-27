# Model Card: 12-Month Prepayment Prediction (Logistic Regression)

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | next_12m_prepayment_flag_lr |
| **Model Type** | Logistic Regression |
| **Version** | 1.0 |
| **Training Date** | August 2026 |
| **Purpose** | Predict whether a loan will prepay within 12 months |

## Objective

Predict the probability that a loan will prepay (pay off early) within the next 12 months based on current loan characteristics and payment history. This is a fallback model due to limited signal in the target.

## Data

### Training Data
- **Source**: Synthetic loan panel data (Fannie Mae-inspired)
- **Time Period**: Months 1-20 (training), Months 21-25 (validation), Months 26-30 (test)
- **Sample Size**: ~81,189 panel rows × 5,000 unique loans
- **Target Variable**: `next_12m_prepayment_flag` (binary: 1 if prepay within 12 months, else 0)
- **Positive Rate**: ~26% in training data

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
| Validation | 0.5717 | 0.3044 | 0.434 | 0.2414 | 0.0 |
| Test | 0.5652 | 0.315 | 0.4377 | 0.2459 | 0.0 |

### Comparison to LightGBM
- **LightGBM Test ROC-AUC**: 0.5652
- **Logistic Regression Test ROC-AUC**: 0.5652
- **Improvement**: No improvement (identical performance)

### Key Limitations
- **Very Weak Signal**: ROC-AUC of 0.57 indicates barely better than random
- **Poor Precision-Recall**: PR-AUC of 0.31 shows limited discrimination
- **Zero Recall at High Precision**: Cannot achieve 80% precision with meaningful recall
- **Identical Performance**: LightGBM shows no improvement, suggesting fundamental data limitations

## Limitations

### Data Limitations
- **Synthetic Data**: Trained on synthetic loan data; real-world performance may differ
- **Weak Target**: Prepayment events may be poorly predictable with current features
- **Long Horizon**: 12-month prediction may be beyond current feature predictive power
- **Rate Environment**: Prepayment highly sensitive to interest rate changes not captured

### Model Limitations
- **Linear Assumptions**: Cannot capture non-linear relationships
- **Feature Interactions**: Limited ability to model complex feature interactions
- **Weak Performance**: Model shows very limited predictive ability
- **Rate Sensitivity**: Does not incorporate future rate expectations

### Temporal Limitations
- **Concept Drift**: Prepayment patterns highly sensitive to economic conditions
- **Economic Sensitivity**: Not trained on different interest rate environments
- **Market Conditions**: Does not account for housing market or refinance activity

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
1. **Interest Rate Changes**: Model not sensitive to future rate movements
2. **Housing Market Shifts**: Does not incorporate housing price trends
3. **Refinance Activity**: Cannot predict refinance waves

### Data Quality Issues
1. **Prepayment Definition**: May not capture all prepayment reasons
2. **Rate Data**: Current interest rate may not reflect future rate environment
3. **Borrower Behavior**: Prepayment often driven by factors not in current features

### Edge Cases
1. **Rate Changes**: Performance degrades during rate volatility
2. **Seasonal Patterns**: May not capture seasonal prepayment patterns
3. **New Loans**: Very poor performance on recently originated loans

## Intended Use

### Primary Use Cases
- **Baseline Comparison**: Serve as baseline for more complex models
- **Interpretability**: Provide interpretable risk scores
- **Limited Production Use**: Only if no better model available

### Not Suitable For
- **Pricing Decisions**: Not accurate enough for pricing or valuation
- **Portfolio Strategy**: Too weak for strategic prepayment risk management
- **Investment Decisions**: Not suitable for MBS investment decisions

## Recommendations

### Immediate Actions
1. **Feature Engineering**: Add interest rate forecasts and market indicators
2. **Target Definition**: Consider different prepayment definitions (voluntary vs involuntary)
3. **Data Collection**: Incorporate external economic indicators

### Future Improvements
1. **Survival Models**: Use survival analysis for time-to-prepayment modeling
2. **Rate Sensitivity**: Model prepayment as function of rate spread and expected changes
3. **Market Data**: Incorporate housing market indices and refinance activity data

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
- **Recommended**: Monthly retraining given high economic sensitivity
- **Trigger**: Significant interest rate changes or performance degradation
- **Monitoring**: Track ROC-AUC, calibration, and economic conditions

### Update Process
1. Collect new loan performance data
2. Update economic indicators and rate forecasts
3. Apply same feature engineering pipeline
4. Retrain model with updated hyperparameters
5. Validate on recent time period
6. Deploy after performance verification

## References

- **Training Code**: `engine/src/models/delinquency_default_prepay.py`
- **Feature Engineering**: `engine/src/features/build_features.py`
- **Data Pipeline**: `engine/src/data/splits.py`
- **Configuration**: `engine/config.yaml`
- **Model Artifacts**: `engine/data/processed/models/next_12m_prepayment_flag_lr.pkl`

---

**Model Card Version**: 1.0  
**Last Updated**: September 2026  
**Maintained By**: Intain Campus FinTech Challenge 2026 AI Track Team

**Note**: This model shows very weak performance (ROC-AUC ~0.57, barely above random) and is primarily used as a baseline. For production use, significant feature engineering incorporating interest rate forecasts and economic indicators is required.
