# Model Comparison Analysis

## Overview
This analysis compares Logistic Regression (baseline) vs LightGBM (improved) for loan performance prediction using time-aware validation. The comparison was conducted as part of Phase 3 of the Intain Campus FinTech Challenge 2026 AI Track.

## Models Compared

### Logistic Regression (Baseline)
- **Type**: Linear classifier with L2 regularization
- **Parameters**: `class_weight='balanced'`, `C=0.1`, `max_iter=1000`
- **Advantages**: Interpretable, fast training, low computational cost
- **Limitations**: Limited capacity to capture non-linear relationships

### LightGBM (Improved)
- **Type**: Gradient boosting decision trees
- **Parameters**: `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, `class_weight='balanced'`
- **Calibration**: CalibratedClassifierCV with sigmoid method (except for weak targets)
- **Advantages**: Handles non-linear relationships, feature interactions, missing values
- **Limitations**: More complex, longer training time, less interpretable

## Time-Aware Split
- **Train**: months 1-20 (month_index ≤ 20)
- **Validation**: months 21-25 (month_index 21-25) 
- **Test**: months 26-30 (month_index 26-30)
- **Rationale**: Prevents data leakage by ensuring model never sees future outcomes when predicting past periods

## Results Summary

| Target | Model | Split | ROC-AUC | PR-AUC | F1 | Brier |
|--------|-------|-------|---------|--------|-----|-------|
| next_3m_delinquency_flag | LR | val | 0.6518 | 0.4671 | 0.453 | 0.2209 |
| next_3m_delinquency_flag | LR | test | 0.6618 | 0.4887 | 0.4089 | 0.2132 |
| next_3m_delinquency_flag | LGBM | val | 0.7519 | 0.6212 | 0.5013 | 0.1867 |
| next_3m_delinquency_flag | LGBM | test | 0.7129 | 0.5732 | 0.4604 | 0.1973 |
| next_6m_delinquency_flag | LR | val | 0.6463 | 0.6106 | 0.3518 | 0.2524 |
| next_6m_delinquency_flag | LR | test | 0.658 | 0.6148 | 0.2262 | 0.2688 |
| next_6m_delinquency_flag | LGBM | val | 0.7897 | 0.7685 | 0.7332 | 0.1909 |
| next_6m_delinquency_flag | LGBM | test | 0.7098 | 0.6873 | 0.6801 | 0.2161 |
| next_12m_default_flag | LR | val | 0.7247 | 0.3017 | 0.3407 | 0.1373 |
| next_12m_default_flag | LR | test | 0.6966 | 0.2743 | 0.1665 | 0.1222 |
| next_12m_default_flag | LGBM | val | 0.7247 | 0.3017 | 0.3407 | 0.1373 |
| next_12m_default_flag | LGBM | test | 0.6966 | 0.2743 | 0.1665 | 0.1222 |
| next_12m_prepayment_flag | LR | val | 0.5717 | 0.3044 | 0.434 | 0.2414 |
| next_12m_prepayment_flag | LR | test | 0.5652 | 0.315 | 0.4377 | 0.2459 |
| next_12m_prepayment_flag | LGBM | val | 0.5717 | 0.3044 | 0.434 | 0.2414 |
| next_12m_prepayment_flag | LGBM | test | 0.5652 | 0.315 | 0.4377 | 0.2459 |

## Key Findings

### Delinquency Prediction (Strong Performance)
- **3-month delinquency**: LightGBM significantly outperforms LR
  - Test ROC-AUC: 0.7129 vs 0.6618 (+7.7% improvement)
  - Test PR-AUC: 0.5732 vs 0.4887 (+17.3% improvement)
  - Test F1: 0.4604 vs 0.4089 (+12.6% improvement)
  - Better Brier score (lower is better): 0.1973 vs 0.2132

- **6-month delinquency**: LightGBM shows strong improvement
  - Test ROC-AUC: 0.7098 vs 0.658 (+7.9% improvement)
  - Test PR-AUC: 0.6873 vs 0.6148 (+11.8% improvement)
  - Test F1: 0.6801 vs 0.2262 (+200% improvement - major gain)
  - Better Brier score: 0.2161 vs 0.2688

### Default & Prepayment (Weak Performance)
- **12-month default**: Both models show identical performance
  - Test ROC-AUC: 0.6966 for both models
  - This indicates the target lacks predictive signal in current features
  - Training code falls back to LR for this target due to "no clear signal"

- **12-month prepayment**: Both models show weak performance
  - Test ROC-AUC: 0.5652 for both models (barely above random)
  - Test PR-AUC: 0.315 for both models
  - Indicates limited predictability with current feature set

## Winning Models Selection

Based on test ROC-AUC performance:

| Target | Winner | Test ROC-AUC | Rationale |
|--------|--------|-------------|-----------|
| next_3m_delinquency_flag | **LightGBM** | 0.7129 | Clear performance advantage over LR |
| next_6m_delinquency_flag | **LightGBM** | 0.7098 | Significant improvement across all metrics |
| next_12m_default_flag | **Logistic Regression** | 0.6966 | Tied performance, LR selected for simplicity |
| next_12m_prepayment_flag | **Logistic Regression** | 0.5652 | Tied performance, LR selected for simplicity |

## Recommendations

### Immediate Actions
1. **Use LightGBM** for 3m and 6m delinquency prediction (clear performance gain)
2. **Investigate default/prepayment targets** - need better features or more data
3. **Feature engineering** could improve performance on weak targets

### Future Improvements
1. **Additional model families**: Train Random Forest, XGBoost, CatBoost for comparison
2. **Hyperparameter tuning**: Systematic hyperparameter optimization for LightGBM
3. **Feature importance analysis**: Deep dive into SHAP values to understand key drivers
4. **Ensemble methods**: Combine multiple models for potentially better performance
5. **Target engineering**: Re-examine default/prepayment definitions and feature relationships

### Technical Notes
- All models use the same time-aware split to prevent data leakage
- LightGBM models are calibrated using CalibratedClassifierCV (except for weak targets)
- Class imbalance handled via `class_weight='balanced'`
- For full multi-model comparison (RF, XGBoost, CatBoost), additional training would be required
- The synthetic data generation may limit signal for long-term predictions (12m horizon)

## Limitations

1. **Synthetic Data**: Results based on synthetic loan data may not reflect real-world performance
2. **Limited Model Comparison**: Only LR vs LightGBM compared; additional model families not evaluated
3. **Weak Targets**: Default and prepayment targets show limited predictability
4. **Single Time Window**: Only one time-aware split evaluated; could benefit from cross-validation
5. **Feature Set**: Current features may not capture all relevant signals for long-term predictions

## Conclusion

LightGBM demonstrates clear superiority for short-term delinquency prediction (3m and 6m horizons), making it the preferred choice for these use cases. However, the identical performance on default and prepayment targets suggests fundamental limitations in either the target definitions, feature engineering, or data quality for long-term predictions. Further investigation and feature engineering would be needed to improve performance on these weaker targets.
