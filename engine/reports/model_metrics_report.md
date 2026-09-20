# Model Performance Metrics

Generated on 2026-08-31 20:02

## Binary Targets (Baseline LR vs. Improved LightGBM)

| Target                   | Model_Split   | target                   |   roc_auc |   pr_auc |     f1 |   brier |   rec_p80 |
|:-------------------------|:--------------|:-------------------------|----------:|---------:|-------:|--------:|----------:|
| next_3m_delinquency_flag | LR_val        | next_3m_delinquency_flag |    0.6518 |   0.4671 | 0.453  |  0.2209 |    0      |
| next_3m_delinquency_flag | LR_test       | next_3m_delinquency_flag |    0.6618 |   0.4887 | 0.4089 |  0.2132 |    0      |
| next_3m_delinquency_flag | LGBM_val      | next_3m_delinquency_flag |    0.7519 |   0.6212 | 0.5013 |  0.1867 |    0.1233 |
| next_3m_delinquency_flag | LGBM_test     | next_3m_delinquency_flag |    0.7129 |   0.5732 | 0.4604 |  0.1973 |    0.0968 |
| next_6m_delinquency_flag | LR_val        | next_6m_delinquency_flag |    0.6463 |   0.6106 | 0.3518 |  0.2524 |    0.0099 |
| next_6m_delinquency_flag | LR_test       | next_6m_delinquency_flag |    0.658  |   0.6148 | 0.2262 |  0.2688 |    0      |
| next_6m_delinquency_flag | LGBM_val      | next_6m_delinquency_flag |    0.7897 |   0.7685 | 0.7332 |  0.1909 |    0.4142 |
| next_6m_delinquency_flag | LGBM_test     | next_6m_delinquency_flag |    0.7098 |   0.6873 | 0.6801 |  0.2161 |    0.2094 |
| next_12m_default_flag    | LR_val        | next_12m_default_flag    |    0.7247 |   0.3017 | 0.3407 |  0.1373 |    0      |
| next_12m_default_flag    | LR_test       | next_12m_default_flag    |    0.6966 |   0.2743 | 0.1665 |  0.1222 |    0      |
| next_12m_default_flag    | LGBM_val      | next_12m_default_flag    |    0.7247 |   0.3017 | 0.3407 |  0.1373 |    0      |
| next_12m_default_flag    | LGBM_test     | next_12m_default_flag    |    0.6966 |   0.2743 | 0.1665 |  0.1222 |    0      |
| next_12m_prepayment_flag | LR_val        | next_12m_prepayment_flag |    0.5717 |   0.3044 | 0.434  |  0.2414 |    0      |
| next_12m_prepayment_flag | LR_test       | next_12m_prepayment_flag |    0.5652 |   0.315  | 0.4377 |  0.2459 |    0      |
| next_12m_prepayment_flag | LGBM_val      | next_12m_prepayment_flag |    0.5717 |   0.3044 | 0.434  |  0.2414 |    0      |
| next_12m_prepayment_flag | LGBM_test     | next_12m_prepayment_flag |    0.5652 |   0.315  | 0.4377 |  0.2459 |    0      |

## Next State Multiclass

| Model_Split   |   macro_f1 |   weighted_f1 |
|:--------------|-----------:|--------------:|
| LR_val        |     0.3336 |        0.5562 |
| LR_test       |     0.314  |        0.5693 |
| LGBM_val      |     0.5219 |        0.7035 |
| LGBM_test     |     0.5083 |        0.7142 |

## Survival/Hazard Model

- **naive_hazard_rate**: 0.05176

- **km_median_survival**: {'Excellent': 16.0, 'Good': 16.0, 'Fair': 13.0, 'Poor': 10.0}

- **cox_concordance**: 0.5617
