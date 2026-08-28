# Scenario & Stress Simulation Report

Generated on 2026-08-28 02:08

## Aggregate Projected Rates

| scenario        |   next_3m_delinquency_flag |   next_12m_default_flag |   next_12m_prepayment_flag |
|:----------------|---------------------------:|------------------------:|---------------------------:|
| base            |                   0.364888 |              0.00502978 |                  0.0175429 |
| adverse_credit  |                   0.365492 |              0.00507968 |                  0.0173668 |
| high_prepayment |                   0.347377 |              0.00350061 |                  0.0156986 |

![Scenario Comparison](plots/scenario_comparison.png)

## Segment-Level Default Rate by Scenario

| segment_col       | segment_val   | scenario        |   default_rate |   n_loans |
|:------------------|:--------------|:----------------|---------------:|----------:|
| credit_score_band | Poor          | base            |         0.0123 |       359 |
| credit_score_band | Poor          | adverse_credit  |         0.0112 |       359 |
| credit_score_band | Poor          | high_prepayment |         0.0112 |       359 |
| state             | NV            | adverse_credit  |         0.0087 |       253 |
| state             | NV            | base            |         0.0081 |       253 |
| state             | TX            | adverse_credit  |         0.007  |       416 |
| state             | TX            | base            |         0.0068 |       416 |
| servicer_name     | ServicerB     | base            |         0.0063 |      1075 |
| state             | NV            | high_prepayment |         0.0062 |       253 |
| servicer_name     | ServicerB     | adverse_credit  |         0.0062 |      1075 |
| state             | IL            | base            |         0.0056 |       441 |
| credit_score_band | Good          | adverse_credit  |         0.0055 |      1681 |
| credit_score_band | Fair          | adverse_credit  |         0.0055 |       944 |
| credit_score_band | Good          | base            |         0.0055 |      1681 |
| state             | AZ            | adverse_credit  |         0.0054 |       504 |
| servicer_name     | ServicerC     | adverse_credit  |         0.0052 |      1107 |
| credit_score_band | Fair          | base            |         0.0052 |       944 |
| state             | GA            | base            |         0.0052 |       495 |
| state             | TX            | high_prepayment |         0.0051 |       416 |
| state             | AZ            | base            |         0.005  |       504 |
| state             | WA            | adverse_credit  |         0.0049 |       427 |
| servicer_name     | ServicerB     | high_prepayment |         0.0049 |      1075 |
| state             | IL            | adverse_credit  |         0.0049 |       441 |
| servicer_name     | ServicerC     | base            |         0.0048 |      1107 |
| servicer_name     | ServicerA     | base            |         0.0046 |      1083 |
| servicer_name     | ServicerA     | adverse_credit  |         0.0046 |      1083 |
| state             | AZ            | high_prepayment |         0.0045 |       504 |
| state             | OH            | adverse_credit  |         0.0045 |       439 |
| state             | NY            | adverse_credit  |         0.0045 |       474 |
| state             | OH            | base            |         0.0045 |       439 |

![Credit Band × Scenario Heatmap](plots/scenario_credit_heatmap.png)

## Scenario Sensitivity Drivers

The most sensitive inputs under each scenario:

- **adverse_credit**: `credit_score_band_enc` (+1 notch down) drives a 0.000 absolute increase in 12m default rate.

- **high_prepayment**: `interest_rate` shift (−0.75pp) and increased prepay propensity drive prepayment rate up by 0.002.
