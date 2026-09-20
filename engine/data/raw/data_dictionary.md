# Data Dictionary — Loan Performance Intelligence Engine

## loan_static_attributes.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Unique loan identifier (format: LN0000001) |
| original_balance | integer | Original loan balance in USD at origination |
| credit_score_band | categorical | Borrower credit quality: Excellent/Good/Fair/Poor |
| ltv_band | categorical | Loan-to-value ratio band: <=60/60-75/75-90/>90 |
| dti_band | categorical | Debt-to-income ratio band: <=28/28-36/36-43/>43 |
| state | categorical | US state abbreviation |
| loan_purpose | categorical | Purchase/Refinance/CashOut |
| occupancy_type | categorical | PrimaryResidence/SecondHome/Investment |
| property_type | categorical | SingleFamily/Condo/MultiUnit/Townhouse |
| vintage | string | Year of origination |

## loan_monthly_performance_train.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Foreign key to loan_static_attributes |
| month_index | integer | 1-based month sequence since origination |
| reporting_month | string | YYYY-MM reporting period |
| origination_month | string | YYYY-MM origination period |
| loan_age_months | integer | Months since origination |
| remaining_term_months | integer | Months remaining on 360-month term |
| original_balance | integer | Original balance (repeated for join convenience) |
| current_balance | float | Ending balance this month |
| interest_rate | float | Annual interest rate (%) |
| credit_score_band | categorical | May change if modification occurs |
| ltv_band | categorical | May change with balance / HPI changes |
| dti_band | categorical | Borrower DTI band |
| state | categorical | Loan state |
| loan_purpose | categorical | Purchase/Refinance/CashOut |
| occupancy_type | categorical | Occupancy type |
| property_type | categorical | Property type |
| servicer_name | categorical | Loan servicer |
| current_status | categorical | Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed |
| days_past_due | integer | Days past due (0 if current) |
| modification_flag | binary | 1 if loan was modified this period |
| prepayment_flag | binary | 1 if prepaid this period |
| default_flag | binary | 1 if defaulted this period |
| loss_severity_band | categorical | Low/Medium/High expected loss severity |
| last_updated_at | date | Date of last record update (YYYY-MM-DD) |
| source_system | string | Source system identifier |
| document_status | categorical | Complete/Incomplete/Pending/Waived |
| next_3m_delinquency_flag | binary | TARGET: delinquent within next 3 months |
| next_6m_delinquency_flag | binary | TARGET: delinquent within next 6 months |
| next_12m_default_flag | binary | TARGET: defaults within next 12 months |
| next_12m_prepayment_flag | binary | TARGET: prepays within next 12 months |
| next_state | categorical | TARGET: predicted state next month |
| exception_required | binary | TARGET: record requires exception review |
| exception_type | categorical | TARGET: type of exception |

## servicer_updates.csv
| Field | Type | Description |
|---|---|---|
| loan_id | string | Loan identifier |
| month_index | integer | Month sequence |
| reporting_month | string | YYYY-MM |
| current_status | categorical | Status per servicer (may conflict with primary) |
| days_past_due | integer | DPD per servicer |
| last_updated_at | date | Servicer timestamp (may be stale) |
| servicer_name | categorical | Servicer name |
| source_system | string | SERVICER_FEED |
| conflict_flag | binary | 1 if this record conflicts with primary |

## macro_scenarios.csv
| Field | Type | Description |
|---|---|---|
| month_index | integer | Month index |
| scenario | categorical | base/adverse_credit/high_prepayment |
| default_hazard_multiplier | float | Multiplier on base default hazard rate |
| prepay_propensity_mult | float | Multiplier on base prepayment propensity |
| rate_shift_pp | float | Interest rate shift in percentage points |
| hpi_shock_pct | float | Home price index shock (fraction) |
| unemployment_shock | float | Unemployment rate shock (pp) |
