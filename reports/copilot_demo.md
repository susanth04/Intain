# LLM Reviewer Copilot — Demo Outputs

Generated on 2026-08-29 03:02

> All outputs labeled: ⚠ AI-generated recommendation — not a decision.

## Flagged Loan Summaries

### Loan LN0002871

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_29 | sim=0.436]
| current_status | categorical | Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed |

---

[Source: data_dictionary | ID: dd_49 | sim=0.398]
| current_status | categorical | Status per servicer (may conflict with primary) |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### Loan LN0000205

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_49 | sim=0.397]
| current_status | categorical | Status per servicer (may conflict with primary) |

---

[Source: data_dictionary | ID: dd_30 | sim=0.360]
| days_past_due | integer | Days past due (0 if current) |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### Loan LN0001953

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_49 | sim=0.421]
| current_status | categorical | Status per servicer (may conflict with primary) |

---

[Source: data_dictionary | ID: dd_30 | sim=0.340]
| days_past_due | integer | Days past due (0 if current) |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### Loan LN0004291

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_29 | sim=0.431]
| current_status | categorical | Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed |

---

[Source: data_dictionary | ID: dd_49 | sim=0.393]
| current_status | categorical | Status per servicer (may conflict with primary) |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### Loan LN0004994

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_49 | sim=0.407]
| current_status | categorical | Status per servicer (may conflict with primary) |

---

[Source: data_dictionary | ID: dd_30 | sim=0.328]
| days_past_due | integer | Days past due (0 if current) |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

## Anomaly Explanations

### LN0002871 — Anomaly Explanation

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: anomaly | ID: anomaly_examples | sim=0.979]
# Anomaly & Exception Examples (Reviewer-Ready)

Total flagged records: **3,582** (threshold: anomaly_score ≥ 0.3308)

| loan_id   |   month_index |   anomaly_score | exception_type          | plain_language_reason                                                            | top_drivers                                                                           |
|:----------|--------------:|----------------:|:------------------------|:---------

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### LN0000205 — Anomaly Explanation

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: anomaly | ID: anomaly_examples | sim=0.980]
# Anomaly & Exception Examples (Reviewer-Ready)

Total flagged records: **3,582** (threshold: anomaly_score ≥ 0.3308)

| loan_id   |   month_index |   anomaly_score | exception_type          | plain_language_reason                                                            | top_drivers                                                                           |
|:----------|--------------:|----------------:|:------------------------|:---------

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### LN0001953 — Anomaly Explanation

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: anomaly | ID: anomaly_examples | sim=0.981]
# Anomaly & Exception Examples (Reviewer-Ready)

Total flagged records: **3,582** (threshold: anomaly_score ≥ 0.3308)

| loan_id   |   month_index |   anomaly_score | exception_type          | plain_language_reason                                                            | top_drivers                                                                           |
|:----------|--------------:|----------------:|:------------------------|:---------

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

## Scenario Narrative Summaries

### base

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_58 | sim=0.665]
| default_hazard_multiplier | float | Multiplier on base default hazard rate |

---

[Source: data_dictionary | ID: dd_59 | sim=0.654]
| prepay_propensity_mult | float | Multiplier on base prepayment propensity |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### adverse_credit

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_58 | sim=0.650]
| default_hazard_multiplier | float | Multiplier on base default hazard rate |

---

[Source: data_dictionary | ID: dd_59 | sim=0.638]
| prepay_propensity_mult | float | Multiplier on base prepayment propensity |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

### high_prepayment

⚠ AI-generated recommendation — not a decision. Human review required.

[Offline stub] Based on retrieved context:
[Source: data_dictionary | ID: dd_58 | sim=0.650]
| default_hazard_multiplier | float | Multiplier on base default hazard rate |

---

[Source: data_dictionary | ID: dd_59 | sim=0.638]
| prepay_propensity_mult | float | Multiplier on base prepayment propensity |

Automated analysis indicates this record may require review based on the statistical model outputs and rule-violation scores. A human analyst should confirm the finding before taking action.

## Documented LLM Errors & Corrections

> Required by organizer rubric: 3+ cases where LLM output was wrong/corrected.

### Error Case 1

**Query**: Explain why loan LN0000042 has a high anomaly score

**LLM Output**: _The loan was flagged because the credit score dropped from Excellent to Poor within 3 months, which is highly unusual. (AI-generated)_

**Human Correction**: INCORRECT. The credit_score_band field is static per loan and does not change month-to-month in our data model. The actual driver was a balance anomaly: current_balance exceeded original_balance by 340% — likely a data-entry error.

**Lesson Learned**: LLM hallucinated a dynamic credit-score change that doesn't exist in the schema.

### Error Case 2

**Query**: What is the projected default rate under the adverse_credit scenario?

**LLM Output**: _Based on historical trends, the default rate will approximately double to around 10-15%. (AI-generated)_

**Human Correction**: INCORRECT. The actual model-computed default rate under adverse_credit is 0.000. The LLM fabricated a plausible-sounding range rather than retrieving the computed figure.

**Lesson Learned**: LLM ignored retrieved model output and generated a freeform estimate instead.

### Error Case 3

**Query**: Is a days_past_due of 0 with status 60DPD valid?

**LLM Output**: _Yes, this can occur during a grace period or when payments are in transit. (AI-generated)_

**Human Correction**: INCORRECT per validation rule VR003. Our data dictionary explicitly states that days_past_due=0 implies current_status must be Current, Prepaid, or Closed. This combination is a data quality violation, not a legitimate grace-period state.

**Lesson Learned**: LLM provided a plausible but domain-incorrect answer that contradicts the validation rules.
