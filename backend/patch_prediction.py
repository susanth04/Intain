import re

path = r'C:\Users\susan\Downloads\intain-loan-intelligence-dashboard\backend\pipeline\prediction.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add LR helper after BINARY_TARGETS definition
lr_block = '''
LR_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def _run_lr_predictions(feat_df):
    lr_results = {}
    for tgt in LR_TARGETS:
        model = MODELS.get(f"{tgt}_lr")
        if model is None:
            continue
        try:
            try:
                proba = model.predict_proba(feat_df)[:, 1]
            except Exception:
                if hasattr(model, "feature_names_in_"):
                    aligned = feat_df.reindex(columns=list(model.feature_names_in_), fill_value=0)
                    proba = model.predict_proba(aligned)[:, 1]
                else:
                    continue
            lr_results[tgt] = _safe_float(proba[0])
        except Exception:
            pass
    return lr_results

'''

# Inject before def run(
content = content.replace('def run(loan_id: str) -> dict:', lr_block + 'def run(loan_id: str) -> dict:')

# Change the final return to include ensemble confidence
old_return = '''    return {
        "loan_id": loan_id,
        "predictions": predictions,
        "next_state": next_state_result,
        "recommended_action": recommended_action,
        "features_used": len(feature_names),
        "execution_ms": elapsed_ms,
    }'''

new_return = '''    lr_predictions = _run_lr_predictions(feat_df)
    ensemble = {}
    for tgt in BINARY_TARGETS:
        lgbm_p = predictions.get(tgt, {}).get("probability")
        lr_p   = lr_predictions.get(tgt)
        if lgbm_p is not None and lr_p is not None:
            diff = abs(lgbm_p - lr_p)
            ensemble[tgt] = {
                "lgbm_probability": lgbm_p,
                "lr_probability":   lr_p,
                "agreement_gap":    round(diff, 4),
                "confidence":       "high" if diff < 0.10 else ("medium" if diff < 0.25 else "low"),
            }

    return {
        "loan_id": loan_id,
        "predictions": predictions,
        "ensemble_confidence": ensemble,
        "next_state": next_state_result,
        "recommended_action": recommended_action,
        "features_used": len(feature_names),
        "execution_ms": elapsed_ms,
    }'''

content = content.replace(old_return, new_return)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done - prediction.py updated')
