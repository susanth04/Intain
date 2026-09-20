import pandas as pd
import sys
import numpy as np

def validate(csv_path, expected_months):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FAIL: Could not read {csv_path} - {e}")
        return False
        
    passed_all = True
    
    # Check 1: prob_default_12m
    if 'prob_default_12m' in df.columns:
        p_def = df['prob_default_12m']
        near_zero = (p_def < 0.01).mean() * 100
        exact_0_1 = ((p_def == 0.0) | (p_def == 1.0)).mean() * 100
        std_dev = p_def.std()
        
        if near_zero > 25.0:
            print(f"FAIL: prob_default_12m: {near_zero:.1f}% of rows are ~0 (limit 25.0%)")
            passed_all = False
        if exact_0_1 > 5.0:
            print(f"FAIL: prob_default_12m: {exact_0_1:.1f}% of rows sit at exactly 0.0 or 1.0 (limit 5.0%)")
            passed_all = False
        if std_dev < 0.03:
            print(f"FAIL: prob_default_12m: std={std_dev:.3f} (limit >= 0.03)")
            passed_all = False
    
    # Check 1: prob_prepayment_12m
    if 'prob_prepayment_12m' in df.columns:
        p_pre = df['prob_prepayment_12m']
        near_zero = (p_pre < 0.01).mean() * 100
        exact_0_1 = ((p_pre == 0.0) | (p_pre == 1.0)).mean() * 100
        std_dev = p_pre.std()
        
        if near_zero > 25.0:
            print(f"FAIL: prob_prepayment_12m: {near_zero:.1f}% of rows are ~0 (limit 25.0%)")
            passed_all = False
        if exact_0_1 > 5.0:
            print(f"FAIL: prob_prepayment_12m: {exact_0_1:.1f}% of rows sit at exactly 0.0 or 1.0 (limit 5.0%)")
            passed_all = False
        if std_dev < 0.03:
            print(f"FAIL: prob_prepayment_12m: std={std_dev:.3f} (limit >= 0.03)")
            passed_all = False
            
    # Check 2: recommended_action disconnected
    if 'predicted_next_state' in df.columns and 'recommended_action' in df.columns:
        default_rows = df[df['predicted_next_state'] == 'Default']
        prepaid_rows = df[df['predicted_next_state'] == 'Prepaid']
        
        if len(default_rows) > 0:
            def_action = default_rows['recommended_action'].str.lower().str.contains('default|review|monitor').mean() * 100
            if def_action < 60.0:
                print(f"FAIL: only {def_action:.1f}% of predicted_next_state=Default rows get a default-relevant action")
                passed_all = False
                
        if len(prepaid_rows) > 0:
            pre_action = prepaid_rows['recommended_action'].str.lower().str.contains('prepay').mean() * 100
            if pre_action < 60.0:
                print(f"FAIL: only {pre_action:.1f}% of predicted_next_state=Prepaid rows get a prepayment-relevant action")
                passed_all = False

    # Check 3: Anomaly top_drivers
    if 'top_drivers' in df.columns and 'loan_id' in df.columns:
        # Check consecutive identical
        # group by loan_id
        loans_with_repeats = 0
        for loan_id, group in df.groupby('loan_id'):
            if len(group) >= 4:
                drivers = group['top_drivers'].fillna('')
                # check if there are 4 consecutive identical drivers that are not empty
                mask = (drivers != '')
                if mask.sum() >= 4:
                    # just check if all non-empty are identical
                    non_empty = drivers[mask]
                    if len(non_empty.unique()) == 1 and len(non_empty) >= 4:
                        print(f"FAIL: top_drivers repeats identically for {len(non_empty)} consecutive months for loan {loan_id}")
                        passed_all = False
                        break
        
        # Check magnitude
        import re
        max_dev = 0
        for d in df['top_drivers'].dropna():
            matches = re.findall(r'dev=([\d\.]+)', str(d))
            for m in matches:
                max_dev = max(max_dev, float(m))
        if max_dev > 2e6:
            print(f"FAIL: current_balance deviation as high as {max_dev} appears in top_drivers")
            passed_all = False

    # Check 4: Exception rate
    if 'exception_flag' in df.columns:
        exc_rate = df['exception_flag'].mean() * 100
        if exc_rate > 15.0 or exc_rate < 1.0:
            print(f"FAIL: exception rate {exc_rate:.1f}% exceeds 15% (or is <1%)")
            passed_all = False
            
    # Check 5: Scope of test window
    if 'loan_id' in df.columns:
        max_rows = df.groupby('loan_id').size().max()
        if max_rows > expected_months * 1.5:
            print(f"FAIL: some loans have {max_rows} rows, well above the expected test window of ~{expected_months}")
            passed_all = False

    if passed_all:
        print("ALL CHECKS PASSED")
    return passed_all

if __name__ == "__main__":
    validate(sys.argv[1], int(sys.argv[2]))
