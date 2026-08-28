"""
delinquency_default_prepay.py
==============================
Task 2 — Predictive Modeling (binary targets)

Trains for each of:
  next_3m_delinquency_flag, next_6m_delinquency_flag,
  next_12m_default_flag, next_12m_prepayment_flag

Models:
  - Logistic Regression baseline (class_weight=balanced, C=0.1)
  - LightGBM improved model (class_weight=balanced, calibrated)

Metrics reported: ROC-AUC, PR-AUC, F1, Recall@Precision80, Brier score

Outputs:
  data/processed/models/<target>_{lr,lgbm}.pkl
  reports/model_metrics.md
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, brier_score_loss,
                             precision_recall_curve, roc_curve)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def recall_at_precision(y_true, y_score, min_precision=0.80):
    """Return the max recall achievable with precision >= min_precision."""
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    valid = prec >= min_precision
    if valid.any():
        return float(rec[valid].max())
    return 0.0


def train_binary_models(X_train, y_train, X_val, y_val,
                        X_test, y_test, target: str, cfg: dict) -> dict:
    """Train LR + LightGBM for one binary target. Returns metrics dict."""
    plots_dir = ROOT / cfg["PATHS"].get("plots","reports/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / cfg["PATHS"].get("models","data/processed/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Drop NaN targets
    mask_tr = y_train.notna()
    mask_va = y_val.notna()
    mask_te = y_test.notna()
    Xtr, ytr = X_train[mask_tr], y_train[mask_tr].astype(int)
    Xva, yva = X_val[mask_va],   y_val[mask_va].astype(int)
    Xte, yte = X_test[mask_te],  y_test[mask_te].astype(int)

    pos_rate = ytr.mean()
    print(f"    {target}: pos_rate={pos_rate:.3f}, "
          f"train={len(ytr):,}, val={len(yva):,}, test={len(yte):,}")

    if pos_rate < 0.001 or pos_rate > 0.999:
        print(f"    Skipping {target} — degenerate target")
        return {}

    results = {}

    # ── Logistic Regression baseline ─────────────────────────────────────
    lr_params = cfg["MODELS"]["logreg"]
    lr = LogisticRegression(
        max_iter=lr_params["max_iter"],
        class_weight=lr_params["class_weight"],
        C=lr_params["C"],
        solver="lbfgs",
        random_state=cfg["RANDOM_SEED"],
    )
    lr.fit(Xtr, ytr)
    joblib.dump(lr, models_dir / f"{target}_lr.pkl")

    for split_name, Xs, ys in [("val",Xva,yva),("test",Xte,yte)]:
        prob = lr.predict_proba(Xs)[:, 1]
        pred = (prob > 0.5).astype(int)
        results[f"LR_{split_name}"] = _metrics(ys, prob, pred, target)

    # ── LightGBM improved model ───────────────────────────────────────────
    lg_params = cfg["MODELS"]["lgbm"]
    lgbm = lgb.LGBMClassifier(
        n_estimators=lg_params["n_estimators"],
        learning_rate=lg_params["learning_rate"],
        num_leaves=lg_params["num_leaves"],
        min_child_samples=lg_params["min_child_samples"],
        subsample=lg_params["subsample"],
        colsample_bytree=lg_params["colsample_bytree"],
        class_weight=lg_params["class_weight"],
        random_state=cfg["RANDOM_SEED"],
        verbosity=-1,
    )
    lgbm.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Isotonic calibration
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(lgbm, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)
    joblib.dump(lgbm, models_dir / f"{target}_lgbm_raw.pkl")
    joblib.dump(cal, models_dir / f"{target}_lgbm_cal.pkl")

    for split_name, Xs, ys in [("val",Xva,yva),("test",Xte,yte)]:
        prob = cal.predict_proba(Xs)[:, 1]
        pred = (prob > 0.5).astype(int)
        results[f"LGBM_{split_name}"] = _metrics(ys, prob, pred, target)

    # ── ROC + PR curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model_name, model_obj in [("LR", lr), ("LGBM", cal)]:
        prob_te = model_obj.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, prob_te)
        auc_val = roc_auc_score(yte, prob_te)
        axes[0].plot(fpr, tpr, label=f"{model_name} AUC={auc_val:.3f}")
        prec, rec, _ = precision_recall_curve(yte, prob_te)
        ap_val = average_precision_score(yte, prob_te)
        axes[1].plot(rec, prec, label=f"{model_name} AP={ap_val:.3f}")

    axes[0].set_title(f"ROC — {target}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(); axes[0].plot([0,1],[0,1],"k--")
    axes[1].set_title(f"PR — {target}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(plots_dir / f"{target}_curves.png", dpi=100)
    plt.close(fig)

    return results


def _metrics(y_true, y_prob, y_pred, target) -> dict:
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = np.nan
    return {
        "target":    target,
        "roc_auc":   round(roc, 4),
        "pr_auc":    round(pr, 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "brier":     round(brier_score_loss(y_true, y_prob), 4),
        "rec_p80":   round(recall_at_precision(y_true, y_prob, 0.80), 4),
    }


def run(X_train, X_val, X_test, y_dict, cfg) -> dict:
    """Train all binary targets. Returns nested dict of results."""
    print("\n[binary models] Training …")
    all_results = {}
    for tgt in BINARY_TARGETS:
        if tgt not in y_dict:
            continue
        y_tr, y_va, y_te = y_dict[tgt]
        res = train_binary_models(X_train, y_tr, X_val, y_va, X_test, y_te, tgt, cfg)
        all_results[tgt] = res

    # Save metrics
    proc_dir = ROOT / cfg["PATHS"]["processed_data"]
    proc_dir.mkdir(parents=True, exist_ok=True)
    with open(proc_dir / "binary_model_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
