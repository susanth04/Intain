"""
next_state.py
=============
Task 2 — Next State Multiclass Prediction

Predicts the next loan state (Current/30DPD/60DPD/90DPD/Default/Prepaid/Closed)
using:
  - Logistic Regression baseline (multinomial)
  - LightGBM multiclass classifier

Metrics: macro-F1, weighted-F1, per-class precision/recall
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]

STATES = ["Current","30DPD","60DPD","90DPD","Default","Prepaid","Closed"]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def run(X_train, X_val, X_test, y_dict, cfg) -> dict:
    tgt = "next_state"
    if tgt not in y_dict:
        print("[next_state] Target not available, skipping.")
        return {}

    y_tr, y_va, y_te = y_dict[tgt]
    plots_dir  = ROOT / cfg["PATHS"].get("plots","reports/plots")
    models_dir = ROOT / cfg["PATHS"].get("models","data/processed/models")
    proc_dir   = ROOT / cfg["PATHS"]["processed_data"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels
    le = LabelEncoder()
    le.fit(STATES)
    joblib.dump(le, models_dir / "next_state_le.pkl")

    # Drop NaN rows
    mask_tr = y_tr.notna(); mask_va = y_va.notna(); mask_te = y_te.notna()
    Xtr, ytr = X_train[mask_tr], le.transform(y_tr[mask_tr].astype(str))
    Xva, yva = X_val[mask_va],   le.transform(y_va[mask_va].astype(str))
    Xte, yte = X_test[mask_te],  le.transform(y_te[mask_te].astype(str))

    n_classes = len(le.classes_)
    print(f"  [next_state] {n_classes} classes, train={len(ytr):,}, val={len(yva):,}, test={len(yte):,}")

    results = {}

    # ── LR baseline ───────────────────────────────────────────────────────
    lr_params = cfg["MODELS"]["logreg"]
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=lr_params["max_iter"],
        class_weight=lr_params["class_weight"],
        C=lr_params["C"],
        random_state=cfg["RANDOM_SEED"],
    )
    lr.fit(Xtr, ytr)
    joblib.dump(lr, models_dir / "next_state_lr.pkl")

    for split_name, Xs, ys in [("val",Xva,yva),("test",Xte,yte)]:
        pred = lr.predict(Xs)
        results[f"LR_{split_name}"] = {
            "macro_f1":    round(f1_score(ys, pred, average="macro",   zero_division=0), 4),
            "weighted_f1": round(f1_score(ys, pred, average="weighted",zero_division=0), 4),
        }

    # ── LightGBM ─────────────────────────────────────────────────────────
    lg_params = cfg["MODELS"]["lgbm"]
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
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
    joblib.dump(lgbm, models_dir / "next_state_lgbm.pkl")

    for split_name, Xs, ys in [("val",Xva,yva),("test",Xte,yte)]:
        pred = lgbm.predict(Xs)
        results[f"LGBM_{split_name}"] = {
            "macro_f1":    round(f1_score(ys, pred, average="macro",   zero_division=0), 4),
            "weighted_f1": round(f1_score(ys, pred, average="weighted",zero_division=0), 4),
        }

    # Confusion matrix on test set
    pred_te = lgbm.predict(Xte)
    cm = confusion_matrix(yte, pred_te, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Next State — Confusion Matrix (LGBM, Test)")
    plt.tight_layout()
    fig.savefig(plots_dir / "next_state_confusion.png", dpi=100)
    plt.close(fig)

    print(f"  [next_state] LR macro-F1 test: {results['LR_test']['macro_f1']:.4f}")
    print(f"  [next_state] LGBM macro-F1 test: {results['LGBM_test']['macro_f1']:.4f}")

    with open(proc_dir / "next_state_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Store for submission
    results["_lgbm_model"]    = lgbm
    results["_lr_model"]      = lr
    results["_le"]            = le
    results["_test_indices"]  = X_test[mask_te].index.tolist()
    results["_pred_test"]     = le.inverse_transform(pred_te).tolist()
    results["_prob_test"]     = lgbm.predict_proba(Xte).tolist()

    return results
