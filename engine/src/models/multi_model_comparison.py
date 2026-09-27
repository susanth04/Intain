"""Train comparable binary model families on the engine's time-aware splits."""

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.load_and_validate import load_all
from src.data.splits import make_splits
from src.features.build_features import build_features

warnings.filterwarnings("ignore")
TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]


def metrics(y_true, probabilities, predictions):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def train_models(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    models = {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=seed),
        "RF": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            objective="binary:logistic", eval_metric="logloss", n_jobs=-1,
            tree_method="hist", random_state=seed,
        ),
        "LGBM": lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", verbosity=-1, random_state=seed,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        for split, features, labels in (("val", X_val, y_val), ("test", X_test, y_test)):
            probabilities = model.predict_proba(features)[:, 1]
            results[f"{name}_{split}"] = metrics(labels, probabilities, (probabilities >= 0.5).astype(int))
    return results


def main():
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    datasets = load_all(cfg)
    panel = pd.concat([datasets["train"], datasets["test"]], ignore_index=True)
    train, validation, test = make_splits(panel, cfg)
    X_train, X_val, X_test, targets, *_ = build_features(train, validation, test, cfg)

    comparison = {}
    winners = {}
    for target in TARGETS:
        y_train, y_val, y_test = targets[target]
        masks = (y_train.notna(), y_val.notna(), y_test.notna())
        result = train_models(
            X_train[masks[0]], y_train[masks[0]].astype(int),
            X_val[masks[1]], y_val[masks[1]].astype(int),
            X_test[masks[2]], y_test[masks[2]].astype(int), cfg["RANDOM_SEED"],
        )
        comparison[target] = result
        test_results = {name: values for name, values in result.items() if name.endswith("_test")}
        winner_name, winner_metrics = max(test_results.items(), key=lambda item: item[1]["roc_auc"])
        winners[target] = {
            "model": winner_name.removesuffix("_test"),
            "metric": "roc_auc",
            "test_roc_auc": winner_metrics["roc_auc"],
            "pr_auc": winner_metrics["pr_auc"],
            "f1": winner_metrics["f1"],
            "brier": winner_metrics["brier"],
            "models_compared": [name.removesuffix("_test") for name in test_results],
        }
        print(f"{target}: {winners[target]['model']} ({winner_metrics['roc_auc']:.4f} ROC-AUC)")

    output_dir = ROOT / cfg["PATHS"]["processed_data"]
    (output_dir / "model_comparison_results.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "model_winners.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    rows = []
    for target, result in comparison.items():
        for key, values in result.items():
            if key.endswith("_test"):
                rows.append({"target": target, "model": key.removesuffix("_test"), **values})
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
