"""
calibration.py
==============
Probability calibration + reliability diagrams for binary models.

Generates reliability diagrams comparing uncalibrated vs. calibrated
probabilities and saves them to reports/plots/.
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import joblib
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    p = cfg_path or ROOT / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def reliability_diagram(y_true, y_prob_raw, y_prob_cal, target: str, plots_dir: Path):
    """Plot reliability (calibration) diagrams for raw vs. calibrated probs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_bins = 10

    for ax, (probs, label) in zip(axes, [
        (y_prob_raw, "Uncalibrated"),
        (y_prob_cal, "Calibrated (Isotonic)"),
    ]):
        try:
            prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
            ax.plot(prob_pred, prob_true, "s-", label=label, color="#2563EB")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"Reliability Diagram — {target}\n{label}")
            ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

    plt.tight_layout()
    out = plots_dir / f"{target}_calibration.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def run(X_val, y_dict, binary_results, cfg):
    """Generate calibration plots for all binary targets."""
    plots_dir  = ROOT / cfg["PATHS"].get("plots","reports/plots")
    models_dir = ROOT / cfg["PATHS"].get("models","data/processed/models")
    plots_dir.mkdir(parents=True, exist_ok=True)

    from src.models.delinquency_default_prepay import BINARY_TARGETS

    print("\n[calibration] Generating reliability diagrams …")
    cal_outputs = {}

    for tgt in BINARY_TARGETS:
        if tgt not in y_dict:
            continue
        _, y_va, _ = y_dict[tgt]
        mask_va    = y_va.notna()
        ys         = y_va[mask_va].astype(int)
        Xv         = X_val[mask_va]

        try:
            lr_path   = models_dir / f"{tgt}_lr.pkl"
            cal_path  = models_dir / f"{tgt}_lgbm_cal.pkl"
            if not lr_path.exists() or not cal_path.exists():
                continue
            lr  = joblib.load(lr_path)
            cal = joblib.load(cal_path)
            lgbm_raw = joblib.load(models_dir / f"{tgt}_lgbm_raw.pkl")
            prob_raw = lgbm_raw.predict_proba(Xv)[:, 1]
            prob_cal = cal.predict_proba(Xv)[:, 1]
            out = reliability_diagram(ys.values, prob_raw, prob_cal, tgt, plots_dir)
            cal_outputs[tgt] = str(out)
            print(f"  ✓ Reliability diagram: {out.name}")
        except Exception as e:
            print(f"  ⚠ Calibration skipped for {tgt}: {e}")

    return cal_outputs
