"""
explainability.py
=================
Task 6 — Explainability Layer

Generates:
  - Global SHAP summary plots for all trained models
  - Local SHAP force/waterfall plots for representative loans
    (TP, FP, FN per target)
  - False-positive / false-negative analysis
  - Prediction uncertainty report
  - reports/explainability_report.md
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


def compute_shap(model, X: pd.DataFrame, plots_dir: Path, tag: str) -> pd.Series:
    """
    Compute SHAP values for the given model and feature matrix.
    Saves summary bar plot. Returns mean |SHAP| per feature.
    """
    import shap
    try:
        # Try TreeExplainer (fast for LGBM)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary: shap_values may be [neg_class, pos_class]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv = shap_values if not isinstance(shap_values, list) else shap_values[0]

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {tag}")
        plt.tight_layout()
        fig.savefig(plots_dir / f"shap_summary_{tag}.png", dpi=100, bbox_inches="tight")
        plt.close("all")

        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
        return mean_abs, sv, explainer

    except Exception as e:
        print(f"    ⚠ SHAP failed for {tag}: {e}")
        return pd.Series(dtype=float), None, None


def fp_fn_analysis(y_true: pd.Series, y_prob: np.ndarray,
                   df_raw: pd.DataFrame, target: str) -> str:
    """
    Analyse false positives and false negatives.
    Returns markdown text summarising findings.
    """
    y_pred = (y_prob > 0.5).astype(int)
    y_true_np = y_true.astype(int).values

    fp_mask = (y_pred == 1) & (y_true_np == 0)
    fn_mask = (y_pred == 0) & (y_true_np == 1)

    lines = [f"\n### FP/FN Analysis — {target}\n"]
    lines.append(f"- **False Positives**: {fp_mask.sum():,} ({100*fp_mask.mean():.1f}% of predictions)")
    lines.append(f"- **False Negatives**: {fn_mask.sum():,} ({100*fn_mask.mean():.1f}% of predictions)\n")

    for error_type, mask, label in [
        (fp_mask, df_raw.copy() if len(df_raw) else None, "False Positives"),
        (fn_mask, df_raw.copy() if len(df_raw) else None, "False Negatives"),
    ]:
        if mask is None or not any(error_type): continue
        sub = df_raw[error_type]
        lines.append(f"**{label} characteristics:**")
        for col in ["credit_score_band","ltv_band","days_past_due","loan_age_months"]:
            if col in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[col]):
                    lines.append(f"  - `{col}` mean: {sub[col].mean():.2f}")
                else:
                    lines.append(f"  - `{col}` mode: {sub[col].mode().iloc[0] if len(sub[col].mode()) else 'N/A'}")
        lines.append("")

    return "\n".join(lines)


def run(X_train, X_val, X_test, y_dict, df_test_raw, binary_results, cfg) -> str:
    """
    Main explainability runner.
    Returns path to explainability report.
    """
    plots_dir  = ROOT / cfg["PATHS"].get("plots","reports/plots")
    models_dir = ROOT / cfg["PATHS"].get("models","data/processed/models")
    reports    = ROOT / cfg["PATHS"]["reports"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Explainability Report\n"]
    lines.append(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    print("\n[explain] Computing SHAP explanations …")

    for tgt in BINARY_TARGETS:
        cal_path = models_dir / f"{tgt}_lgbm_cal.pkl"
        if not cal_path.exists():
            continue

        cal = joblib.load(cal_path)
        # Get base LGBM from saved raw model
        lgbm_base = joblib.load(models_dir / f"{tgt}_lgbm_raw.pkl")

        y_tr, y_va, y_te = y_dict.get(tgt, (None,None,None))
        if y_te is None: continue
        mask_te = y_te.notna()
        Xte = X_test[mask_te]
        yte = y_te[mask_te].astype(int)

        lines.append(f"\n## {tgt}\n")

        # ── Global SHAP ───────────────────────────────────────────────────
        sample_size = min(2000, len(Xte))
        Xs = Xte.sample(n=sample_size, random_state=cfg["RANDOM_SEED"])
        mean_shap, sv, explainer = compute_shap(lgbm_base, Xs, plots_dir, tgt)

        if mean_shap is not None and len(mean_shap):
            lines.append("### Global Feature Importance (mean |SHAP|)\n")
            top10 = mean_shap.sort_values(ascending=False).head(10)
            imp_df = pd.DataFrame({"Feature": top10.index, "Mean_|SHAP|": top10.values.round(5)})
            lines.append(imp_df.to_markdown(index=False) + "\n")
            lines.append(f"![SHAP Summary](plots/shap_summary_{tgt}.png)\n")

        # ── Local SHAP for TP, FP, FN ─────────────────────────────────────
        if sv is not None:
            try:
                prob_te = cal.predict_proba(Xte)[:, 1]
                pred_te = (prob_te > 0.5).astype(int)
                yte_np  = yte.values

                # Find example indices
                tp_idx = np.where((pred_te == 1) & (yte_np == 1))[0]
                fp_idx = np.where((pred_te == 1) & (yte_np == 0))[0]
                fn_idx = np.where((pred_te == 0) & (yte_np == 1))[0]

                import shap
                for case_type, idx_arr in [("TP",tp_idx),("FP",fp_idx),("FN",fn_idx)]:
                    if len(idx_arr) == 0: continue
                    pick = idx_arr[0]

                    # Waterfall plot for this record
                    pick_sv = sv[pick] if pick < len(sv) else sv[0]
                    feature_names = list(Xs.columns)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sorted_idx = np.argsort(np.abs(pick_sv))[::-1][:10]
                    y_pos = np.arange(len(sorted_idx))
                    ax.barh(y_pos, pick_sv[sorted_idx],
                            color=["#DC2626" if v > 0 else "#2563EB" for v in pick_sv[sorted_idx]])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
                    ax.set_xlabel("SHAP value")
                    ax.set_title(f"Local SHAP ({case_type}) — {tgt}")
                    ax.axvline(0, color="black", linewidth=0.5)
                    plt.tight_layout()
                    fname = f"shap_local_{tgt}_{case_type}.png"
                    fig.savefig(plots_dir / fname, dpi=100)
                    plt.close(fig)
                    lines.append(f"![Local SHAP {case_type}](plots/{fname})\n")

                # FP/FN analysis
                dfr = df_test_raw[mask_te].reset_index(drop=True) if df_test_raw is not None else pd.DataFrame()
                lines.append(fp_fn_analysis(yte, prob_te, dfr, tgt))

            except Exception as e:
                lines.append(f"*Local SHAP skipped: {e}*\n")

        # ── Uncertainty / confidence ──────────────────────────────────────
        lines.append("### Prediction Uncertainty\n")
        try:
            proba = cal.predict_proba(Xte)
            # Use prediction entropy as uncertainty proxy
            eps = 1e-10
            entropy = -(proba * np.log(proba + eps)).sum(axis=1)
            lines.append(f"Mean prediction entropy: **{entropy.mean():.4f}** "
                         f"(higher = more uncertain)\n")
            lines.append(f"Fraction of records with entropy > 0.5: "
                         f"**{(entropy > 0.5).mean():.3f}**\n")
        except Exception as e:
            lines.append(f"*Uncertainty estimation failed: {e}*\n")

    # ── Next state SHAP ───────────────────────────────────────────────────
    ns_path = models_dir / "next_state_lgbm.pkl"
    if ns_path.exists():
        lines.append("\n## next_state (Multiclass)\n")
        ns_model = joblib.load(ns_path)
        Xs_ns = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        mean_shap_ns, _, _ = compute_shap(ns_model, Xs_ns, plots_dir, "next_state")
        if mean_shap_ns is not None and len(mean_shap_ns):
            lines.append("### Global Feature Importance\n")
            top10 = mean_shap_ns.sort_values(ascending=False).head(10)
            imp_df = pd.DataFrame({"Feature": top10.index, "Mean_|SHAP|": top10.values.round(5)})
            lines.append(imp_df.to_markdown(index=False) + "\n")
            lines.append(f"![SHAP next_state](plots/shap_summary_next_state.png)\n")

    report_text = "\n".join(lines)
    (reports / "explainability_report.md").write_text(report_text, encoding="utf-8")
    print(f"  ✓ explainability_report.md written")
    return str(reports / "explainability_report.md")
