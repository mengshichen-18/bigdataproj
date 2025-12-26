#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


def load_metrics():
    baseline_path = ROOT / "baselines" / "output" / "baseline_metrics.json"
    ours_path = ROOT / "baselines" / "output" / "ours_v1" / "metrics.json"
    baseline = json.loads(baseline_path.read_text())
    ours = json.loads(ours_path.read_text())
    return baseline["metrics"], ours


def plot_model_compare(metrics, ours):
    labels = ["Baseline A", "Baseline B-L1", "Baseline B-L2", "Baseline C", "Ours-v1"]
    auc_vals = [
        metrics["baseline_a"]["auc"],
        metrics["baseline_b_l1"]["auc"],
        metrics["baseline_b_l2"]["auc"],
        metrics["baseline_c"]["auc"],
        ours["oof_auc"],
    ]
    logloss_vals = [
        metrics["baseline_a"]["logloss"],
        metrics["baseline_b_l1"]["logloss"],
        metrics["baseline_b_l2"]["logloss"],
        metrics["baseline_c"]["logloss"],
        ours["oof_logloss"],
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, auc_vals, color="#2c7fb8")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, max(auc_vals) + 0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Model Comparison (OOF AUC)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_compare_auc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, logloss_vals, color="#2c7fb8")
    ax.set_ylabel("Logloss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Model Comparison (OOF Logloss)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_compare_logloss.png", dpi=200)
    plt.close(fig)


def find_baseline_c_importance():
    candidates = list((ROOT / "baselines" / "output").glob("baseline_c_*_importance.csv"))
    if not candidates:
        return None
    return candidates[0]


def plot_feature_importance():
    baseline_path = find_baseline_c_importance()
    ours_path = ROOT / "baselines" / "output" / "ours_v1" / "feature_importance.csv"
    if baseline_path is None or not ours_path.exists():
        print("Skip feature importance: missing baseline or ours importance file.")
        return

    base_df = pd.read_csv(baseline_path)
    base_metric = "gain" if "gain" in base_df.columns else "importance"
    base_df = base_df.sort_values(base_metric, ascending=False).head(15)

    ours_df = pd.read_csv(ours_path).sort_values("gain", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(base_df["feature"][::-1], base_df[base_metric][::-1], color="#756bb1")
    axes[0].set_title("Baseline C Top Features")
    axes[0].set_xlabel("Gain" if base_metric == "gain" else "Importance")

    axes[1].barh(ours_df["feature"][::-1], ours_df["gain"][::-1], color="#31a354")
    axes[1].set_title("Ours-v1 Top Features")
    axes[1].set_xlabel("Gain")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_compare.png", dpi=200)
    plt.close(fig)


def plot_ours_roc():
    oof_path = ROOT / "baselines" / "output" / "ours_v1" / "oof_predictions.csv"
    if not oof_path.exists():
        print("Skip ROC: missing ours OOF predictions.")
        return

    df = pd.read_csv(oof_path)
    y_true = df["label"].to_numpy()
    y_score = df["oof_prob"].to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#2c7fb8", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ours-v1 ROC (OOF)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ours_roc.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, ours = load_metrics()
    plot_model_compare(metrics, ours)
    plot_feature_importance()
    plot_ours_roc()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
