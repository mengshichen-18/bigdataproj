#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from build_features import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DEFAULT_OUT_DIR,
    RANDOM_STATE,
    build_features,
    build_logistic_preprocess,
    evaluate_probs,
    get_numeric_features,
    load_payload,
    save_payload,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline B: L1/L2 Logistic Regression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train/evaluate baseline B.")
    train_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    train_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    train_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    train_parser.add_argument("--force-features", action="store_true")
    train_parser.add_argument("--mode", choices=["both", "l1", "l2"], default="both")
    train_parser.add_argument("--l1-runs", default=3, type=int)

    pred_parser = subparsers.add_parser("predict", help="Generate predictions with baseline B.")
    pred_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    pred_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    pred_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    pred_parser.add_argument("--force-features", action="store_true")
    pred_parser.add_argument("--penalty", choices=["l1", "l2"], default="l2")
    pred_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def train_logistic(train_feat: pd.DataFrame, penalty: str, out_dir: Path):
    numeric_features = get_numeric_features()
    X = train_feat[numeric_features + CATEGORICAL_FEATURES]
    y = train_feat["label"].values
    groups = train_feat["user_id"].values

    def build_model():
        preprocess = build_logistic_preprocess(numeric_features, CATEGORICAL_FEATURES)
        clf = LogisticRegression(
            penalty=penalty,
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        )
        return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in cv.split(X, y, groups):
        model = build_model()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

    metrics = evaluate_probs(y, oof)

    model = build_model()
    model.fit(X, y)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)

    coef_path = out_dir / f"baseline_b_{penalty}_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    return metrics, model, coef_df


def l1_stability(train_feat: pd.DataFrame, runs: int):
    numeric_features = get_numeric_features()
    X = train_feat[numeric_features + CATEGORICAL_FEATURES]
    y = train_feat["label"].values

    counts = Counter()
    metrics_list = []

    for i in tqdm(range(runs), desc="L1 stability", unit="run"):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE + i, stratify=y
        )
        preprocess = build_logistic_preprocess(numeric_features, CATEGORICAL_FEATURES)
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        )
        model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        metrics_list.append(evaluate_probs(y_val, probs))

        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        coef = model.named_steps["clf"].coef_.ravel()
        selected = feature_names[np.abs(coef) > 1e-6]
        counts.update(selected)

    stability = pd.DataFrame(
        {
            "feature": list(counts.keys()),
            "selected_runs": list(counts.values()),
        }
    )
    stability["selection_rate"] = stability["selected_runs"] / runs
    stability = stability.sort_values(
        ["selection_rate", "selected_runs"], ascending=False
    )

    avg_metrics = {
        "auc": float(np.mean([m["auc"] for m in metrics_list])),
        "logloss": float(np.mean([m["logloss"] for m in metrics_list])),
    }

    return stability, avg_metrics


def train_baseline_b(train_feat: pd.DataFrame, out_dir: Path, mode: str, l1_runs: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload(out_dir)
    metrics = payload["metrics"]

    penalties = ["l1", "l2"] if mode == "both" else [mode]
    for penalty in tqdm(penalties, desc="Train penalties", unit="model"):
        metric, model, coef_df = train_logistic(train_feat, penalty, out_dir)
        metrics[f"baseline_b_{penalty}"] = metric
        payload[f"baseline_b_{penalty}_top_coef"] = coef_df.head(8).to_dict("records")

    if "l1" in penalties and l1_runs > 1:
        stability_df, stability_metrics = l1_stability(train_feat, l1_runs)
        stability_path = out_dir / "baseline_b_l1_stability.csv"
        stability_df.to_csv(stability_path, index=False)
        payload["l1_stability"] = stability_df.head(10).to_dict("records")
        metrics["baseline_b_l1_avg"] = stability_metrics

    save_payload(out_dir, payload)
    write_report(out_dir / "reports", payload)


def predict_baseline_b(
    model, test_feat: pd.DataFrame, out_dir: Path, output_path: Optional[Path], penalty: str
):
    numeric_features = get_numeric_features()
    X_test = test_feat[numeric_features + CATEGORICAL_FEATURES]
    probs = model.predict_proba(X_test)[:, 1]

    submission = test_feat[["user_id", "merchant_id"]].copy()
    submission["prob"] = probs

    out_path = output_path or out_dir / f"prediction_b-{penalty}.csv"
    submission.to_csv(out_path, index=False)


def main():
    args = parse_args()
    train_feat, test_feat = build_features(
        args.data_dir, args.out_dir, args.chunk_size, args.force_features
    )

    if args.command == "train":
        train_baseline_b(train_feat, args.out_dir, args.mode, args.l1_runs)
        return

    metric, model, _ = train_logistic(train_feat, args.penalty, args.out_dir)
    predict_baseline_b(model, test_feat, args.out_dir, args.output, args.penalty)


if __name__ == "__main__":
    main()
