#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features import (
    BASELINE_A_FEATURES,
    DATA_DIR,
    DEFAULT_OUT_DIR,
    RANDOM_STATE,
    build_features,
    evaluate_probs,
    load_payload,
    save_payload,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline A: simple Logistic Regression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train/evaluate baseline A.")
    train_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    train_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    train_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    train_parser.add_argument("--force-features", action="store_true")

    pred_parser = subparsers.add_parser("predict", help="Generate predictions with baseline A.")
    pred_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    pred_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    pred_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    pred_parser.add_argument("--force-features", action="store_true")
    pred_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def train_baseline_a(train_feat: pd.DataFrame, out_dir: Path):
    X = train_feat[BASELINE_A_FEATURES]
    y = train_feat["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_probs(y_val, probs)

    coef = model.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame(
        {"feature": BASELINE_A_FEATURES, "coef": coef}
    ).sort_values("coef", key=np.abs, ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    coef_path = out_dir / "baseline_a_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    payload = load_payload(out_dir)
    payload["metrics"]["baseline_a"] = metrics
    payload["baseline_a_top_coef"] = coef_df.head(5).to_dict("records")
    save_payload(out_dir, payload)
    write_report(out_dir / "reports", payload)

    return metrics, model


def predict_baseline_a(model, test_feat: pd.DataFrame, out_dir: Path, output_path: Optional[Path]):
    probs = model.predict_proba(test_feat[BASELINE_A_FEATURES])[:, 1]
    submission = test_feat[["user_id", "merchant_id"]].copy()
    submission["prob"] = probs

    out_path = output_path or out_dir / "prediction_a.csv"
    submission.to_csv(out_path, index=False)


def main():
    args = parse_args()

    train_feat, test_feat = build_features(
        args.data_dir, args.out_dir, args.chunk_size, args.force_features
    )

    if args.command == "train":
        train_baseline_a(train_feat, args.out_dir)
        return

    metrics, model = train_baseline_a(train_feat, args.out_dir)
    predict_baseline_a(model, test_feat, args.out_dir, args.output)


if __name__ == "__main__":
    main()
