#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baselines with StratifiedGroupKFold(user_id).")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=Path("./output/features/train_features.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("./output/baseline_groupcv.json"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def main():
    args = parse_args()
    df = pd.read_parquet(args.train_features)
    y = df["label"].to_numpy(dtype=np.int8)
    groups = df["user_id"].to_numpy(dtype=np.int64)

    baseline_a_features = ["last7_buy", "last7_click", "last7_fav"]

    total_cols = ["total_click", "total_cart", "total_buy", "total_fav"]
    last7_cols = ["last7_click", "last7_cart", "last7_buy", "last7_fav"]
    numeric_b = total_cols + last7_cols + [
        "total_action",
        "last7_action",
        "buy_rate_total",
        "buy_rate_last7",
        "recent_activity_ratio",
        "click_to_buy_rate",
        "cart_to_buy_rate",
        "fav_to_buy_rate",
        "last7_buy_share",
    ]
    categorical_b = ["age_range", "gender"]

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # Baseline A
    model_a = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
        ]
    )
    oof_a = np.zeros(len(df), dtype=np.float32)
    for tr, va in tqdm(list(cv.split(df, y, groups)), desc="Baseline A", unit="fold"):
        model_a.fit(df.iloc[tr][baseline_a_features], y[tr])
        oof_a[va] = model_a.predict_proba(df.iloc[va][baseline_a_features])[:, 1]

    # Baseline B (L2)
    preprocess_b = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_b,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_b,
            ),
        ]
    )
    model_b = Pipeline(
        [
            ("pre", preprocess_b),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
        ]
    )
    oof_b = np.zeros(len(df), dtype=np.float32)
    for tr, va in tqdm(list(cv.split(df, y, groups)), desc="Baseline B-L2", unit="fold"):
        model_b.fit(df.iloc[tr][numeric_b + categorical_b], y[tr])
        oof_b[va] = model_b.predict_proba(df.iloc[va][numeric_b + categorical_b])[:, 1]

    out = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "baseline_a": {
            "oof_auc": float(roc_auc_score(y, oof_a)),
            "oof_logloss": float(log_loss(y, clip_probs(oof_a))),
        },
        "baseline_b_l2": {
            "oof_auc": float(roc_auc_score(y, oof_b)),
            "oof_logloss": float(log_loss(y, clip_probs(oof_b))),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

