#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from build_features import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DEFAULT_OUT_DIR,
    RANDOM_STATE,
    build_features,
    evaluate_probs,
    get_numeric_features,
    load_payload,
    save_payload,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline C: Tree model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train/evaluate baseline C.")
    train_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    train_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    train_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    train_parser.add_argument("--force-features", action="store_true")
    train_parser.add_argument("--tree-model", choices=["lightgbm", "xgboost"], required=True)

    pred_parser = subparsers.add_parser("predict", help="Generate predictions with baseline C.")
    pred_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    pred_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    pred_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    pred_parser.add_argument("--force-features", action="store_true")
    pred_parser.add_argument("--tree-model", choices=["lightgbm", "xgboost"], required=True)
    pred_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def build_tree_model(model_name: str, scale_pos_weight: float):
    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError("model_name must be 'lightgbm' or 'xgboost'")


def train_baseline_c(train_feat: pd.DataFrame, out_dir: Path, model_name: str):
    numeric_features = get_numeric_features()
    X = train_feat[numeric_features + CATEGORICAL_FEATURES]
    y = train_feat["label"].values
    groups = train_feat["user_id"].values

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in cv.split(X, y, groups):
        preprocess = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )

        X_train_t = preprocess.fit_transform(X.iloc[tr_idx])
        X_val_t = preprocess.transform(X.iloc[va_idx])

        pos = y[tr_idx].sum()
        neg = len(tr_idx) - pos
        scale_pos_weight = neg / max(pos, 1)

        model = build_tree_model(model_name, scale_pos_weight)
        model.fit(X_train_t, y[tr_idx])
        oof[va_idx] = model.predict_proba(X_val_t)[:, 1]

    metrics = evaluate_probs(y, oof)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    X_all_t = preprocess.fit_transform(X)
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / max(pos, 1)
    model = build_tree_model(model_name, scale_pos_weight)
    model.fit(X_all_t, y)

    feature_names = preprocess.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_path = out_dir / f"baseline_c_{model_name}_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    payload = load_payload(out_dir)
    payload["metrics"]["baseline_c"] = metrics
    payload["baseline_c_top_importance"] = importance_df.head(10).to_dict("records")
    payload["tree_model"] = model_name
    save_payload(out_dir, payload)
    write_report(out_dir / "reports", payload)

    return metrics, (model, preprocess), importance_df


def predict_baseline_c(model_bundle, test_feat: pd.DataFrame, out_dir: Path, output_path: Optional[Path]):
    model, preprocess = model_bundle
    numeric_features = get_numeric_features()
    X_test = test_feat[numeric_features + CATEGORICAL_FEATURES]
    X_test_t = preprocess.transform(X_test)
    probs = model.predict_proba(X_test_t)[:, 1]

    submission = test_feat[["user_id", "merchant_id"]].copy()
    submission["prob"] = probs

    out_path = output_path or out_dir / "prediction_c.csv"
    submission.to_csv(out_path, index=False)


def main():
    args = parse_args()
    train_feat, test_feat = build_features(
        args.data_dir, args.out_dir, args.chunk_size, args.force_features
    )

    if args.command == "train":
        train_baseline_c(train_feat, args.out_dir, args.tree_model)
        return

    _, model_bundle, _ = train_baseline_c(train_feat, args.out_dir, args.tree_model)
    predict_baseline_c(model_bundle, test_feat, args.out_dir, args.output)


if __name__ == "__main__":
    main()
