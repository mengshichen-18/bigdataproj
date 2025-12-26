#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from ours_v1_build_features import build_features
from ours_v1_utils import CATEGORICAL_FEATURES, DATA_DIR, DEFAULT_OUT_DIR, RANDOM_STATE


def parse_args():
    parser = argparse.ArgumentParser(description="Ours v1: train with GroupKFold + target encoding.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--chunk-size", default=1_000_000, type=int)
    parser.add_argument("--force-features", action="store_true")

    parser.add_argument("--model", choices=["xgboost", "lightgbm", "hgb"], default="xgboost")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)

    parser.add_argument("--te-prior", type=float, default=20.0)

    parser.add_argument("--early-stop-metric", choices=["auc", "logloss"], default="auc")
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--verbose-eval", type=int, default=200)
    return parser.parse_args()


def _clip_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(probs, eps, 1 - eps)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def one_hot_cat(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str]):
    combined = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0, ignore_index=True)
    combined = combined.fillna("unk").astype(str)
    dummies = pd.get_dummies(combined, columns=cat_cols, drop_first=False)
    train_d = dummies.iloc[: len(train_df)].to_numpy(dtype=np.float32)
    test_d = dummies.iloc[len(train_df) :].to_numpy(dtype=np.float32)
    return train_d, test_d, list(dummies.columns)


def target_encode_merchant(
    merchant_train: pd.Series,
    y_train: np.ndarray,
    merchant_val: pd.Series,
    merchant_test: pd.Series,
    prior: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"merchant_id": merchant_train.values, "y": y_train})
    stats = df.groupby("merchant_id")["y"].agg(["count", "sum"])
    global_mean = float(df["y"].mean())

    cnt = stats["count"]
    summ = stats["sum"]

    te_map = (summ + global_mean * prior) / (cnt + prior)
    cnt_map = cnt

    # Leave-one-out for training rows (avoid using the row's own label).
    sum_row = merchant_train.map(summ).to_numpy(dtype=np.float32)
    cnt_row = merchant_train.map(cnt).to_numpy(dtype=np.float32)

    denom = cnt_row - 1.0 + prior
    te_train = (sum_row - y_train + global_mean * prior) / np.where(denom > 0, denom, 1.0)
    te_train = np.where(cnt_row > 1.0, te_train, global_mean).astype(np.float32)

    te_val = merchant_val.map(te_map).fillna(global_mean).to_numpy(dtype=np.float32)
    te_test = merchant_test.map(te_map).fillna(global_mean).to_numpy(dtype=np.float32)

    cnt_train = np.log1p(cnt_row).astype(np.float32)
    cnt_val = np.log1p(merchant_val.map(cnt_map).fillna(0).to_numpy(dtype=np.float32)).astype(
        np.float32
    )
    cnt_test = np.log1p(merchant_test.map(cnt_map).fillna(0).to_numpy(dtype=np.float32)).astype(
        np.float32
    )

    return te_train, te_val, te_test, cnt_train, cnt_val, cnt_test


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: int,
    early_stop_metric: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is not installed; try `pip install xgboost`") from e

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1.0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": early_stop_metric,
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": seed,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        maximize=(early_stop_metric == "auc"),
        verbose_eval=verbose_eval,
    )
    best_it = int(getattr(booster, "best_iteration", num_boost_round - 1))

    val_pred = booster.predict(dvalid, iteration_range=(0, best_it + 1))
    test_pred = booster.predict(dtest, iteration_range=(0, best_it + 1))

    importance_gain = booster.get_score(importance_type="gain")
    importance_weight = booster.get_score(importance_type="weight")
    return (
        val_pred.astype(np.float32),
        test_pred.astype(np.float32),
        {k: float(v) for k, v in importance_gain.items()},
        {k: float(v) for k, v in importance_weight.items()},
    )


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    early_stop_metric: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is not installed; try `pip install lightgbm`") from e

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1.0)

    model = lgb.LGBMClassifier(
        n_estimators=num_boost_round,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc" if early_stop_metric == "auc" else "binary_logloss",
        callbacks=[
            lgb.early_stopping(
                early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )

    best_iteration = int(getattr(model, "best_iteration_", 0) or num_boost_round)
    val_pred = model.predict_proba(X_val, num_iteration=best_iteration)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1].astype(np.float32)

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    imp_gain = dict(zip(feature_names, gain.astype(float)))
    imp_split = dict(zip(feature_names, split.astype(float)))

    return val_pred, test_pred, imp_gain, imp_split


def train_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    val_pred = model.predict_proba(X_val)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)
    return val_pred, test_pred


def main():
    args = parse_args()
    out_dir = args.out_dir / "ours_v1"
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "reports")

    if args.model == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "xgboost is not installed. Install it with `pip install xgboost` "
                "or run with `--model hgb`."
            ) from e
    if args.model == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "lightgbm is not installed. Install it with `pip install lightgbm` "
                "or run with `--model hgb`."
            ) from e

    train_feat_path, test_feat_path = build_features(
        args.data_dir, args.out_dir, args.chunk_size, args.force_features
    )
    train_df = pd.read_parquet(train_feat_path)
    test_df = pd.read_parquet(test_feat_path)

    y = train_df["label"].to_numpy(dtype=np.int8)
    groups = train_df["user_id"].to_numpy(dtype=np.int64)

    cat_train, cat_test, cat_feature_names = one_hot_cat(train_df, test_df, CATEGORICAL_FEATURES)

    drop_cols = {"label", "user_id", "merchant_id", *CATEGORICAL_FEATURES}
    num_cols = [c for c in train_df.columns if c not in drop_cols]
    train_num = train_df[num_cols].to_numpy(dtype=np.float32)
    test_num = test_df[num_cols].to_numpy(dtype=np.float32)

    base_train = np.concatenate([train_num, cat_train], axis=1)
    base_test = np.concatenate([test_num, cat_test], axis=1)
    base_feature_names = num_cols + cat_feature_names

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)

    fold_metrics = []
    importance_gain_total: Dict[str, float] = {}
    importance_weight_total: Dict[str, float] = {}

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(cv.split(base_train, y, groups))

    for fold, (tr_idx, va_idx) in enumerate(tqdm(splits, desc="CV folds", unit="fold"), start=1):
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)

        te_tr, te_va, te_te, cnt_tr, cnt_va, cnt_te = target_encode_merchant(
            merchant_train=train_df.iloc[tr_idx]["merchant_id"],
            y_train=y_tr,
            merchant_val=train_df.iloc[va_idx]["merchant_id"],
            merchant_test=test_df["merchant_id"],
            prior=args.te_prior,
        )

        X_tr = np.concatenate(
            [base_train[tr_idx], te_tr.reshape(-1, 1), cnt_tr.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        X_va = np.concatenate(
            [base_train[va_idx], te_va.reshape(-1, 1), cnt_va.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        X_te = np.concatenate(
            [base_test, te_te.reshape(-1, 1), cnt_te.reshape(-1, 1)], axis=1
        ).astype(np.float32)

        feature_names = base_feature_names + ["te_merchant", "te_merchant_log1p_count"]

        if args.model == "xgboost":
            va_pred, te_pred, imp_gain, imp_weight = train_xgboost(
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                feature_names,
                seed=args.seed + fold,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                verbose_eval=args.verbose_eval,
                early_stop_metric=args.early_stop_metric,
            )
        elif args.model == "lightgbm":
            va_pred, te_pred, imp_gain, imp_weight = train_lightgbm(
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                feature_names,
                seed=args.seed + fold,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                early_stop_metric=args.early_stop_metric,
            )
        else:
            va_pred, te_pred = train_hgb(
                X_tr, y_tr.astype(np.int8), X_va, y_va.astype(np.int8), X_te, seed=args.seed + fold
            )
            imp_gain, imp_weight = {}, {}

        oof[va_idx] = va_pred
        test_pred += te_pred / args.n_splits

        fold_auc = float(roc_auc_score(y_va, va_pred))
        fold_logloss = float(log_loss(y_va, _clip_probs(va_pred)))
        fold_metrics.append({"fold": fold, "auc": fold_auc, "logloss": fold_logloss})

        for k, v in imp_gain.items():
            importance_gain_total[k] = importance_gain_total.get(k, 0.0) + v
        for k, v in imp_weight.items():
            importance_weight_total[k] = importance_weight_total.get(k, 0.0) + v

    overall_auc = float(roc_auc_score(y, oof))
    overall_logloss = float(log_loss(y, _clip_probs(oof)))

    metrics = {
        "model": args.model,
        "n_splits": args.n_splits,
        "te_prior": args.te_prior,
        "oof_auc": overall_auc,
        "oof_logloss": overall_logloss,
        "folds": fold_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    oof_df = train_df[["user_id", "merchant_id", "label"]].copy()
    oof_df["oof_prob"] = oof
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    submission = test_df[["user_id", "merchant_id"]].copy()
    submission["prob"] = test_pred
    submission.to_csv(out_dir / "prediction.csv", index=False)

    if importance_gain_total:
        imp = pd.DataFrame(
            {
                "feature": list(importance_gain_total.keys()),
                "gain": list(importance_gain_total.values()),
                "weight": [importance_weight_total.get(k, 0.0) for k in importance_gain_total.keys()],
            }
        ).sort_values("gain", ascending=False)
        imp.to_csv(out_dir / "feature_importance.csv", index=False)

    report_lines = []
    report_lines.append("# Ours v1 Report")
    report_lines.append("")
    report_lines.append(f"- Model: `{metrics['model']}`")
    report_lines.append(f"- OOF AUC: `{overall_auc:.6f}`")
    report_lines.append(f"- OOF Logloss: `{overall_logloss:.6f}`")
    report_lines.append(f"- CV: StratifiedGroupKFold(user_id), n_splits={args.n_splits}")
    report_lines.append(f"- Target Encoding: merchant_id with prior={args.te_prior} (train LOO)")
    report_lines.append("")
    report_lines.append("## Fold Metrics")
    report_lines.append("| Fold | AUC | Logloss |")
    report_lines.append("| --- | ---: | ---: |")
    for row in fold_metrics:
        report_lines.append(f"| {row['fold']} | {row['auc']:.6f} | {row['logloss']:.6f} |")
    report_lines.append("")

    if (out_dir / "feature_importance.csv").exists():
        report_lines.append("## Top Features (gain)")
        report_lines.append("")
        top_imp = pd.read_csv(out_dir / "feature_importance.csv").head(20)
        report_lines.append("| Feature | Gain | Weight |")
        report_lines.append("| --- | ---: | ---: |")
        for _, r in top_imp.iterrows():
            report_lines.append(f"| {r['feature']} | {float(r['gain']):.6f} | {float(r['weight']):.0f} |")
        report_lines.append("")

    (out_dir / "reports" / "report.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
