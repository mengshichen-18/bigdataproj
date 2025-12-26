#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

DATA_DIR = Path("dataset/data_format1")
DEFAULT_OUT_DIR = Path("baselines/output")
LAST7_START = 1105
RANDOM_STATE = 42

ACTION_TYPES = {
    0: "click",
    1: "cart",
    2: "buy",
    3: "fav",
}
TOTAL_COLS = [f"total_{name}" for name in ACTION_TYPES.values()]
LAST7_COLS = [f"last7_{name}" for name in ACTION_TYPES.values()]

BASELINE_A_FEATURES = ["last7_buy", "last7_click", "last7_fav"]
CATEGORICAL_FEATURES = ["age_range", "gender"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build cached features.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--chunk-size", default=1_000_000, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_pairs(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path, usecols=["user_id", "merchant_id", "label"])
    test = pd.read_csv(test_path, usecols=["user_id", "merchant_id"])
    pairs = pd.concat(
        [train[["user_id", "merchant_id"]], test[["user_id", "merchant_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    return train, test, pairs


def init_agg(pairs: pd.DataFrame) -> pd.DataFrame:
    idx = pd.MultiIndex.from_frame(pairs[["user_id", "merchant_id"]])
    agg = pd.DataFrame(0, index=idx, columns=TOTAL_COLS + LAST7_COLS, dtype=np.int64)
    agg.index.names = ["user_id", "merchant_id"]
    return agg


def update_counts(agg: pd.DataFrame, grouped: pd.DataFrame, columns):
    if grouped.empty:
        return
    agg.loc[grouped.index, columns] = (
        agg.loc[grouped.index, columns].values + grouped.values
    )


def build_action_counts(log_path: Path, pairs: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    agg = init_agg(pairs)
    pair_df = pairs[["user_id", "merchant_id"]].copy()

    usecols = ["user_id", "seller_id", "time_stamp", "action_type"]
    dtypes = {
        "user_id": "int64",
        "seller_id": "int64",
        "time_stamp": "str",
        "action_type": "int8",
    }

    chunks = pd.read_csv(log_path, usecols=usecols, dtype=dtypes, chunksize=chunk_size)
    for chunk in tqdm(chunks, desc="Scan logs", unit="chunk"):
        chunk = chunk.rename(columns={"seller_id": "merchant_id"})
        chunk = chunk.merge(pair_df, on=["user_id", "merchant_id"], how="inner")
        if chunk.empty:
            continue
        chunk["time_stamp"] = chunk["time_stamp"].astype(int)

        total = (
            chunk.groupby(["user_id", "merchant_id", "action_type"]).size().unstack(fill_value=0)
        )
        total = total.reindex(columns=sorted(ACTION_TYPES), fill_value=0)
        total.columns = TOTAL_COLS
        update_counts(agg, total, TOTAL_COLS)

        last7 = chunk[chunk["time_stamp"] >= LAST7_START]
        if not last7.empty:
            last7_counts = (
                last7.groupby(["user_id", "merchant_id", "action_type"]).size().unstack(fill_value=0)
            )
            last7_counts = last7_counts.reindex(columns=sorted(ACTION_TYPES), fill_value=0)
            last7_counts.columns = LAST7_COLS
            update_counts(agg, last7_counts, LAST7_COLS)

    return agg.reset_index()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_action"] = df[TOTAL_COLS].sum(axis=1)
    df["last7_action"] = df[LAST7_COLS].sum(axis=1)

    df["buy_rate_total"] = df["total_buy"] / (df["total_action"] + 1)
    df["buy_rate_last7"] = df["last7_buy"] / (df["last7_action"] + 1)
    df["recent_activity_ratio"] = df["last7_action"] / (df["total_action"] + 1)
    df["click_to_buy_rate"] = df["total_buy"] / (df["total_click"] + 1)
    df["cart_to_buy_rate"] = df["total_buy"] / (df["total_cart"] + 1)
    df["fav_to_buy_rate"] = df["total_buy"] / (df["total_fav"] + 1)
    df["last7_buy_share"] = df["last7_buy"] / (df["total_buy"] + 1)
    return df


def build_features(data_dir: Path, out_dir: Path, chunk_size: int, force: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_format1.csv"
    test_path = data_dir / "test_format1.csv"
    user_info_path = data_dir / "user_info_format1.csv"
    log_path = data_dir / "user_log_format1.csv"

    train_feat_path = feature_dir / "train_features.parquet"
    test_feat_path = feature_dir / "test_features.parquet"

    if train_feat_path.exists() and test_feat_path.exists() and not force:
        train_feat = pd.read_parquet(train_feat_path)
        test_feat = pd.read_parquet(test_feat_path)
        return train_feat, test_feat

    train, test, pairs = load_pairs(train_path, test_path)
    counts = build_action_counts(log_path, pairs, chunk_size)
    counts = add_derived_features(counts)

    user_info = pd.read_csv(user_info_path, usecols=["user_id", "age_range", "gender"])
    user_info["age_range"] = user_info["age_range"].fillna("unk").astype(str)
    user_info["gender"] = user_info["gender"].fillna("unk").astype(str)

    train_feat = train.merge(counts, on=["user_id", "merchant_id"], how="left")
    train_feat = train_feat.merge(user_info, on="user_id", how="left")
    test_feat = test.merge(counts, on=["user_id", "merchant_id"], how="left")
    test_feat = test_feat.merge(user_info, on="user_id", how="left")

    numeric_cols = get_numeric_features()

    train_feat[numeric_cols] = train_feat[numeric_cols].fillna(0)
    test_feat[numeric_cols] = test_feat[numeric_cols].fillna(0)

    train_feat[CATEGORICAL_FEATURES] = train_feat[CATEGORICAL_FEATURES].fillna("unk").astype(str)
    test_feat[CATEGORICAL_FEATURES] = test_feat[CATEGORICAL_FEATURES].fillna("unk").astype(str)

    train_feat.to_parquet(train_feat_path, index=False)
    test_feat.to_parquet(test_feat_path, index=False)

    return train_feat, test_feat


def get_numeric_features():
    return TOTAL_COLS + LAST7_COLS + [
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


def evaluate_probs(y_true, probs):
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "logloss": float(log_loss(y_true, probs)),
    }


def build_logistic_preprocess(numeric_features, categorical_features):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )


def load_payload(out_dir: Path) -> dict:
    metrics_path = out_dir / "baseline_metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            payload = {"metrics": {}}
    else:
        payload = {"metrics": {}}
    payload.setdefault("metrics", {})
    return payload


def save_payload(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))


def write_report(report_dir: Path, payload: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_report.md"

    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    metrics = payload.get("metrics", {})
    a = metrics.get("baseline_a")
    b_l1 = metrics.get("baseline_b_l1")
    b_l2 = metrics.get("baseline_b_l2")
    c = metrics.get("baseline_c")

    model_name = payload.get("tree_model", "N/A")

    lines = []
    lines.append("# Baseline Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("- Baseline A: Logistic Regression with last-7-day click/buy/fav counts for interpretability.")
    lines.append("- Baseline B: L1/L2 Logistic Regression with richer count/ratio features and demographics for variable selection.")
    lines.append("- Baseline C: Tree model (LightGBM/XGBoost) to capture nonlinearities and feature interactions.")
    lines.append("")

    lines.append("## Metric Comparison")
    lines.append("| Baseline | AUC | Logloss | Notes |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| A (LogReg) | {fmt(a['auc']) if a else 'N/A'} | {fmt(a['logloss']) if a else 'N/A'} | last7 counts only |"
    )
    lines.append(
        f"| B-L1 (LogReg) | {fmt(b_l1['auc']) if b_l1 else 'N/A'} | {fmt(b_l1['logloss']) if b_l1 else 'N/A'} | L1 variable selection |"
    )
    lines.append(
        f"| B-L2 (LogReg) | {fmt(b_l2['auc']) if b_l2 else 'N/A'} | {fmt(b_l2['logloss']) if b_l2 else 'N/A'} | L2 regularization |"
    )
    lines.append(
        f"| C ({model_name}) | {fmt(c['auc']) if c else 'N/A'} | {fmt(c['logloss']) if c else 'N/A'} | tree model |"
    )
    lines.append("")

    lines.append("## Baseline A Notes")
    lines.append("- Features: last7_buy, last7_click, last7_fav.")
    lines.append("- Interpretation: coefficients show marginal effect of recent actions on repeat-buy probability.")
    if payload.get("baseline_a_top_coef"):
        lines.append("")
        lines.append("Top coefficients:")
        lines.append("")
        lines.append("| Feature | Coef |")
        lines.append("| --- | --- |")
        for row in payload["baseline_a_top_coef"]:
            lines.append(f"| {row['feature']} | {row['coef']:.6f} |")
    lines.append("")

    lines.append("## Baseline B Notes")
    lines.append("- Features: full-window counts, last7 counts, ratio features, age/gender (one-hot).")
    lines.append("- L1 focuses on sparse selection; L2 keeps dense weights for overall shrinkage.")
    if payload.get("l1_stability"):
        lines.append("")
        lines.append("Stable L1 features (selection frequency):")
        lines.append("")
        lines.append("| Feature | Selected Runs | Selection Rate |")
        lines.append("| --- | --- | --- |")
        for row in payload["l1_stability"]:
            lines.append(
                f"| {row['feature']} | {row['selected_runs']} | {row['selection_rate']:.2f} |"
            )
    if payload.get("baseline_b_l1_top_coef"):
        lines.append("")
        lines.append("Top L1 coefficients:")
        lines.append("")
        lines.append("| Feature | Coef |")
        lines.append("| --- | --- |")
        for row in payload["baseline_b_l1_top_coef"]:
            lines.append(f"| {row['feature']} | {row['coef']:.6f} |")
    if payload.get("baseline_b_l2_top_coef"):
        lines.append("")
        lines.append("Top L2 coefficients:")
        lines.append("")
        lines.append("| Feature | Coef |")
        lines.append("| --- | --- |")
        for row in payload["baseline_b_l2_top_coef"]:
            lines.append(f"| {row['feature']} | {row['coef']:.6f} |")
    lines.append("")

    lines.append("## Baseline C Notes")
    lines.append("- Tree models capture nonlinear thresholds and interactions without manual feature crosses.")
    lines.append("- They handle sparse one-hot features well via split selection.")
    lines.append("- Each tree layer builds conditional logic, suitable for heterogeneous user behaviors.")
    if payload.get("baseline_c_top_importance"):
        lines.append("")
        lines.append("Top feature importances:")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("| --- | --- |")
        for row in payload["baseline_c_top_importance"]:
            lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
    lines.append("")

    report_path.write_text("\n".join(lines))


def main():
    args = parse_args()
    build_features(args.data_dir, args.out_dir, args.chunk_size, args.force)


if __name__ == "__main__":
    main()
