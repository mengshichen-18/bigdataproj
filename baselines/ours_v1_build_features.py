#!/usr/bin/env python3
from typing import Union

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ours_v1_utils import (
    ACTION_TYPES,
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DEFAULT_OUT_DIR,
    RECENCY_DEFAULT,
    WINDOW_LABELS,
    compute_recency,
    compute_window_counts,
    expected_count_columns,
    mmdd_to_days_before_1111,
    safe_div,
    update_min,
    update_sum,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ours v1: build richer cached features.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument("--chunk-size", default=1_000_000, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_train_test(data_dir: Path):
    train = pd.read_csv(data_dir / "train_format1.csv", usecols=["user_id", "merchant_id", "label"])
    test = pd.read_csv(data_dir / "test_format1.csv", usecols=["user_id", "merchant_id"])
    pairs = pd.concat(
        [train[["user_id", "merchant_id"]], test[["user_id", "merchant_id"]]], ignore_index=True
    ).drop_duplicates()
    return train, test, pairs


def init_sum_agg(index: Union[pd.Index, pd.MultiIndex], prefix: str) -> pd.DataFrame:
    cols = expected_count_columns(prefix)
    return pd.DataFrame(0, index=index, columns=cols, dtype=np.int32)


def init_min_agg(index: Union[pd.Index, pd.MultiIndex], prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_recency_{name}" for name in ACTION_TYPES.values()] + [f"{prefix}_recency_any"]
    return pd.DataFrame(int(RECENCY_DEFAULT), index=index, columns=cols, dtype=np.int16)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for prefix in ["pair", "user", "merchant"]:
        for w in WINDOW_LABELS:
            df[f"{prefix}_action_w{w}"] = (
                df[f"{prefix}_click_w{w}"]
                + df[f"{prefix}_cart_w{w}"]
                + df[f"{prefix}_buy_w{w}"]
                + df[f"{prefix}_fav_w{w}"]
            )

    # Pre-11/11 behavior (everything before day 0)
    for action in ["click", "cart", "buy", "fav", "action"]:
        df[f"pair_pre1111_{action}"] = df[f"pair_{action}_wall"] - df[f"pair_{action}_w1"]
        df[f"user_pre1111_{action}"] = df[f"user_{action}_wall"] - df[f"user_{action}_w1"]
        df[f"merchant_pre1111_{action}"] = df[f"merchant_{action}_wall"] - df[f"merchant_{action}_w1"]

    # Recency gaps (smaller means closer to 11/11)
    df["pair_recency_gap_click_buy"] = df["pair_recency_click"] - df["pair_recency_buy"]
    df["pair_recency_gap_fav_buy"] = df["pair_recency_fav"] - df["pair_recency_buy"]

    # Recentness ratios
    df["pair_recent_ratio_w7_all"] = safe_div(df["pair_action_w7"], df["pair_action_wall"])
    df["pair_recent_ratio_w30_all"] = safe_div(df["pair_action_w30"], df["pair_action_wall"])
    df["user_recent_ratio_w30_all"] = safe_div(df["user_action_w30"], df["user_action_wall"])

    # Conversion-like ratios (within pair)
    for w in ["7", "30", "all"]:
        df[f"pair_buy_to_click_w{w}"] = safe_div(df[f"pair_buy_w{w}"], df[f"pair_click_w{w}"])
        df[f"pair_buy_to_cart_w{w}"] = safe_div(df[f"pair_buy_w{w}"], df[f"pair_cart_w{w}"])
        df[f"pair_buy_to_fav_w{w}"] = safe_div(df[f"pair_buy_w{w}"], df[f"pair_fav_w{w}"])

    # Affinity shares: how important this merchant is for the user (and vice versa)
    df["pair_buy_share_user_all"] = safe_div(df["pair_buy_wall"], df["user_buy_wall"])
    df["pair_click_share_user_all"] = safe_div(df["pair_click_wall"], df["user_click_wall"])
    df["pair_buy_share_user_w30"] = safe_div(df["pair_buy_w30"], df["user_buy_w30"])
    df["pair_click_share_user_w30"] = safe_div(df["pair_click_w30"], df["user_click_w30"])

    df["pair_buy_share_merchant_all"] = safe_div(df["pair_buy_wall"], df["merchant_buy_wall"])
    df["pair_click_share_merchant_all"] = safe_div(df["pair_click_wall"], df["merchant_click_wall"])

    # Log transforms (stabilize heavy tails)
    df["log1p_pair_action_all"] = np.log1p(df["pair_action_wall"])
    df["log1p_user_action_all"] = np.log1p(df["user_action_wall"])
    df["log1p_merchant_action_all"] = np.log1p(df["merchant_action_wall"])

    return df


def build_features(data_dir: Path, out_dir: Path, chunk_size: int, force: bool):
    ours_dir = out_dir / "ours_v1"
    feature_dir = ours_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    train_out = feature_dir / "train_features.parquet"
    test_out = feature_dir / "test_features.parquet"

    if train_out.exists() and test_out.exists() and not force:
        return train_out, test_out

    train, test, pairs = load_train_test(data_dir)

    pair_index = pd.MultiIndex.from_frame(pairs[["user_id", "merchant_id"]])
    pair_index.names = ["user_id", "merchant_id"]
    user_index = pd.Index(pairs["user_id"].unique(), name="user_id")
    merchant_index = pd.Index(pairs["merchant_id"].unique(), name="merchant_id")

    pair_df = pairs[["user_id", "merchant_id"]].copy()
    user_set = set(user_index.to_numpy())
    merchant_set = set(merchant_index.to_numpy())

    pair_sum = init_sum_agg(pair_index, "pair")
    pair_min = init_min_agg(pair_index, "pair")

    user_sum = init_sum_agg(user_index, "user")
    user_min = init_min_agg(user_index, "user")

    merchant_sum = init_sum_agg(merchant_index, "merchant")

    log_path = data_dir / "user_log_format1.csv"
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

        mmdd = chunk["time_stamp"].astype(int).to_numpy()
        chunk = chunk.assign(days_before=mmdd_to_days_before_1111(mmdd))

        chunk_users = chunk[chunk["user_id"].isin(user_set)]

        # Pair-level (only train/test pairs)
        if not chunk_users.empty:
            chunk_pairs = chunk_users.merge(pair_df, on=["user_id", "merchant_id"], how="inner")
            if not chunk_pairs.empty:
                pair_counts = compute_window_counts(chunk_pairs, ["user_id", "merchant_id"], "pair")
                pair_rec = compute_recency(chunk_pairs, ["user_id", "merchant_id"], "pair")
                update_sum(pair_sum, pair_counts)
                update_min(pair_min, pair_rec)

            # User-level (across all merchants)
            user_counts = compute_window_counts(chunk_users, ["user_id"], "user")
            user_rec = compute_recency(chunk_users, ["user_id"], "user")
            update_sum(user_sum, user_counts)
            update_min(user_min, user_rec)

        # Merchant-level (only merchants in our train/test universe)
        chunk_merchants = chunk[chunk["merchant_id"].isin(merchant_set)]
        if not chunk_merchants.empty:
            merchant_counts = compute_window_counts(chunk_merchants, ["merchant_id"], "merchant")
            update_sum(merchant_sum, merchant_counts)

    pair_agg = pair_sum.join(pair_min)
    user_agg = user_sum.join(user_min)
    merchant_agg = merchant_sum

    user_info = pd.read_csv(
        data_dir / "user_info_format1.csv", usecols=["user_id", "age_range", "gender"]
    )
    user_info["age_range"] = user_info["age_range"].fillna("unk").astype(str)
    user_info["gender"] = user_info["gender"].fillna("unk").astype(str)

    pair_agg = pair_agg.reset_index()
    user_agg = user_agg.reset_index()
    merchant_agg = merchant_agg.reset_index()

    train_feat = train.merge(pair_agg, on=["user_id", "merchant_id"], how="left")
    train_feat = train_feat.merge(user_agg, on="user_id", how="left")
    train_feat = train_feat.merge(merchant_agg, on="merchant_id", how="left")
    train_feat = train_feat.merge(user_info, on="user_id", how="left")

    test_feat = test.merge(pair_agg, on=["user_id", "merchant_id"], how="left")
    test_feat = test_feat.merge(user_agg, on="user_id", how="left")
    test_feat = test_feat.merge(merchant_agg, on="merchant_id", how="left")
    test_feat = test_feat.merge(user_info, on="user_id", how="left")

    # Fill missing (should be rare, but keep robust)
    count_cols = (
        expected_count_columns("pair")
        + expected_count_columns("user")
        + expected_count_columns("merchant")
    )
    rec_cols = (
        [f"pair_recency_{n}" for n in ACTION_TYPES.values()]
        + ["pair_recency_any"]
        + [f"user_recency_{n}" for n in ACTION_TYPES.values()]
        + ["user_recency_any"]
    )

    train_feat[count_cols] = train_feat[count_cols].fillna(0).astype(np.int32)
    test_feat[count_cols] = test_feat[count_cols].fillna(0).astype(np.int32)
    train_feat[rec_cols] = train_feat[rec_cols].fillna(int(RECENCY_DEFAULT)).astype(np.int16)
    test_feat[rec_cols] = test_feat[rec_cols].fillna(int(RECENCY_DEFAULT)).astype(np.int16)

    train_feat[CATEGORICAL_FEATURES] = train_feat[CATEGORICAL_FEATURES].fillna("unk").astype(str)
    test_feat[CATEGORICAL_FEATURES] = test_feat[CATEGORICAL_FEATURES].fillna("unk").astype(str)

    train_feat = add_derived_features(train_feat)
    test_feat = add_derived_features(test_feat)

    train_feat.to_parquet(train_out, index=False)
    test_feat.to_parquet(test_out, index=False)

    return train_out, test_out


def main():
    args = parse_args()
    build_features(args.data_dir, args.out_dir, args.chunk_size, args.force)


if __name__ == "__main__":
    main()
