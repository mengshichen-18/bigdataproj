#!/usr/bin/env python3
import argparse
from pathlib import Path

from baseline_a import predict_baseline_a, train_baseline_a
from baseline_b import predict_baseline_b, train_baseline_b, train_logistic
from baseline_c import predict_baseline_c, train_baseline_c
from build_features import DATA_DIR, DEFAULT_OUT_DIR, build_features


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines for repeat buyer prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-features", help="Build feature cache.")
    build_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    build_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    build_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    build_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run-baselines", help="Train and evaluate baselines.")
    run_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    run_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    run_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    run_parser.add_argument("--force-features", action="store_true")
    run_parser.add_argument("--l1-runs", default=3, type=int)
    run_parser.add_argument("--tree-model", choices=["lightgbm", "xgboost"], default=None)

    predict_parser = subparsers.add_parser("predict", help="Train a baseline and generate test predictions.")
    predict_parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    predict_parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    predict_parser.add_argument("--chunk-size", default=1_000_000, type=int)
    predict_parser.add_argument("--baseline", choices=["A", "B-L1", "B-L2", "C"], required=True)
    predict_parser.add_argument("--tree-model", choices=["lightgbm", "xgboost"], default=None)
    predict_parser.add_argument("--output", type=Path, default=None)
    predict_parser.add_argument("--force-features", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "build-features":
        build_features(args.data_dir, args.out_dir, args.chunk_size, args.force)
        return

    train_feat, test_feat = build_features(
        args.data_dir, args.out_dir, args.chunk_size, args.force_features
    )

    if args.command == "run-baselines":
        train_baseline_a(train_feat, args.out_dir)
        train_baseline_b(train_feat, args.out_dir, mode="both", l1_runs=args.l1_runs)
        if args.tree_model:
            train_baseline_c(train_feat, args.out_dir, args.tree_model)
        return

    baseline = args.baseline
    if baseline == "A":
        _, model = train_baseline_a(train_feat, args.out_dir)
        predict_baseline_a(model, test_feat, args.out_dir, args.output)
        return
    if baseline == "B-L1":
        _, model, _ = train_logistic(train_feat, "l1", args.out_dir)
        predict_baseline_b(model, test_feat, args.out_dir, args.output, "l1")
        return
    if baseline == "B-L2":
        _, model, _ = train_logistic(train_feat, "l2", args.out_dir)
        predict_baseline_b(model, test_feat, args.out_dir, args.output, "l2")
        return
    if baseline == "C":
        if not args.tree_model:
            raise ValueError("--tree-model is required for baseline C")
        _, model_bundle, _ = train_baseline_c(train_feat, args.out_dir, args.tree_model)
        predict_baseline_c(model_bundle, test_feat, args.out_dir, args.output)
        return


if __name__ == "__main__":
    main()
