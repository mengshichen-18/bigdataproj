#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="dataset/data_format1"
OUT_DIR="baselines/output"
CHUNK_SIZE=1000000
L1_RUNS=3
TREE_MODEL="xgboost" # set to "lightgbm" if preferred; set empty to skip baseline C

python baselines/build_features.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --chunk-size "$CHUNK_SIZE"

python baselines/baseline_a.py train \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --chunk-size "$CHUNK_SIZE"

python baselines/baseline_b.py train \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --chunk-size "$CHUNK_SIZE" \
  --mode both \
  --l1-runs "$L1_RUNS"

if [[ -n "$TREE_MODEL" ]]; then
  python baselines/baseline_c.py train \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --tree-model "$TREE_MODEL"
fi

# Optional predictions (uncomment if needed)
# python baselines/baseline_a.py predict --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --chunk-size "$CHUNK_SIZE"
# python baselines/baseline_b.py predict --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --chunk-size "$CHUNK_SIZE" --penalty l2
# python baselines/baseline_c.py predict --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --chunk-size "$CHUNK_SIZE" --tree-model "$TREE_MODEL"
