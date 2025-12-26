#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="dataset/data_format1"
OUT_DIR="baselines/output"
CHUNK_SIZE=1000000

# Model options:
# - hgb: sklearn HistGradientBoosting (no extra deps)
# - xgboost: requires `pip install xgboost`
# - lightgbm: requires `pip install lightgbm`
MODEL="${MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  if python -c "import xgboost" >/dev/null 2>&1; then
    MODEL="xgboost"
  elif python -c "import lightgbm" >/dev/null 2>&1; then
    MODEL="lightgbm"
  else
    MODEL="hgb"
  fi
fi

EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-auc}"
TE_PRIOR="${TE_PRIOR:-20}"

python baselines/ours_v1_build_features.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --chunk-size "$CHUNK_SIZE"

python baselines/ours_v1_train.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --chunk-size "$CHUNK_SIZE" \
  --model "$MODEL" \
  --early-stop-metric "$EARLY_STOP_METRIC" \
  --te-prior "$TE_PRIOR"

echo "Done. See: baselines/output/ours_v1/prediction.csv"
