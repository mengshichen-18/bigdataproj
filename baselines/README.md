# Baselines

## Data
This script uses `dataset/data_format1`:
- `train_format1.csv`
- `test_format1.csv`
- `user_info_format1.csv`
- `user_log_format1.csv`

## Feature window
The "last7" window is defined as 11/05-11/11 (inclusive), based on `time_stamp >= 1105`.

## Usage
Run commands from the project root so relative paths resolve.

### Ours v1 (recommended)
Build richer multi-window features and train with `StratifiedGroupKFold(user_id)` + merchant target encoding:

```bash
bash baselines/run_ours_v1.sh
```

Environment overrides:

```bash
MODEL=xgboost EARLY_STOP_METRIC=auc TE_PRIOR=20 bash baselines/run_ours_v1.sh
```

Or run step-by-step:

```bash
python baselines/ours_v1_build_features.py
python baselines/ours_v1_train.py --model xgboost --early-stop-metric auc --te-prior 20
```

If you don't want extra dependencies, use the sklearn fallback:

```bash
python baselines/ours_v1_train.py --model hgb
```

### Baselines
Build features (cached to `baselines/output/features`):

```bash
python baselines/build_features.py
```

Train/evaluate Baseline A:

```bash
python baselines/baseline_a.py train
```

Train/evaluate Baseline B (L1/L2 + stability):

```bash
python baselines/baseline_b.py train --mode both --l1-runs 3
```

Train/evaluate Baseline C:

```bash
python baselines/baseline_c.py train --tree-model xgboost
```

Generate predictions:

```bash
python baselines/baseline_a.py predict
python baselines/baseline_b.py predict --penalty l2
python baselines/baseline_c.py predict --tree-model xgboost
```

Optional wrapper (runs the same steps in one CLI):

```bash
python baselines/run_baselines.py run-baselines --tree-model xgboost
```

### Fair evaluation (recommended)
Because the test users do not overlap with train users, you can evaluate baselines with `StratifiedGroupKFold(user_id)`:

```bash
python baselines/eval_baselines_groupcv.py
```

## Outputs
- `baselines/output/features/train_features.parquet`
- `baselines/output/features/test_features.parquet`
- `baselines/output/baseline_metrics.json`
- `baselines/output/reports/baseline_report.md`
- `baselines/output/baseline_a_coefficients.csv`
- `baselines/output/baseline_b_l1_coefficients.csv`
- `baselines/output/baseline_b_l2_coefficients.csv`
- `baselines/output/baseline_c_<model>_importance.csv`
- `baselines/output/prediction_a.csv`
- `baselines/output/prediction_b-l1.csv`
- `baselines/output/prediction_b-l2.csv`
- `baselines/output/prediction_c.csv`
- `baselines/output/ours_v1/features/train_features.parquet`
- `baselines/output/ours_v1/features/test_features.parquet`
- `baselines/output/ours_v1/metrics.json`
- `baselines/output/ours_v1/oof_predictions.csv`
- `baselines/output/ours_v1/prediction.csv`
- `baselines/output/ours_v1/feature_importance.csv`
- `baselines/output/ours_v1/reports/report.md`

## Dependencies
Baseline C requires `lightgbm` or `xgboost`:

```bash
pip install lightgbm xgboost
```

Ours v1 (if using `--model xgboost`) requires:

```bash
pip install xgboost
```
