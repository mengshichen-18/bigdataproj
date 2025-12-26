# bigdataproj

本仓库包含 Tmall 复购预测的三组 baseline（A/B/C）以及一个更强的 `ours_v1` 方法（多窗口特征 + Group-CV + GBDT）。

## 数据

将数据放到：`dataset/data_format1/`（脚本默认读取该路径）。

## 运行 Baselines（A/B/C）

```bash
bash baselines/run_all.sh
```

主要输出：
- `baselines/output/report_v1.md`
- `baselines/output/reports/baseline_report.md`

## 运行 Ours（ours_v1）

```bash
bash baselines/run_ours_v1.sh
```

可选参数（环境变量）：
```bash
MODEL=xgboost EARLY_STOP_METRIC=auc TE_PRIOR=20 bash baselines/run_ours_v1.sh
```

主要输出：
- `baselines/output/ours_v1/reports/report.md`
- `baselines/output/ours_v1/metrics.json`
- `baselines/output/ours_v1/prediction.csv`
