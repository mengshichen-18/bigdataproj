# 模型对比图解读

基于 Group‑CV 的 OOF 指标（AUC 与 Logloss）可以得到以下结论：

## AUC 维度（区分能力）

- Ours‑v1 AUC 最高（0.6518），说明整体区分正负样本的能力最强。
- Baseline B（L1/L2）居中（约 0.6188），明显优于 Baseline A 与 Baseline C。
- Baseline A 最低（0.5912），说明仅靠简单 last7 计数的判别力有限。

## Logloss 维度（概率质量）

- Baseline C 的 Logloss 最低（0.6384），在概率校准上更占优。
- Ours‑v1 的 Logloss 为 0.6574，优于 Baseline A/B，但不如 Baseline C。
- Baseline A/B 的 Logloss 较高（约 0.668–0.679），概率输出较粗糙。

## 综合解读

- AUC 表明 Ours‑v1 的排序能力最好，但 Logloss 显示概率仍可优化（可做校准）。
- Baseline C 在 Logloss 上表现好，可能与模型更“保守”的概率输出有关，但 AUC 不强。
- 如果业务更重视“排序/筛选”，Ours‑v1 更合适；如果重视“概率可信度”，Baseline C 值得参考或结合校准。
