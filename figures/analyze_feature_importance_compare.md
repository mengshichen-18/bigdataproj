# 特征重要性对比分析

本文基于统一口径的 XGBoost gain，对 Baseline C 与 Ours‑v1 的特征重要性进行对比总结。

## 主要观察

- Baseline C 的高权重特征以近期窗口的 pair 统计为主，例如
  `num_last7_buy_share`、`num_total_buy`、`num_cart_to_buy_rate`。
- Ours‑v1 的 Top features 包含跨层级信号，如 `pair_buy_to_cart_wall`、
  `merchant_click_w1`、`te_merchant`，体现出商家层面先验与结构化转化信号的重要性。
- Baseline C 更偏重 last7 窗口；Ours‑v1 的重要性分布更均衡，覆盖多窗口与多层级特征，
  更符合冷启动场景下的泛化需求。

## 解读

- Ours‑v1 的提升主要来自商家侧先验（目标编码）与多窗口行为结构信息，
  而不仅仅是短期行为强度。
- Baseline C 的 “importance” 与 “gain” 排序差异不大，说明该模型下特征排序较稳定；
  这也是两版图看起来差别不大的原因。

## 注意事项

- gain 只在同一模型内部可比较；图表用于对比“特征类型与排序”，不用于跨模型绝对值比较。
