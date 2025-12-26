## 关于目录里的各类文件是什么

- Baseline 代码：`baselines/baseline_a.py`、`baselines/baseline_b.py`、`baselines/baseline_c.py`  
  - 指标输出：`baselines/output/baseline_metrics.json`  
  - 报告表：`baselines/output/reports/baseline_report.md`
- Ours‑v1 代码：`baselines/ours_v1_build_features.py`、`baselines/ours_v1_train.py`  
  - 指标输出：`baselines/output/ours_v1/metrics.json`  
  - 报告表：`baselines/output/ours_v1/reports/report.md`
- 统一评估口径：Group‑CV（按 `user_id` 分组）生成 OOF 指标
- 配图脚本：`plot_figs.py` → 输出到 `figures/`
  - `model_compare_auc.png`、`model_compare_logloss.png`  
  - `feature_importance_compare.png`、`ours_roc.png`



# PPT 框架

## 1. 问题背景与数据格式

- 任务目标：预测用户是否对商家复购（比赛场景）
- 数据组成：train/test、用户画像、行为日志
- 关键特性：train/test `user_id` 不重叠（冷启动场景）
- 评估口径：训练集内验证（Group‑CV，按 `user_id` 分组）

> 注：本部分由另一位同学补充细节与图片，我们这里只给结构。

---

## 2. 方法设计

### 2.1 Baselines

- Baseline A（LR）：对每个 (user, merchant) 统计近 7 天行为计数  
  - 特征：`last7_buy/last7_click/last7_fav`  
  - 标准化 + Logistic Regression（`liblinear`，class_weight=balanced）
- Baseline B（LR）：在 A 基础上加入更丰富的统计量传入逻辑回归（也传入画像）  
  - 计数：`total_*` + `last7_*`（click/cart/buy/fav）  
  - 比率：`buy_rate_total/last7`、`click/cart/fav→buy`、`recent_activity_ratio`、`last7_buy_share`  
  - 画像：`age_range/gender` one‑hot  
  - 模型：Logistic Regression（L1/L2两种正则，`saga`，class_weight=balanced）
- Baseline C（XGBoost）：使用 XGBoost（梯度提升决策树），通过逐棵树拟合残差来提升性能，能自动捕捉非线性和特征交互，比线性 LR 更适合复杂行为模式。   
  - 输入：与 Baseline B 相同的数值 + 画像 one‑hot  
  - 模型：XGBoost（`max_depth=6`、`n_estimators=400`、`tree_method=hist`）

可以考虑配图：
- 简单特征表/流程图  
- “基线方法对照表”（方法、特征、优缺点）

### 2.2 Ours‑v1（Baseline C基础上的改进方法）

核心改进点：
- 多窗口统计：在 1/3/7/14/30/60/90/180/all 多个窗口累积行为计数  
- 多粒度特征：  
  - pair 级（用户‑商家交互强度）  
  - user 级（用户整体活跃度与购买倾向）  
  - merchant 级（商家热度与转化结构）  
- 行为时效性：recency（距 11/11 最近一次行为天数）  
- 结构化比率：buy/cart/click 的转化率、占比、`pre1111` 差分等  
- 目标编码：`merchant_id` 的 OOF 目标编码（LOO + prior 平滑）

可配图：
- 特征工程示意图（多窗口 + 多层级）  
- “信息增量”示意（baseline vs ours）

---

## 3. 结果与分析

### 3.1 结果对比

- Baseline 与 Ours 均使用 Group‑CV（按 `user_id` 分组）的 OOF AUC / Logloss  
  - 配图：`model_compare_auc.png`、`model_compare_logloss.png`

表 1：模型 OOF 指标对比（Group‑CV）

| 方法 | OOF AUC | OOF Logloss | 备注 |
| --- | ---: | ---: | --- |
| Baseline A | 0.5912 | 0.6790 | 简单 last7 计数 |
| Baseline B‑L1 | 0.6188 | 0.6683 | 计数+比率+画像 |
| Baseline B‑L2 | 0.6188 | 0.6683 | 计数+比率+画像 |
| Baseline C（XGBoost） | 0.5959 | 0.6384 | 非线性树模型 |
| Ours‑v1（XGBoost） | 0.6518 | 0.6574 | 多窗口+TE |

### 3.2 特征重要性对比

- Baseline C vs Ours‑v1 的 Top features  
  - 配图：`feature_importance_compare.png`  
  - 解读：Baseline 偏 last7，Ours 更依赖商家先验与结构结构化转化

表 2：Top 特征对比（XGBoost gain 口径）

| Baseline C Top‑5 | Ours‑v1 Top‑5 |
| --- | --- |
| num__last7_buy_share | pair_buy_to_cart_wall |
| num__total_buy | merchant_click_w1 |
| num__cart_to_buy_rate | te_merchant |
| num__last7_buy | merchant_cart_w1 |
| num__total_action | merchant_pre1111_cart |

### 3.3 误差与稳定性

- OOF ROC 曲线：`ours_roc.png`  
- Fold AUC 稳定性（可选一行表）

### 3.4 管理学解读

- 商家层面特征的重要性 → 商家质量/热度是复购关键  
- 转化结构特征的重要性 → 关注“加购→购买”路径  
- 近窗口 vs 多窗口 → 近期行为有效，但需要长期结构补充

---

## 4. 结论与建议（可选收尾页）

- 方法层结论：Ours‑v1 排序能力最强，验证口径更合理  
- 业务层建议：优先投放高复购商家、针对高转化路径用户  
- 后续方向：特征扩展、校准 Logloss、轻量融合
