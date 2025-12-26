# 实验汇总报告：Baselines（report_v1）vs Ours-v1（XGBoost）

本报告基于以下两份结果做对照汇总，并补充数据输入说明与原因分析：
- Baselines 分析报告：`baselines/output/report_v1.md`
- Ours-v1 训练报告：`baselines/output/ours_v1/reports/report.md`

另外，为了公平对比（同口径验证），也引用了你已跑出的 Group-CV 基线结果：
- `baselines/output/baseline_groupcv.json`

---

## 1. 数据输入（两边统一来源）

两套方案都使用 `dataset/data_format1/`：
- `dataset/data_format1/train_format1.csv`：训练标签（`user_id, merchant_id, label`）
- `dataset/data_format1/test_format1.csv`：测试集合（`user_id, merchant_id`）
- `dataset/data_format1/user_info_format1.csv`：用户画像（`user_id, age_range, gender`）
- `dataset/data_format1/user_log_format1.csv`：行为日志（`user_id, item_id, cat_id, seller_id, brand_id, time_stamp, action_type`）

关键字段：
- `action_type`：0=click，1=cart，2=buy，3=fav
- `time_stamp`：mmdd（如 0829、1111）
- 本题 train/test 的 `user_id` **不重叠**（更像“冷启动新用户”预测）。

---

## 2. 特征输入（各自产物与口径）

### 2.1 Baselines（来自 `report_v1.md`）
特征缓存（由 `baselines/build_features.py` 生成）：
- `baselines/output/features/train_features.parquet`
- `baselines/output/features/test_features.parquet`

特征口径（pair 级为主）：
- pair 计数：`total_*`（全量窗口）+ `last7_*`（11/05–11/11，`time_stamp>=1105`）
- 派生比率：`buy_rate_total/buy_rate_last7/recent_activity_ratio/click_to_buy_rate/...`
- 人口学：`age_range/gender`（one-hot）

一个关键数据现象（来自 `report_v1.md` 的诊断结论，解释“total 是否真的增强”）：
- 在本题“新客-商家对”的构造下，`total_buy == last7_buy` 在 train/test 中 **100% 成立**；即“总购买次数”并不比“近 7 天购买次数”多提供信息。
- 因此对 baselines 而言，`total_*` 更像是“口径扩展/特征增强的尝试”，但在部分维度上会变成冗余（共线性），收益受限。

### 2.2 Ours-v1（本次更新：XGBoost）
特征缓存（由 `baselines/ours_v1_build_features.py` 生成）：
- `baselines/output/ours_v1/features/train_features.parquet`
- `baselines/output/ours_v1/features/test_features.parquet`

特征口径（多粒度 + 多窗口）：
- pair 级：w1/w3/w7/w14/w30/w60/w90/w180/wall 的累计计数 + recency（距 11/11 最近一次行为的间隔天数）
- user 级：同样的多窗口计数 + recency（跨所有商家，刻画用户整体活跃/购买倾向）
- merchant 级：多窗口计数（跨所有用户，刻画商家热度/转化结构）
- 派生：`pre1111_* = wall - w1`、转化比率、份额特征（该商家占用户总行为比例等）、log1p 稳定化

模型训练（由 `baselines/ours_v1_train.py` 生成）：
- 目标编码：`merchant_id` 做 **OOF 目标编码**（train 内 LOO + prior 平滑=20.0），作为 `te_merchant` 特征
- 主模型：`xgboost`（GBDT）

---

## 3. 评估口径（必须分开看：随机切分 vs Group-CV）

### 3.1 Baselines 的口径：随机 8/2 切分（来自 `report_v1.md`）
`train_test_split(test_size=0.2, stratify=y, random_state=42)`  
这会允许同一 `user_id` 的不同样本同时出现在 train/val（偏“乐观”，与真实测试分布不完全一致）。

### 3.2 Ours-v1 的口径：StratifiedGroupKFold(user_id) OOF（来自 `ours_v1 report`）
`StratifiedGroupKFold(user_id), n_splits=5` 的 OOF 指标  
同一 `user_id` 不会同时出现在 train/val，更贴近本题“新用户”场景。

结论：
- **随机切分指标 ≠ Group-CV 指标**，不要直接混在一起做强弱结论；
- 若要公平比较，需要把 Baselines 也用 Group-CV 评估（你已生成 `baseline_groupcv.json`）。

---

## 4. 结果汇总

### 4.1 Baselines（随机 8/2 切分；来自 `baselines/output/report_v1.md`）
| 方法 | AUC | Logloss | 备注 |
| --- | ---: | ---: | --- |
| Baseline A（LR） | 0.592966 | 0.677814 | last7_buy/click/fav |
| Baseline B-L2（LR） | 0.616000 | 0.667627 | 计数+比率+画像 |
| Baseline C（XGBoost） | 0.594340 | 0.637833 | 树模型（非线性） |

### 4.2 同口径（Group-CV）基线结果（来自 `baselines/output/baseline_groupcv.json`）
| 方法 | 评估方式 | OOF AUC | OOF Logloss |
| --- | --- | ---: | ---: |
| Baseline A（LR） | 5-fold Group OOF | 0.591186 | 0.679031 |
| Baseline B-L2（LR） | 5-fold Group OOF | 0.618804 | 0.668344 |

### 4.3 Ours-v1（XGBoost + merchant TE；来自 `baselines/output/ours_v1/reports/report.md`）
| 方法 | 评估方式 | OOF AUC | OOF Logloss |
| --- | --- | ---: | ---: |
| Ours-v1（XGBoost） | 5-fold Group OOF | 0.650826 | 0.658629 |

相对提升（同口径对比，ours vs Baseline B-L2）：
- AUC：`0.650826 - 0.618804 = +0.0320`
- Logloss：`0.658629 - 0.668344 = -0.0097`（更低更好；仍建议后续做概率校准再看）

Ours-v1 分折指标（稳定性）：
- Fold AUC：0.650~0.665 区间，波动不大

---

## 5. Ours-v1 的“增益从哪来”（结合 Top Features 做原因分析）

Ours-v1（XGBoost）Top features（gain）显示，提升主要来自两类信号：

### 5.1 商家侧信号（merchant-level）很强
Top features 里大量是 `merchant_*` 的多窗口计数（w1/w3/w7/w60/w90/w180/wall）：
- 这类特征刻画“商家热度/短期爆发/长期体量/促销前后差异”，对“新客是否会成为该商家复购”非常关键。
- Baselines 的特征主要是 pair 级计数/比率，缺少 merchant 侧全局统计，因此对“商家质量/规模效应”的表达较弱。

### 5.2 目标编码（`te_merchant`）是强特征，但必须 OOF 才安全
Top features 第二名就是 `te_merchant`，且 split 次数很高：
- 直观含义：不同商家的“复购倾向”差异巨大，目标编码把这种差异直接注入模型。
- 我们在训练中采用 **按折 OOF 计算 + 训练集 LOO**，用 prior=20 平滑，避免 label 泄漏。
- 这类特征常常会显著提升 AUC（尤其在冷启动场景）。

### 5.3 pair 转化结构（如 buy_to_cart）体现了“阈值/非线性”
Top1 为 `pair_buy_to_cart_wall`，且 `pair_buy_to_cart_w7` 也进入前列：
- 说明“加购→购买”的结构性差异对复购强相关，并且关系往往是分段/阈值型；
- XGBoost 能更好地捕获这种非线性与交互，线性 LR 的表达能力有限。

### 5.4 为什么 Baseline C（XGBoost）不如 ours
Baseline C 的输入仍然是“较粗的 pair 级统计 + 画像”，缺少：
- user 全局行为（跨商家）
- merchant 全局行为（跨用户）
- OOF 目标编码（merchant 复购倾向先验）

因此模型类型相同（树），但 **信息量** 不同，ours 的优势来自“更强的特征体系 + 更合理的验证设计”。

---

## 6. 关于 Logloss：如何更可比、更有业务意义

当前 Logloss 仍然偏高，主要原因通常是：
- 类别极不平衡 + 使用了 `scale_pos_weight / class_weight` 做加权训练；
- 加权后 `predict_proba` 不一定是原始分布下的概率，需要校准才能用于概率质量比较/运营。

建议（v2）：
- 在 OOF 预测上做概率校准（Platt/Isotonic），再对比 Logloss/Brier；
- 或固定同一套加权策略，仅看相对变化。

---

## 7. 主要输出文件（便于复现与写论文/汇报）

Baselines：
- 指标：`baselines/output/baseline_metrics.json`（随机切分）
- 系数/重要性：`baselines/output/baseline_a_coefficients.csv`、`baselines/output/baseline_b_l2_coefficients.csv`、`baselines/output/baseline_c_xgboost_importance.csv`
- 同口径 Group-CV：`baselines/output/baseline_groupcv.json`

Ours-v1（XGBoost）：
- 指标：`baselines/output/ours_v1/metrics.json`（Group OOF）
- 分折与 Top features：`baselines/output/ours_v1/reports/report.md`
- OOF 预测：`baselines/output/ours_v1/oof_predictions.csv`
- 测试集预测：`baselines/output/ours_v1/prediction.csv`
- 特征重要性：`baselines/output/ours_v1/feature_importance.csv`

## 8. 下一步建议（v2）
- 加入多样性特征：近 7/30 天 unique `item_id/cat_id/brand_id` 数、活跃天数（distinct `time_stamp`）
- 更精细的时间趋势：`w7 - (w14-w7)`、`w30 - (w60-w30)`（近期相对历史的增量）
- 扩展目标编码：cat_id/brand_id（需严格 OOF）
- 轻量融合：`0.8*ours + 0.2*baseline_B`（用 OOF 搜权重）
