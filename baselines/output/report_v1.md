# Baseline 实验分析报告（v1）

> 说明：本报告基于 `baselines/output` 目录下的运行产物自动整理与补充分析（训练集随机 8/2 切分验证）。  
> 关键产物：`baselines/output/baseline_metrics.json`、`baselines/output/baseline_a_coefficients.csv`、`baselines/output/baseline_b_l1_coefficients.csv`、`baselines/output/baseline_b_l2_coefficients.csv`、`baselines/output/baseline_b_l1_stability.csv`、`baselines/output/baseline_c_xgboost_importance.csv`。

## 1. 实验设置

### 1.1 数据与窗口
- 数据：`dataset/data_format1`
- 目标：预测新客在未来 6 个月是否会复购同一商家（`label`）。
- 特征窗口：
  - `last7_*`：`time_stamp >= 1105`（11/05–11/11 含当天）
  - `total_*`：全量 6 个月 + 11/11（数据天然截至 11/11）

### 1.2 特征集合（本次实现）
按 `(user_id, merchant_id)` 聚合行为计数与派生比率：
- 计数特征：
  - 全窗口：`total_click / total_cart / total_buy / total_fav`
  - 近 7 天：`last7_click / last7_cart / last7_buy / last7_fav`
  - 汇总：`total_action`、`last7_action`
- 比率/强度特征（带 `+1` 平滑）：
  - `buy_rate_total`、`buy_rate_last7`
  - `recent_activity_ratio`
  - `click_to_buy_rate`、`cart_to_buy_rate`、`fav_to_buy_rate`
  - `last7_buy_share`
- 人口学：
  - `age_range`、`gender`（缺失填 `unk`，one-hot）

### 1.3 训练/验证与模型要点
- 切分：`train_test_split(test_size=0.2, stratify=y, random_state=42)`
- Baseline A：LR（仅 3 个“最基础统计特征”），用于可解释基线。
- Baseline B：LR（L1/L2），用于变量选择与系数对比；额外做 L1 稳定性（3 次不同随机种子切分）。
- Baseline C：XGBoost（树模型），用于捕获非线性与交互；特征采用数值 + one-hot。

## 2. Baseline A/B/C 对照表（方法论层面）

| Baseline | 模型 | 特征复杂度 | 可解释性 | 擅长捕获 | 典型不足 |
| --- | --- | --- | --- | --- | --- |
| A | LR | 低（3 个近 7 天计数） | 强（系数直观） | 单调线性关系 | 不擅长阈值/交互/非线性 |
| B | LR + L1/L2 | 中（计数+比率+画像） | 强（系数+稀疏） | 线性 + 稳定特征筛选 | 仍难捕获非线性阈值；共线性会影响系数符号 |
| C | XGBoost | 中（同一批特征，非线性学习） | 中（重要性/分裂规则） | 非线性阈值、特征交互、稀疏 one-hot | 需要调参/更多特征；概率可能需校准 |

## 3. 指标结果（验证集）

来源：`baselines/output/baseline_metrics.json`

| Baseline | AUC（越大越好） | Logloss（越小越好） |
| --- | ---: | ---: |
| A（LR） | 0.592966 | 0.677814 |
| B-L1（LR） | 0.616020 | 0.667620 |
| B-L2（LR） | 0.616000 | 0.667627 |
| C（XGBoost） | 0.594340 | 0.637833 |

增量（相对 Baseline A）：
- B-L1 AUC +0.02305，Logloss -0.01019
- B-L2 AUC +0.02303，Logloss -0.01019
- C AUC +0.00137，Logloss -0.03998

解读：
- **排序能力（AUC）**：B 明显优于 A/C（当前特征集下，线性模型已能稳定提升可分性）。
- **概率损失（Logloss）**：C 明显更低，但 AUC 并未跟随提升（更像做了局部阈值拟合/概率收缩）。

## 4. Baseline A（可解释基线）

### 4.1 系数
来源：`baselines/output/baseline_a_coefficients.csv`

| Feature | Coef |
| --- | ---: |
| last7_buy | 0.229173 |
| last7_click | 0.127380 |
| last7_fav | 0.047344 |

解读（业务含义）：
- 三个系数均为正，符合直觉：近 7 天买/点/收藏越多，越可能复购。
- 影响强度排序：**购买次数 > 点击次数 > 收藏次数**。

### 4.2 数据侧佐证：`last7_buy` 的单变量区分度
基于缓存特征 `baselines/output/features/train_features.parquet` 统计：
- `last7_buy == 1` 占比约 79.38%，其复购率约 5.16%
- `last7_buy >= 2` 占比约 20.62%，复购率显著更高（如下表）

| last7_buy | 样本数 | 复购率（label=1） |
| ---: | ---: | ---: |
| 1 | 207,061 | 0.0516 |
| 2 | 35,599 | 0.0853 |
| 3 | 9,844 | 0.1109 |
| 4 | 4,404 | 0.1272 |
| 5 | 1,907 | 0.1416 |
| 6+ | 2,049 | 0.1562 |

结论：Baseline A 的核心信息主要来自“是否只买一次/买了多次”的近因强度，这是一个非常稳健且可解释的起点。

## 5. Baseline B（L1/L2 Logistic：变量选择 + 系数解释）

### 5.1 B-L1 vs B-L2：几乎一致
- 指标：AUC/Logloss 基本相同（0.61602 vs 0.61600）。
- 系数：Top 特征与方向高度一致，说明在当前特征集下：
  - “使用 L1 做稀疏”并未带来明显额外收益；
  - 主要提升来自“特征更丰富”，而非正则形式差异。

### 5.2 Top 系数与含义（L1）
来源：`baselines/output/baseline_b_l1_coefficients.csv`

| Feature | Coef | 直观解释（建议口径） |
| --- | ---: | --- |
| cat__gender_unk | -0.5045 | 画像缺失用户更像一次性流量（风险更高） |
| cat__age_range_unk | -0.3601 | 年龄缺失同上（风险更高） |
| cat__age_range_1.0 | -0.3397 | 样本极少（仅 13 条），更像噪声/不稳定项 |
| num__last7_buy_share | +0.3048 | 近 7 天购买强度（近似在编码“买了几次”） |
| num__recent_activity_ratio | -0.2087 | 活跃集中于最后一周可能更“促销驱动” |
| num__buy_rate_total | -0.2074 | “少行为直接买”未必忠诚（需结合共线性谨慎解释） |
| cat__age_range_2.0 | -0.1776 | 特定年龄段复购倾向偏低（以数据为准） |
| cat__gender_2.0 | +0.1687 | 性别未知/2 的群体复购略高（样本量较小） |

### 5.3 Top 系数与含义（L2）
来源：`baselines/output/baseline_b_l2_coefficients.csv`（结论与 L1 非常接近）

| Feature | Coef |
| --- | ---: |
| cat__gender_unk | -0.4935 |
| cat__age_range_1.0 | -0.4368 |
| cat__age_range_unk | -0.3301 |
| num__last7_buy_share | +0.3050 |
| num__recent_activity_ratio | -0.2089 |
| num__buy_rate_total | -0.2074 |
| cat__gender_2.0 | +0.1791 |
| cat__age_range_5.0 | +0.1700 |

### 5.4 L1 稳定性（“哪些特征稳定进入模型”）
来源：`baselines/output/baseline_b_l1_stability.csv`

现象：
- 本次 L1 稳定性（3 次切分）中，**几乎所有特征都 3/3 次被选中**（selection_rate=1.0），只有 `cat__age_range_8.0` 为 2/3。

解读：
- 当前维度（31 个）偏低、且默认正则强度不够“稀疏”，导致 L1 没有产生“明显的变量淘汰效果”。
- 若想让 B 真正体现“变量选择”，建议后续对 `C` 做网格/路径（例如 `C ∈ {0.05, 0.1, 0.2, 0.5, 1, 2}`），对比“性能-稀疏度”曲线，再讨论“稳定入模变量”。

## 6. Baseline C（XGBoost：非线性 + 交互 + 稀疏特征）

### 6.1 指标解读
- AUC：0.59434（略高于 A，但明显低于 B）
- Logloss：0.63783（当前三类中最低）

一个合理解释路径：
- 树模型利用阈值切分能把一部分样本概率拉开、从而降低 Logloss；
- 但在本次“较粗的聚合特征”下，树模型的排序收益（AUC）没有被充分激活（需要更丰富的特征工程或调参/验证策略）。

### 6.2 Top 特征重要性（XGBoost）
来源：`baselines/output/baseline_c_xgboost_importance.csv`

| Feature | Importance |
| --- | ---: |
| num__last7_buy_share | 0.2516 |
| num__cart_to_buy_rate | 0.1073 |
| num__total_buy | 0.0715 |
| num__total_action | 0.0439 |
| cat__gender_unk | 0.0327 |
| num__click_to_buy_rate | 0.0249 |
| cat__age_range_2.0 | 0.0243 |
| num__recent_activity_ratio | 0.0242 |
| num__buy_rate_total | 0.0241 |
| num__total_click | 0.0225 |

解读：
- Top1 `last7_buy_share` 占比 ~25%：再次说明“购买强度”是最强信号。
- `cart_to_buy_rate` 在树里非常靠前，但在 LR 中系数接近 0，这是典型现象：  
  **树模型能自然学习非线性阈值/分段效应，而线性模型只能给一个“平均斜率”。**

### 6.3 用“数据现象”解释树模型优势（本次可直接写进分析）
（基于 `train_features.parquet` 的分桶统计）

1) `cart_to_buy_rate` 明显“阈值型”：
- `cart_to_buy_rate >= 2`：样本 53,122，复购率约 9.83%
- `cart_to_buy_rate >= 5`：样本 3,904，复购率约 14.91%
- `cart_to_buy_rate >= 10`：样本 120，复购率约 17.50%

这类“到某个阈值后复购率跳升”的关系，树模型一两层就能切出来；LR 很难只靠一个线性系数表达。

2) `recent_activity_ratio` 呈现非单调（近似 U 型）：
- 极低（<0.25）或极高（>0.99）复购率更高；
- 中间区间复购率更低。

这也更符合树模型的优势：通过多段切分表达非单调关系。

## 7. 数据诊断与重要现象（避免误读系数/指标）

### 7.1 类别分布（训练集）
- 正例率（label=1）：约 6.115%
- `gender=unk` 占比约 1.42%，复购率约 3.07%（显著偏低）
- `age_range=unk` 占比约 0.48%，复购率约 2.55%（显著偏低）

这解释了为何 `gender_unk / age_range_unk` 在 LR 系数中显著为负、在树模型重要性中也靠前：**缺失本身就是强信号**。

### 7.2 关键“冗余/共线性”现象
在你当前的样本构成下（新客-商家对）：
- `total_buy == last7_buy` 在训练/测试中均为 100% 成立  
  => “总购买次数”与“近 7 天购买次数”完全共线（因为新客对该商家历史上无更早购买）。
- `total_action == last7_action` 约 79.8% 成立  
  => 大部分用户与该商家的互动都集中在最后 7 天，只有约 20% 有更早的浏览/加购/收藏历史。

这会带来两个后果：
- 线性模型里相关特征的系数符号/大小会受到共线性影响（解释时要结合分桶/单变量统计）。
- 树模型在重要性上更偏向选择“能提供最好阈值切分”的那一个共线特征。

### 7.3 关于 Logloss：为什么数值偏高（重要备注）
本次 LR 使用了 `class_weight="balanced"`，树使用 `scale_pos_weight`：
- 这会改变模型的最优目标，使输出 `predict_proba` **不再天然等价于原始分布下的真实概率**；
- 因此 Logloss 更适合“同设定下的相对比较”，不宜直接当作“概率质量”结论。

若你要把 `prob` 作为可提交/可运营的概率，建议后续补充：
- 不加权训练 + 阈值/采样处理（看 AUC），或
- 保留加权但做概率校准（Platt/Isotonic）后再看 Logloss/Brier。

## 8. 小结与下一步建议（v2 迭代方向）

### 8.1 结论小结
- **Baseline A**：可解释且符合直觉，核心是“近 7 天购买强度”。
- **Baseline B**：在当前特征集上 AUC 最好，说明“计数+比率+画像”对排序有稳定增益；L1 与 L2 表现接近。
- **Baseline C**：Logloss 最好且重要性显示对阈值/非线性敏感特征（如 `cart_to_buy_rate`）更友好，但目前 AUC 未超过 B，表明还需要更丰富特征/更合理验证/调参。

### 8.2 建议的 v2 迭代清单（按性价比排序）
1) 让 L1 真正做变量选择：对 `C` 做网格/路径，并输出“稀疏度-性能”曲线与稳定入模特征。
2) 增加多窗口特征：1/3/7/14/30 天（点击/加购/收藏/购买及其比率），以及“距 11/11 最近一次行为间隔”等 recency 特征。
3) 加入多样性特征（对树模型很友好）：近 7/30 天 unique `item_id/cat_id/brand_id` 数、Top 品类占比等（能刻画兴趣集中度）。
4) 加入用户全局特征与商家画像：用户全站活跃度、商家整体复购率/新客率等（注意只用训练窗口内可得信息）。
5) 评估策略升级：用按 `user_id` 分组的验证（避免同一用户多商家泄漏），并在最终提交前做概率校准。

