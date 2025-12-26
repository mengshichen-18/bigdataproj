# Baseline Report

## Overview
- Baseline A: Logistic Regression with last-7-day click/buy/fav counts for interpretability.
- Baseline B: L1/L2 Logistic Regression with richer count/ratio features and demographics for variable selection.
- Baseline C: Tree model (LightGBM/XGBoost) to capture nonlinearities and feature interactions.

## Metric Comparison
| Baseline | AUC | Logloss | Notes |
| --- | --- | --- | --- |
| A (LogReg) | 0.592966 | 0.677814 | last7 counts only |
| B-L1 (LogReg) | 0.616020 | 0.667620 | L1 variable selection |
| B-L2 (LogReg) | 0.616000 | 0.667627 | L2 regularization |
| C (xgboost) | 0.594340 | 0.637833 | tree model |

## Baseline A Notes
- Features: last7_buy, last7_click, last7_fav.
- Interpretation: coefficients show marginal effect of recent actions on repeat-buy probability.

Top coefficients:

| Feature | Coef |
| --- | --- |
| last7_buy | 0.229173 |
| last7_click | 0.127380 |
| last7_fav | 0.047344 |

## Baseline B Notes
- Features: full-window counts, last7 counts, ratio features, age/gender (one-hot).
- L1 focuses on sparse selection; L2 keeps dense weights for overall shrinkage.

Stable L1 features (selection frequency):

| Feature | Selected Runs | Selection Rate |
| --- | --- | --- |
| num__total_click | 3 | 1.00 |
| num__total_cart | 3 | 1.00 |
| num__total_buy | 3 | 1.00 |
| num__total_fav | 3 | 1.00 |
| num__last7_click | 3 | 1.00 |
| num__last7_cart | 3 | 1.00 |
| num__last7_buy | 3 | 1.00 |
| num__last7_fav | 3 | 1.00 |
| num__total_action | 3 | 1.00 |
| num__last7_action | 3 | 1.00 |

Top L1 coefficients:

| Feature | Coef |
| --- | --- |
| cat__gender_unk | -0.504534 |
| cat__age_range_unk | -0.360099 |
| cat__age_range_1.0 | -0.339662 |
| num__last7_buy_share | 0.304838 |
| num__recent_activity_ratio | -0.208701 |
| num__buy_rate_total | -0.207414 |
| cat__age_range_2.0 | -0.177647 |
| cat__gender_2.0 | 0.168738 |

Top L2 coefficients:

| Feature | Coef |
| --- | --- |
| cat__gender_unk | -0.493506 |
| cat__age_range_1.0 | -0.436766 |
| cat__age_range_unk | -0.330078 |
| num__last7_buy_share | 0.305037 |
| num__recent_activity_ratio | -0.208887 |
| num__buy_rate_total | -0.207390 |
| cat__gender_2.0 | 0.179054 |
| cat__age_range_5.0 | 0.170012 |

## Baseline C Notes
- Tree models capture nonlinear thresholds and interactions without manual feature crosses.
- They handle sparse one-hot features well via split selection.
- Each tree layer builds conditional logic, suitable for heterogeneous user behaviors.

Top feature importances:

| Feature | Importance |
| --- | --- |
| num__last7_buy_share | 0.251576 |
| num__cart_to_buy_rate | 0.107308 |
| num__total_buy | 0.071512 |
| num__total_action | 0.043921 |
| cat__gender_unk | 0.032713 |
| num__click_to_buy_rate | 0.024928 |
| cat__age_range_2.0 | 0.024292 |
| num__recent_activity_ratio | 0.024208 |
| num__buy_rate_total | 0.024079 |
| num__total_click | 0.022479 |
