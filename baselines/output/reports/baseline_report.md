# Baseline Report

## Overview
- Baseline A: Logistic Regression with last-7-day click/buy/fav counts for interpretability.
- Baseline B: L1/L2 Logistic Regression with richer count/ratio features and demographics for variable selection.
- Baseline C: Tree model (LightGBM/XGBoost) to capture nonlinearities and feature interactions.

## Metric Comparison
| Baseline | AUC | Logloss | Notes |
| --- | --- | --- | --- |
| A (LogReg) | 0.591186 | 0.679031 | last7 counts only |
| B-L1 (LogReg) | 0.618807 | 0.668344 | L1 variable selection |
| B-L2 (LogReg) | 0.618805 | 0.668347 | L2 regularization |
| C (xgboost) | 0.595916 | 0.638448 | tree model |

## Baseline A Notes
- Features: last7_buy, last7_click, last7_fav.
- Interpretation: coefficients show marginal effect of recent actions on repeat-buy probability.

Top coefficients:

| Feature | Coef |
| --- | --- |
| last7_buy | 0.227559 |
| last7_click | 0.152607 |
| last7_fav | 0.043085 |

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
| cat__gender_unk | -0.437364 |
| num__last7_buy_share | 0.323048 |
| cat__age_range_unk | -0.269201 |
| num__buy_rate_total | -0.203347 |
| num__recent_activity_ratio | -0.197778 |
| cat__age_range_2.0 | -0.185205 |
| num__click_to_buy_rate | 0.149606 |
| cat__gender_2.0 | 0.141399 |

Top L2 coefficients:

| Feature | Coef |
| --- | --- |
| cat__gender_unk | -0.434345 |
| num__last7_buy_share | 0.323184 |
| cat__age_range_unk | -0.272330 |
| num__buy_rate_total | -0.203303 |
| num__recent_activity_ratio | -0.197940 |
| cat__age_range_2.0 | -0.184828 |
| num__click_to_buy_rate | 0.149747 |
| cat__gender_2.0 | 0.145153 |

## Baseline C Notes
- Tree models capture nonlinear thresholds and interactions without manual feature crosses.
- They handle sparse one-hot features well via split selection.
- Each tree layer builds conditional logic, suitable for heterogeneous user behaviors.

Top feature importances:

| Feature | Importance |
| --- | --- |
| num__last7_buy_share | 353.324890 |
| num__total_buy | 114.629852 |
| num__cart_to_buy_rate | 113.640274 |
| num__last7_buy | 57.125294 |
| num__total_action | 50.408516 |
| cat__gender_unk | 32.801231 |
| num__recent_activity_ratio | 27.854376 |
| cat__age_range_2.0 | 26.110693 |
| num__total_click | 26.070709 |
| num__buy_rate_total | 25.416172 |
