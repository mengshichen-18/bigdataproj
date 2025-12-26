# Ours v1 Report

- Model: `xgboost`
- OOF AUC: `0.650826`
- OOF Logloss: `0.658629`
- CV: StratifiedGroupKFold(user_id), n_splits=5
- Target Encoding: merchant_id with prior=20.0 (train LOO)

## Fold Metrics
| Fold | AUC | Logloss |
| --- | ---: | ---: |
| 1 | 0.660605 | 0.666573 |
| 2 | 0.655069 | 0.671302 |
| 3 | 0.650337 | 0.647266 |
| 4 | 0.654498 | 0.671212 |
| 5 | 0.664998 | 0.636857 |

## Top Features (gain)

| Feature | Gain | Weight |
| --- | ---: | ---: |
| pair_buy_to_cart_wall | 2714.795715 | 126 |
| te_merchant | 1893.333710 | 9588 |
| merchant_click_w1 | 1833.296661 | 1787 |
| merchant_cart_w1 | 1653.625641 | 873 |
| merchant_cart_w3 | 1394.783203 | 517 |
| merchant_pre1111_cart | 1368.045532 | 447 |
| pair_buy_to_cart_w7 | 1323.297867 | 294 |
| merchant_buy_w1 | 1283.367599 | 897 |
| merchant_buy_w180 | 1237.674301 | 541 |
| merchant_click_w7 | 1186.675591 | 241 |
| merchant_fav_w1 | 1183.562637 | 789 |
| merchant_click_wall | 1170.003967 | 68 |
| merchant_cart_w7 | 1160.060211 | 368 |
| merchant_cart_w14 | 1154.025024 | 232 |
| log1p_pair_action_all | 1153.869286 | 23 |
| merchant_buy_w60 | 1130.613510 | 419 |
| merchant_fav_w90 | 1110.052765 | 334 |
| merchant_click_w90 | 1109.865082 | 234 |
| merchant_click_w60 | 1103.639252 | 270 |
| merchant_click_w3 | 1095.057953 | 316 |
