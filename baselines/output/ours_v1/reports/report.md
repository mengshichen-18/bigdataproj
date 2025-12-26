# Ours v1 Report

- Model: `xgboost`
- OOF AUC: `0.651815`
- OOF Logloss: `0.657393`
- CV: StratifiedGroupKFold(user_id), n_splits=5
- Target Encoding: merchant_id with prior=20.0 (train LOO)

## Fold Metrics
| Fold | AUC | Logloss |
| --- | ---: | ---: |
| 1 | 0.660605 | 0.666573 |
| 2 | 0.655127 | 0.671292 |
| 3 | 0.650233 | 0.647280 |
| 4 | 0.654847 | 0.664693 |
| 5 | 0.664445 | 0.637167 |

## Top Features (gain)

| Feature | Gain | Weight |
| --- | ---: | ---: |
| pair_buy_to_cart_wall | 2778.331726 | 119 |
| merchant_click_w1 | 1890.918152 | 1793 |
| te_merchant | 1881.317383 | 9702 |
| merchant_cart_w1 | 1663.437744 | 907 |
| merchant_pre1111_cart | 1486.079269 | 446 |
| merchant_cart_w3 | 1387.241791 | 532 |
| pair_buy_to_cart_w7 | 1342.052185 | 295 |
| merchant_buy_w1 | 1316.070618 | 907 |
| merchant_click_w7 | 1251.384369 | 232 |
| log1p_pair_action_all | 1225.310345 | 27 |
| merchant_click_w60 | 1203.404968 | 283 |
| merchant_buy_w180 | 1192.389954 | 524 |
| merchant_cart_w7 | 1171.513489 | 371 |
| merchant_fav_w1 | 1135.341675 | 871 |
| merchant_click_w180 | 1133.653587 | 387 |
| merchant_fav_w180 | 1126.244110 | 393 |
| merchant_buy_w3 | 1107.323135 | 304 |
| merchant_fav_w90 | 1096.523819 | 370 |
| merchant_click_w14 | 1090.061523 | 236 |
| merchant_click_w3 | 1075.169098 | 315 |
