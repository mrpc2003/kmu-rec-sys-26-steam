# User-gated UserKNN residual probe

No Kaggle submission is performed.

- verdict: **WEAK_SIGNAL**
- strict pass count: `0`

| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `score_userknn_max_w0.25__n_high_q0.8__B2` | +0.000900 | +0.000200 | 3/3 | 406 | 352 | 0.05415 |
| 2 | `score_userknn_max_w0.25__train_deg_high_q0.2__B2` | +0.000867 | -0.000200 | 2/3 | 516 | 464 | 0.1032 |
| 3 | `score_userknn_max_w0.1__n_high_q0.8` | +0.000800 | +0.000500 | 3/3 | 236 | 188 | 0.02235 |
| 4 | `score_userknn_max_w0.1__n_high_q0.8__B2` | +0.000800 | +0.000500 | 3/3 | 236 | 188 | 0.02235 |
| 5 | `score_userknn_max_w0.1__n_high_q0.8__B3` | +0.000800 | +0.000500 | 3/3 | 236 | 188 | 0.02235 |
| 6 | `score_userknn_max_w0.25__n_high_q0.8__B3` | +0.000800 | +0.000200 | 3/3 | 406 | 358 | 0.08899 |
| 7 | `score_userknn_max_w0.3__score_userknn_max_mean_low_q0.6__B2` | +0.000800 | -0.000100 | 2/3 | 436 | 388 | 0.1015 |
| 8 | `score_userknn_max_w0.25__train_deg_high_q0.2__B3` | +0.000800 | -0.000200 | 2/3 | 516 | 468 | 0.134 |
| 9 | `score_userknn_max_w0.25__n_high_q0.8` | +0.000767 | +0.000200 | 3/3 | 405 | 359 | 0.1035 |
| 10 | `score_userknn_max_w0.25__score_userknn_popnorm_base_changed_high_q0.8__B2` | +0.000767 | -0.001100 | 2/3 | 458 | 412 | 0.1271 |
| 11 | `score_userknn_max_w0.25__n_high_q0.6__B2` | +0.000767 | -0.000600 | 2/3 | 503 | 457 | 0.1464 |
| 12 | `score_userknn_max_w0.25__train_deg_high_q0.2` | +0.000767 | -0.000200 | 2/3 | 515 | 469 | 0.1514 |
| 13 | `score_userknn_max_w0.25__score_userknn_top10_base_changed_high_q0.8__B2` | +0.000733 | -0.001000 | 2/3 | 477 | 433 | 0.154 |
| 14 | `score_userknn_max_w0.25__score_userknn_max_std_high_q0.2__B2` | +0.000733 | -0.000500 | 2/3 | 501 | 457 | 0.1647 |
| 15 | `score_userknn_max_w0.2__n_high_q0.8__B2` | +0.000733 | +0.000300 | 3/3 | 324 | 280 | 0.08009 |
| 16 | `score_userknn_max_w0.1__train_deg_high_q0.2` | +0.000733 | +0.000200 | 3/3 | 352 | 308 | 0.0941 |
| 17 | `score_userknn_max_w0.1__train_deg_high_q0.2__B2` | +0.000733 | +0.000200 | 3/3 | 352 | 308 | 0.0941 |
| 18 | `score_userknn_max_w0.1__train_deg_high_q0.2__B3` | +0.000733 | +0.000200 | 3/3 | 352 | 308 | 0.0941 |
| 19 | `score_userknn_max_w0.25__score_userknn_max_mean_low_q0.6__B2` | +0.000733 | +0.000100 | 3/3 | 379 | 335 | 0.1075 |
| 20 | `score_userknn_max_w0.3__train_deg_high_q0.2__B2` | +0.000733 | -0.000900 | 2/3 | 600 | 556 | 0.206 |
| 21 | `score_userknn_max_w0.3__score_userknn_max_mean_low_q0.6` | +0.000700 | -0.000100 | 2/3 | 440 | 398 | 0.1566 |
| 22 | `score_userknn_max_w0.2__train_deg_high_q0.2__B2` | +0.000700 | +0.000000 | 2/3 | 438 | 396 | 0.1556 |
| 23 | `score_userknn_max_w0.25__score_userknn_max_mean_low_q0.8__B2` | +0.000700 | -0.000500 | 2/3 | 505 | 463 | 0.1875 |
| 24 | `score_userknn_max_w0.3__score_userknn_popnorm_base_changed_high_q0.8__B2` | +0.000700 | -0.001700 | 2/3 | 519 | 477 | 0.1939 |
| 25 | `score_userknn_max_w0.3__score_userknn_max_mean_low_q0.6__B3` | +0.000667 | -0.000100 | 2/3 | 438 | 398 | 0.1774 |
| 26 | `score_userknn_max_w0.2__train_deg_high_q0.2` | +0.000667 | +0.000000 | 2/3 | 441 | 401 | 0.1789 |
| 27 | `score_userknn_max_w0.2__train_deg_high_q0.2__B3` | +0.000667 | +0.000000 | 2/3 | 441 | 401 | 0.1789 |
| 28 | `score_userknn_max_w0.25__score_userknn_popnorm_base_changed_high_q0.8` | +0.000667 | -0.001100 | 2/3 | 458 | 418 | 0.1876 |
| 29 | `score_userknn_max_w0.25__score_userknn_popnorm_base_changed_high_q0.8__B3` | +0.000667 | -0.001100 | 2/3 | 458 | 418 | 0.1876 |
| 30 | `score_userknn_max_w0.25__n_high_q0.6__B3` | +0.000667 | -0.000600 | 2/3 | 503 | 463 | 0.2095 |