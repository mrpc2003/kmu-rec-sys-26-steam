# KMU RecSys 26 Steam — Stage3 blend validation

Stage2 scores + new time-decay BM25 / CW-lite / review pseudo-cat axes merged and re-evaluated. No Kaggle submission.

## Best by split

| split | best score | row acc | per-user mean acc |
|---|---|---:|---:|
| `val_random_sqrtpop_seed42` | `score_stage3_all_mean_z` | 0.662232 | 0.677972 |
| `val_recent_sqrtpop_seed42` | `score_stage3_all_mean_z` | 0.625525 | 0.629688 |
| `val_random_popbin_seed42` | `score_stage3_all_mean_z` | 0.583417 | 0.597567 |

## Stage2 anchor comparison

| split | Stage2 best (score_blend_mean_z) | Stage3 best |
|---|---:|---:|
| `val_random_sqrtpop_seed42` | 0.659732 | 0.662232 (+0.002500) |
| `val_recent_sqrtpop_seed42` | 0.626025 | 0.625525 (-0.000500) |
| `val_random_popbin_seed42` | 0.590818 | 0.583417 (-0.007401) |

## Full tables

### val_random_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_stage3_all_mean_z` | 0.662232 | 0.677972 |
| 2 | `score_stage3_weighted_z` | 0.661132 | 0.676783 |
| 3 | `score_stage3_top4_cw_mean_z` | 0.660132 | 0.677179 |
| 4 | `score_stage2_mean_z` | 0.659732 | 0.675421 |
| 5 | `score_stage3_top4_mean_z` | 0.658932 | 0.674297 |
| 6 | `score_als_f32_it30_alpha20_popa2` | 0.650930 | 0.670208 |
| 7 | `score_time_itemknn_bm25_hl90_top3` | 0.650130 | 0.668042 |
| 8 | `score_itemknn_bm25_top3` | 0.650130 | 0.667937 |
| 9 | `score_ease_lambda1000` | 0.646329 | 0.660149 |
| 10 | `score_graph_svd_k64` | 0.582416 | 0.581615 |
| 11 | `score_cw_weighted_implicit_logit` | 0.567313 | 0.574202 |
| 12 | `score_review_pseudocat_affinity` | 0.564013 | 0.577593 |

### val_recent_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_stage3_all_mean_z` | 0.625525 | 0.629688 |
| 2 | `score_stage3_weighted_z` | 0.624625 | 0.631363 |
| 3 | `score_stage2_mean_z` | 0.624625 | 0.629875 |
| 4 | `score_stage3_top4_mean_z` | 0.623925 | 0.629944 |
| 5 | `score_stage3_top4_cw_mean_z` | 0.623925 | 0.629664 |
| 6 | `score_itemknn_bm25_top3` | 0.622825 | 0.632081 |
| 7 | `score_time_itemknn_bm25_hl90_top3` | 0.622825 | 0.631988 |
| 8 | `score_als_f32_it30_alpha20_popa2` | 0.612923 | 0.620256 |
| 9 | `score_ease_lambda1000` | 0.612723 | 0.615268 |
| 10 | `score_review_pseudocat_affinity` | 0.569214 | 0.571063 |
| 11 | `score_cw_weighted_implicit_logit` | 0.544809 | 0.543252 |
| 12 | `score_graph_svd_k64` | 0.544309 | 0.539342 |

### val_random_popbin_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_stage3_all_mean_z` | 0.583417 | 0.597567 |
| 2 | `score_stage3_top4_cw_mean_z` | 0.579316 | 0.595532 |
| 3 | `score_stage3_weighted_z` | 0.578616 | 0.592730 |
| 4 | `score_stage2_mean_z` | 0.576215 | 0.592451 |
| 5 | `score_stage3_top4_mean_z` | 0.575715 | 0.589502 |
| 6 | `score_als_f32_it30_alpha20_popa2` | 0.575215 | 0.588609 |
| 7 | `score_itemknn_bm25_top3` | 0.569914 | 0.584741 |
| 8 | `score_time_itemknn_bm25_hl90_top3` | 0.569514 | 0.584072 |
| 9 | `score_ease_lambda1000` | 0.562212 | 0.576154 |
| 10 | `score_cw_weighted_implicit_logit` | 0.554511 | 0.554792 |
| 11 | `score_graph_svd_k64` | 0.548310 | 0.547524 |
| 12 | `score_review_pseudocat_affinity` | 0.542609 | 0.552284 |

