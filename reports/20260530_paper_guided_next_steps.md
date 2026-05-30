# KMU RecSys 26 Steam — paper-guided next-step run

Validation-only run covering ICPNS-style community negatives, CW/PU-inspired weighted implicit scorer, time-decay graph scores, and train-only review pseudo-categories. No Kaggle submission was performed.

## Community-aware validation splits built

| split | rows | users | positives | negatives | communities |
|---|---:|---:|---:|---:|---:|
| `val_random_communitypop_seed42` | 19996 | 4736 | 9998 | 9998 | 24 |
| `val_recent_communitypop_seed42` | 19996 | 4736 | 9998 | 9998 | 24 |

## Best score by split

| split | best score | row acc | per-user mean acc | duration sec |
|---|---|---:|---:|---:|
| `val_random_sqrtpop_seed42` | `score_next_blend_mean_z` | 0.645929 | 0.660864 | 323.199 |
| `val_recent_sqrtpop_seed42` | `score_time_itemknn_hl90_top3` | 0.619524 | 0.629951 | 323.285 |
| `val_random_popbin_seed42` | `score_next_blend_mean_z` | 0.563413 | 0.576768 | 307.066 |
| `val_random_communitypop_seed42` | `score_review_pseudocat_affinity` | 0.535307 | 0.541270 | 309.001 |
| `val_recent_communitypop_seed42` | `score_review_pseudocat_blend` | 0.556111 | 0.565365 | 304.848 |

## Full score tables

### val_random_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_next_blend_mean_z` | 0.645929 | 0.660864 |
| 2 | `score_time_itemknn_hl365_top3` | 0.643129 | 0.659704 |
| 3 | `score_time_itemknn_hl730_top3` | 0.643029 | 0.659598 |
| 4 | `score_time_itemknn_hl90_top3` | 0.643029 | 0.659387 |
| 5 | `score_time_itemknn_hl365_sum` | 0.642629 | 0.659996 |
| 6 | `score_time_itemknn_hl730_sum` | 0.642629 | 0.659996 |
| 7 | `score_time_itemknn_hl90_sum` | 0.642328 | 0.659573 |
| 8 | `score_icpns_comm_global_blend` | 0.619124 | 0.633986 |
| 9 | `score_icpns_comm_log_pop` | 0.618824 | 0.633590 |
| 10 | `score_icpns_comm_rate` | 0.618824 | 0.633590 |
| 11 | `score_item_recency_log_pop365` | 0.614123 | 0.629458 |
| 12 | `score_item_log_pop` | 0.613823 | 0.629462 |
| 13 | `score_time_affinity_last` | 0.613623 | 0.629198 |
| 14 | `score_next_blend_priority_z` | 0.592218 | 0.598210 |
| 15 | `score_review_pseudocat_affinity` | 0.590618 | 0.610563 |
| 16 | `score_graph_svd_k64` | 0.582416 | 0.581615 |
| 17 | `score_review_pseudocat_blend` | 0.556111 | 0.570515 |
| 18 | `score_time_affinity_mean` | 0.545409 | 0.552125 |
| 19 | `score_text_len_affinity` | 0.533407 | 0.537982 |
| 20 | `score_review_pseudocat_log_pop` | 0.529806 | 0.537731 |
| 21 | `score_hours_affinity` | 0.524105 | 0.531221 |
| 22 | `score_cw_weighted_implicit_logit` | 0.434487 | 0.425628 |
| 23 | `score_time_ease_hl365_lambda1000` | 0.352671 | 0.339050 |

### val_recent_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_time_itemknn_hl90_top3` | 0.619524 | 0.629951 |
| 2 | `score_time_itemknn_hl365_top3` | 0.619424 | 0.629942 |
| 3 | `score_time_itemknn_hl730_top3` | 0.619424 | 0.629942 |
| 4 | `score_time_itemknn_hl90_sum` | 0.613923 | 0.618501 |
| 5 | `score_time_itemknn_hl365_sum` | 0.613823 | 0.618254 |
| 6 | `score_time_itemknn_hl730_sum` | 0.613823 | 0.618254 |
| 7 | `score_next_blend_mean_z` | 0.608522 | 0.612333 |
| 8 | `score_review_pseudocat_affinity` | 0.592018 | 0.596684 |
| 9 | `score_item_recency_log_pop365` | 0.587518 | 0.587926 |
| 10 | `score_item_log_pop` | 0.586717 | 0.586757 |
| 11 | `score_time_affinity_last` | 0.586317 | 0.586405 |
| 12 | `score_icpns_comm_global_blend` | 0.585117 | 0.583600 |
| 13 | `score_icpns_comm_log_pop` | 0.581416 | 0.579008 |
| 14 | `score_icpns_comm_rate` | 0.581416 | 0.579008 |
| 15 | `score_review_pseudocat_blend` | 0.572214 | 0.575459 |
| 16 | `score_next_blend_priority_z` | 0.554011 | 0.554359 |
| 17 | `score_review_pseudocat_log_pop` | 0.546009 | 0.546933 |
| 18 | `score_graph_svd_k64` | 0.544309 | 0.539342 |
| 19 | `score_text_len_affinity` | 0.528506 | 0.530230 |
| 20 | `score_hours_affinity` | 0.505201 | 0.506690 |
| 21 | `score_time_affinity_mean` | 0.418784 | 0.417273 |
| 22 | `score_cw_weighted_implicit_logit` | 0.398280 | 0.393867 |
| 23 | `score_time_ease_hl365_lambda1000` | 0.385577 | 0.383681 |

### val_random_popbin_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_next_blend_mean_z` | 0.563413 | 0.576768 |
| 2 | `score_time_itemknn_hl90_sum` | 0.558312 | 0.575950 |
| 3 | `score_time_itemknn_hl365_sum` | 0.558212 | 0.575716 |
| 4 | `score_time_itemknn_hl730_sum` | 0.558112 | 0.575505 |
| 5 | `score_next_blend_priority_z` | 0.556011 | 0.558681 |
| 6 | `score_time_itemknn_hl365_top3` | 0.555911 | 0.566762 |
| 7 | `score_time_itemknn_hl90_top3` | 0.555911 | 0.566748 |
| 8 | `score_time_itemknn_hl730_top3` | 0.555811 | 0.566656 |
| 9 | `score_review_pseudocat_affinity` | 0.548310 | 0.560224 |
| 10 | `score_graph_svd_k64` | 0.548310 | 0.547524 |
| 11 | `score_time_affinity_mean` | 0.536107 | 0.540323 |
| 12 | `score_icpns_comm_log_pop` | 0.536007 | 0.546088 |
| 13 | `score_icpns_comm_rate` | 0.536007 | 0.546088 |
| 14 | `score_icpns_comm_global_blend` | 0.535607 | 0.545817 |
| 15 | `score_review_pseudocat_blend` | 0.526605 | 0.536195 |
| 16 | `score_time_affinity_last` | 0.521204 | 0.527157 |
| 17 | `score_item_log_pop` | 0.520804 | 0.526453 |
| 18 | `score_item_recency_log_pop365` | 0.520404 | 0.526669 |
| 19 | `score_text_len_affinity` | 0.518704 | 0.522390 |
| 20 | `score_cw_weighted_implicit_logit` | 0.517904 | 0.521566 |
| 21 | `score_hours_affinity` | 0.506501 | 0.509339 |
| 22 | `score_review_pseudocat_log_pop` | 0.496299 | 0.497815 |
| 23 | `score_time_ease_hl365_lambda1000` | 0.435187 | 0.421017 |

### val_random_communitypop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_review_pseudocat_affinity` | 0.535307 | 0.541270 |
| 2 | `score_cw_weighted_implicit_logit` | 0.533907 | 0.538651 |
| 3 | `score_time_affinity_mean` | 0.530906 | 0.533721 |
| 4 | `score_time_itemknn_hl730_sum` | 0.530806 | 0.540605 |
| 5 | `score_time_itemknn_hl365_sum` | 0.530706 | 0.540394 |
| 6 | `score_time_itemknn_hl90_sum` | 0.530606 | 0.540288 |
| 7 | `score_time_itemknn_hl730_top3` | 0.529706 | 0.536965 |
| 8 | `score_time_itemknn_hl365_top3` | 0.529406 | 0.536800 |
| 9 | `score_time_itemknn_hl90_top3` | 0.529406 | 0.536619 |
| 10 | `score_review_pseudocat_blend` | 0.528706 | 0.534249 |
| 11 | `score_graph_svd_k64` | 0.524005 | 0.526382 |
| 12 | `score_text_len_affinity` | 0.516703 | 0.520947 |
| 13 | `score_next_blend_mean_z` | 0.513003 | 0.516553 |
| 14 | `score_next_blend_priority_z` | 0.505901 | 0.503905 |
| 15 | `score_hours_affinity` | 0.501600 | 0.502789 |
| 16 | `score_review_pseudocat_log_pop` | 0.500100 | 0.500263 |
| 17 | `score_item_log_pop` | 0.495099 | 0.496598 |
| 18 | `score_item_recency_log_pop365` | 0.494999 | 0.496141 |
| 19 | `score_time_affinity_last` | 0.494699 | 0.496028 |
| 20 | `score_time_ease_hl365_lambda1000` | 0.460992 | 0.449002 |
| 21 | `score_icpns_comm_global_blend` | 0.445689 | 0.437019 |
| 22 | `score_icpns_comm_log_pop` | 0.441388 | 0.432850 |
| 23 | `score_icpns_comm_rate` | 0.441388 | 0.432850 |

### val_recent_communitypop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_review_pseudocat_blend` | 0.556111 | 0.565365 |
| 2 | `score_review_pseudocat_affinity` | 0.551710 | 0.557119 |
| 3 | `score_time_itemknn_hl365_top3` | 0.529106 | 0.541623 |
| 4 | `score_time_itemknn_hl730_top3` | 0.529106 | 0.541623 |
| 5 | `score_time_itemknn_hl90_top3` | 0.528906 | 0.541341 |
| 6 | `score_time_itemknn_hl90_sum` | 0.527305 | 0.538325 |
| 7 | `score_time_itemknn_hl730_sum` | 0.527205 | 0.538360 |
| 8 | `score_time_itemknn_hl365_sum` | 0.527205 | 0.538219 |
| 9 | `score_review_pseudocat_log_pop` | 0.524205 | 0.527692 |
| 10 | `score_graph_svd_k64` | 0.504101 | 0.506254 |
| 11 | `score_text_len_affinity` | 0.503301 | 0.508115 |
| 12 | `score_next_blend_mean_z` | 0.496299 | 0.500175 |
| 13 | `score_item_recency_log_pop365` | 0.493199 | 0.487231 |
| 14 | `score_time_affinity_last` | 0.491298 | 0.484843 |
| 15 | `score_item_log_pop` | 0.491098 | 0.483926 |
| 16 | `score_hours_affinity` | 0.488498 | 0.493212 |
| 17 | `score_time_ease_hl365_lambda1000` | 0.481896 | 0.476155 |
| 18 | `score_next_blend_priority_z` | 0.473195 | 0.475053 |
| 19 | `score_cw_weighted_implicit_logit` | 0.454291 | 0.452192 |
| 20 | `score_icpns_comm_global_blend` | 0.435087 | 0.414767 |
| 21 | `score_icpns_comm_log_pop` | 0.429886 | 0.409511 |
| 22 | `score_icpns_comm_rate` | 0.429886 | 0.409511 |
| 23 | `score_time_affinity_mean` | 0.396079 | 0.391215 |

## Promotion interpretation

- `score_cw_weighted_implicit_logit` is the low-cost PURL/CW proxy: it trains a weighted implicit classifier on fold-train positives and community-reliable sampled negatives.
- `score_icpns_*` and the `val_*_communitypop_seed42` splits are the ICPNS-style exposure/community validation work.
- `score_time_itemknn_*` and `score_time_ease_*` are TFPS-style time-decay graph probes.
- `score_review_pseudocat_*` uses only train reviews to create pseudo semantic categories; it avoids external Steam metadata.
- Any future submission candidate must still beat the existing Stage2 gates and must be approved explicitly by 우현 before Kaggle submission.
