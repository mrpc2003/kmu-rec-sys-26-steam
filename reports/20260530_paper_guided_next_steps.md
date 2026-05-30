# KMU RecSys 26 Steam — paper-guided next-step run

Validation-only run covering ICPNS-style community negatives, CW/PU-inspired weighted implicit scorer, time-decay graph scores, and train-only review pseudo-categories. No Kaggle submission was performed.

## Community-aware validation splits built

| split | rows | users | positives | negatives | communities |
|---|---:|---:|---:|---:|---:|

## Best score by split

| split | best score | row acc | per-user mean acc | duration sec |
|---|---|---:|---:|---:|
| `val_random_sqrtpop_seed42` | `score_next_blend_mean_z` | 0.653231 | 0.670520 | 449.445 |
| `val_recent_sqrtpop_seed42` | `score_time_itemknn_bm25_hl90_top3` | 0.622825 | 0.631988 | 451.815 |
| `val_random_popbin_seed42` | `score_time_itemknn_bm25_hl730_top3` | 0.569914 | 0.584741 | 441.968 |
| `val_random_communitypop_seed42` | `score_cw_weighted_implicit_logit` | 0.553411 | 0.555119 | 426.157 |
| `val_recent_communitypop_seed42` | `score_review_pseudocat_blend` | 0.556111 | 0.565365 | 428.162 |

## Full score tables

### val_random_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_next_blend_mean_z` | 0.653231 | 0.670520 |
| 2 | `score_time_itemknn_bm25_hl90_top3` | 0.650130 | 0.668042 |
| 3 | `score_time_itemknn_bm25_hl365_top3` | 0.650130 | 0.667937 |
| 4 | `score_time_itemknn_bm25_hl730_top3` | 0.650130 | 0.667937 |
| 5 | `score_time_itemknn_hl365_top3` | 0.643129 | 0.659704 |
| 6 | `score_time_itemknn_hl730_top3` | 0.643029 | 0.659598 |
| 7 | `score_time_itemknn_hl90_top3` | 0.643029 | 0.659387 |
| 8 | `score_time_itemknn_hl365_sum` | 0.642629 | 0.659996 |
| 9 | `score_time_itemknn_hl730_sum` | 0.642629 | 0.659996 |
| 10 | `score_time_itemknn_hl90_sum` | 0.642328 | 0.659573 |
| 11 | `score_icpns_comm_global_blend` | 0.619124 | 0.633986 |
| 12 | `score_icpns_comm_log_pop` | 0.618824 | 0.633590 |
| 13 | `score_icpns_comm_rate` | 0.618824 | 0.633590 |
| 14 | `score_next_blend_priority_z` | 0.614923 | 0.617579 |
| 15 | `score_item_recency_log_pop365` | 0.614123 | 0.629458 |
| 16 | `score_item_log_pop` | 0.613823 | 0.629462 |
| 17 | `score_time_affinity_last` | 0.613623 | 0.629198 |
| 18 | `score_review_pseudocat_affinity` | 0.590618 | 0.610563 |
| 19 | `score_graph_svd_k64` | 0.582416 | 0.581615 |
| 20 | `score_cw_weighted_implicit_logit` | 0.567313 | 0.574202 |
| 21 | `score_review_pseudocat_blend` | 0.556111 | 0.570515 |
| 22 | `score_time_affinity_mean` | 0.545409 | 0.552125 |
| 23 | `score_text_len_affinity` | 0.533407 | 0.537982 |
| 24 | `score_review_pseudocat_log_pop` | 0.529806 | 0.537731 |
| 25 | `score_hours_affinity` | 0.524105 | 0.531221 |
| 26 | `score_time_ease_hl365_lambda1000` | 0.352671 | 0.339050 |

### val_recent_sqrtpop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_time_itemknn_bm25_hl90_top3` | 0.622825 | 0.631988 |
| 2 | `score_time_itemknn_bm25_hl365_top3` | 0.622725 | 0.632011 |
| 3 | `score_time_itemknn_bm25_hl730_top3` | 0.622725 | 0.632011 |
| 4 | `score_time_itemknn_hl90_top3` | 0.619524 | 0.629951 |
| 5 | `score_time_itemknn_hl365_top3` | 0.619424 | 0.629942 |
| 6 | `score_time_itemknn_hl730_top3` | 0.619424 | 0.629942 |
| 7 | `score_next_blend_mean_z` | 0.619024 | 0.623835 |
| 8 | `score_time_itemknn_hl90_sum` | 0.613923 | 0.618501 |
| 9 | `score_time_itemknn_hl365_sum` | 0.613823 | 0.618254 |
| 10 | `score_time_itemknn_hl730_sum` | 0.613823 | 0.618254 |
| 11 | `score_review_pseudocat_affinity` | 0.592018 | 0.596684 |
| 12 | `score_item_recency_log_pop365` | 0.587518 | 0.587926 |
| 13 | `score_next_blend_priority_z` | 0.586817 | 0.582670 |
| 14 | `score_item_log_pop` | 0.586717 | 0.586757 |
| 15 | `score_time_affinity_last` | 0.586317 | 0.586405 |
| 16 | `score_icpns_comm_global_blend` | 0.585117 | 0.583600 |
| 17 | `score_icpns_comm_log_pop` | 0.581416 | 0.579008 |
| 18 | `score_icpns_comm_rate` | 0.581416 | 0.579008 |
| 19 | `score_review_pseudocat_blend` | 0.572214 | 0.575459 |
| 20 | `score_review_pseudocat_log_pop` | 0.546009 | 0.546933 |
| 21 | `score_cw_weighted_implicit_logit` | 0.544809 | 0.543252 |
| 22 | `score_graph_svd_k64` | 0.544309 | 0.539342 |
| 23 | `score_text_len_affinity` | 0.528506 | 0.530230 |
| 24 | `score_hours_affinity` | 0.505201 | 0.506690 |
| 25 | `score_time_affinity_mean` | 0.418784 | 0.417273 |
| 26 | `score_time_ease_hl365_lambda1000` | 0.385577 | 0.383681 |

### val_random_popbin_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_time_itemknn_bm25_hl730_top3` | 0.569914 | 0.584741 |
| 2 | `score_time_itemknn_bm25_hl365_top3` | 0.569714 | 0.584319 |
| 3 | `score_time_itemknn_bm25_hl90_top3` | 0.569514 | 0.584072 |
| 4 | `score_next_blend_mean_z` | 0.566513 | 0.580418 |
| 5 | `score_next_blend_priority_z` | 0.560012 | 0.559921 |
| 6 | `score_time_itemknn_hl90_sum` | 0.558312 | 0.575950 |
| 7 | `score_time_itemknn_hl365_sum` | 0.558212 | 0.575716 |
| 8 | `score_time_itemknn_hl730_sum` | 0.558112 | 0.575505 |
| 9 | `score_time_itemknn_hl365_top3` | 0.555911 | 0.566762 |
| 10 | `score_time_itemknn_hl90_top3` | 0.555911 | 0.566748 |
| 11 | `score_time_itemknn_hl730_top3` | 0.555811 | 0.566656 |
| 12 | `score_cw_weighted_implicit_logit` | 0.554511 | 0.554792 |
| 13 | `score_review_pseudocat_affinity` | 0.548310 | 0.560224 |
| 14 | `score_graph_svd_k64` | 0.548310 | 0.547524 |
| 15 | `score_time_affinity_mean` | 0.536107 | 0.540323 |
| 16 | `score_icpns_comm_log_pop` | 0.536007 | 0.546088 |
| 17 | `score_icpns_comm_rate` | 0.536007 | 0.546088 |
| 18 | `score_icpns_comm_global_blend` | 0.535607 | 0.545817 |
| 19 | `score_review_pseudocat_blend` | 0.526605 | 0.536195 |
| 20 | `score_time_affinity_last` | 0.521204 | 0.527157 |
| 21 | `score_item_log_pop` | 0.520804 | 0.526453 |
| 22 | `score_item_recency_log_pop365` | 0.520404 | 0.526669 |
| 23 | `score_text_len_affinity` | 0.518704 | 0.522390 |
| 24 | `score_hours_affinity` | 0.506501 | 0.509339 |
| 25 | `score_review_pseudocat_log_pop` | 0.496299 | 0.497815 |
| 26 | `score_time_ease_hl365_lambda1000` | 0.435187 | 0.421017 |

### val_random_communitypop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_cw_weighted_implicit_logit` | 0.553411 | 0.555119 |
| 2 | `score_time_itemknn_bm25_hl90_top3` | 0.535707 | 0.540887 |
| 3 | `score_time_itemknn_bm25_hl730_top3` | 0.535407 | 0.540402 |
| 4 | `score_review_pseudocat_affinity` | 0.535307 | 0.541270 |
| 5 | `score_time_itemknn_bm25_hl365_top3` | 0.535307 | 0.540359 |
| 6 | `score_time_affinity_mean` | 0.530906 | 0.533721 |
| 7 | `score_time_itemknn_hl730_sum` | 0.530806 | 0.540605 |
| 8 | `score_time_itemknn_hl365_sum` | 0.530706 | 0.540394 |
| 9 | `score_time_itemknn_hl90_sum` | 0.530606 | 0.540288 |
| 10 | `score_time_itemknn_hl730_top3` | 0.529706 | 0.536965 |
| 11 | `score_time_itemknn_hl365_top3` | 0.529406 | 0.536800 |
| 12 | `score_time_itemknn_hl90_top3` | 0.529406 | 0.536619 |
| 13 | `score_review_pseudocat_blend` | 0.528706 | 0.534249 |
| 14 | `score_graph_svd_k64` | 0.524005 | 0.526382 |
| 15 | `score_next_blend_mean_z` | 0.521604 | 0.526673 |
| 16 | `score_text_len_affinity` | 0.516703 | 0.520947 |
| 17 | `score_next_blend_priority_z` | 0.514003 | 0.514635 |
| 18 | `score_hours_affinity` | 0.501600 | 0.502789 |
| 19 | `score_review_pseudocat_log_pop` | 0.500100 | 0.500263 |
| 20 | `score_item_log_pop` | 0.495099 | 0.496598 |
| 21 | `score_item_recency_log_pop365` | 0.494999 | 0.496141 |
| 22 | `score_time_affinity_last` | 0.494699 | 0.496028 |
| 23 | `score_time_ease_hl365_lambda1000` | 0.460992 | 0.449002 |
| 24 | `score_icpns_comm_global_blend` | 0.445689 | 0.437019 |
| 25 | `score_icpns_comm_log_pop` | 0.441388 | 0.432850 |
| 26 | `score_icpns_comm_rate` | 0.441388 | 0.432850 |

### val_recent_communitypop_seed42

| rank | score | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_review_pseudocat_blend` | 0.556111 | 0.565365 |
| 2 | `score_review_pseudocat_affinity` | 0.551710 | 0.557119 |
| 3 | `score_cw_weighted_implicit_logit` | 0.539108 | 0.539603 |
| 4 | `score_time_itemknn_bm25_hl730_top3` | 0.534607 | 0.542415 |
| 5 | `score_time_itemknn_bm25_hl90_top3` | 0.534607 | 0.542246 |
| 6 | `score_time_itemknn_bm25_hl365_top3` | 0.534507 | 0.542204 |
| 7 | `score_time_itemknn_hl365_top3` | 0.529106 | 0.541623 |
| 8 | `score_time_itemknn_hl730_top3` | 0.529106 | 0.541623 |
| 9 | `score_time_itemknn_hl90_top3` | 0.528906 | 0.541341 |
| 10 | `score_time_itemknn_hl90_sum` | 0.527305 | 0.538325 |
| 11 | `score_time_itemknn_hl730_sum` | 0.527205 | 0.538360 |
| 12 | `score_time_itemknn_hl365_sum` | 0.527205 | 0.538219 |
| 13 | `score_review_pseudocat_log_pop` | 0.524205 | 0.527692 |
| 14 | `score_next_blend_mean_z` | 0.510802 | 0.517970 |
| 15 | `score_next_blend_priority_z` | 0.509902 | 0.508465 |
| 16 | `score_graph_svd_k64` | 0.504101 | 0.506254 |
| 17 | `score_text_len_affinity` | 0.503301 | 0.508115 |
| 18 | `score_item_recency_log_pop365` | 0.493199 | 0.487231 |
| 19 | `score_time_affinity_last` | 0.491298 | 0.484843 |
| 20 | `score_item_log_pop` | 0.491098 | 0.483926 |
| 21 | `score_hours_affinity` | 0.488498 | 0.493212 |
| 22 | `score_time_ease_hl365_lambda1000` | 0.481896 | 0.476155 |
| 23 | `score_icpns_comm_global_blend` | 0.435087 | 0.414767 |
| 24 | `score_icpns_comm_log_pop` | 0.429886 | 0.409511 |
| 25 | `score_icpns_comm_rate` | 0.429886 | 0.409511 |
| 26 | `score_time_affinity_mean` | 0.396079 | 0.391215 |

## Promotion interpretation

- `score_cw_weighted_implicit_logit` is the low-cost PURL/CW proxy: it trains a weighted implicit classifier on fold-train positives and community-reliable sampled negatives.
- `score_icpns_*` and the `val_*_communitypop_seed42` splits are the ICPNS-style exposure/community validation work.
- `score_time_itemknn_*` and `score_time_ease_*` are TFPS-style time-decay graph probes.
- `score_review_pseudocat_*` uses only train reviews to create pseudo semantic categories; it avoids external Steam metadata.
- Any future submission candidate must still beat the existing Stage2 gates and must be approved explicitly by 우현 before Kaggle submission.
