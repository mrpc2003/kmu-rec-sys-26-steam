# KMU RecSys 26 Steam — paper-guided feature feasibility probe

이 리포트는 최신 추천시스템 논문 계열을 현재 Steam played prediction 과제에 적용하기 전, 로컬 validation에서 가볍게 검증한 탐색 결과다. Kaggle 제출은 수행하지 않았다.

## Split별 best probe

| split | best score | row acc | per-user mean acc |
|---|---|---:|---:|
| `val_random_sqrtpop_seed42` | `score_item_log_pop` | 0.614323 | 0.630155 |
| `val_recent_sqrtpop_seed42` | `score_item_recency_log_pop` | 0.587417 | 0.587820 |
| `val_random_popbin_seed42` | `score_blend_graph_time_hours` | 0.550410 | 0.549900 |
| `val_random_uniform_seed42` | `score_item_log_pop` | 0.720644 | 0.742005 |

## 논문 계열별 해석

- Graph/global collaborative signal: `score_svd_*`는 LightGCL류의 SVD/global-structure augmentation을 구현하기 전 cheap proxy다. 기존 ItemKNN/EASE/ALS보다 강하면 full graph-contrastive 구현 가치가 있다.
- Temporal filtration: `score_svd_recency_*`, `score_time_affinity`, `score_item_recency_log_pop`은 TFPS/시간 필터링 논문의 핵심인 시간 가중 positive 신호의 1차 proxy다.
- Negative-sampling/debiasing: 동일 feature를 sqrt-pop/recent/pop-bin/uniform split 모두에서 비교해 false-negative와 surrogate mismatch 위험을 본다.
- Review/text-enhanced recommendation: 현재는 text length/count proxy만 포함했다. 이 값이 약하면 LLM/TF-IDF review embedding은 단독 scorer보다 blend/regularizer로 시작한다.
- Hours/intensity: `score_svd_hours_*`, `score_hours_affinity`는 플레이 시간 강도 정보를 반영한다. Accuracy 과제에서는 intensity가 preference와 noise를 동시에 담을 수 있어 split 간 안정성이 중요하다.

## 전체 score table

### val_random_sqrtpop_seed42

| rank | score_col | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_item_log_pop` | 0.614323 | 0.630155 |
| 2 | `score_item_recency_log_pop` | 0.614123 | 0.629458 |
| 3 | `score_text_presence_pop` | 0.613923 | 0.629356 |
| 4 | `score_blend_graph_time_hours` | 0.594519 | 0.590892 |
| 5 | `score_blend_svd_family_mean` | 0.584917 | 0.584079 |
| 6 | `score_svd_binary_k64` | 0.582316 | 0.581404 |
| 7 | `score_svd_recency_k64` | 0.582116 | 0.581770 |
| 8 | `score_svd_hours_k64` | 0.567614 | 0.568634 |
| 9 | `score_time_affinity` | 0.545009 | 0.550922 |
| 10 | `score_text_len_affinity` | 0.533407 | 0.537982 |
| 11 | `score_hours_affinity` | 0.524105 | 0.531221 |
| 12 | `score_last_time_affinity` | 0.499100 | 0.498541 |

### val_recent_sqrtpop_seed42

| rank | score_col | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_item_recency_log_pop` | 0.587417 | 0.587820 |
| 2 | `score_text_presence_pop` | 0.586017 | 0.586347 |
| 3 | `score_item_log_pop` | 0.586017 | 0.586194 |
| 4 | `score_blend_svd_family_mean` | 0.546909 | 0.540742 |
| 5 | `score_blend_graph_time_hours` | 0.545809 | 0.540261 |
| 6 | `score_svd_binary_k64` | 0.544309 | 0.539342 |
| 7 | `score_svd_recency_k64` | 0.543809 | 0.539159 |
| 8 | `score_svd_hours_k64` | 0.539308 | 0.534274 |
| 9 | `score_text_len_affinity` | 0.528506 | 0.530230 |
| 10 | `score_hours_affinity` | 0.505201 | 0.506690 |
| 11 | `score_last_time_affinity` | 0.500200 | 0.499686 |
| 12 | `score_time_affinity` | 0.416983 | 0.414941 |

### val_random_popbin_seed42

| rank | score_col | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_blend_graph_time_hours` | 0.550410 | 0.549900 |
| 2 | `score_svd_recency_k64` | 0.548310 | 0.547735 |
| 3 | `score_svd_binary_k64` | 0.548310 | 0.547524 |
| 4 | `score_blend_svd_family_mean` | 0.546909 | 0.545927 |
| 5 | `score_time_affinity` | 0.536007 | 0.540454 |
| 6 | `score_svd_hours_k64` | 0.526005 | 0.526419 |
| 7 | `score_text_presence_pop` | 0.522104 | 0.528786 |
| 8 | `score_item_log_pop` | 0.521904 | 0.528985 |
| 9 | `score_item_recency_log_pop` | 0.520504 | 0.526880 |
| 10 | `score_text_len_affinity` | 0.518704 | 0.522390 |
| 11 | `score_hours_affinity` | 0.506501 | 0.509339 |
| 12 | `score_last_time_affinity` | 0.499000 | 0.498006 |

### val_random_uniform_seed42

| rank | score_col | row acc | per-user mean acc |
|---:|---|---:|---:|
| 1 | `score_item_log_pop` | 0.720644 | 0.742005 |
| 2 | `score_item_recency_log_pop` | 0.720244 | 0.742372 |
| 3 | `score_text_presence_pop` | 0.720044 | 0.741645 |
| 4 | `score_blend_graph_time_hours` | 0.637227 | 0.635324 |
| 5 | `score_blend_svd_family_mean` | 0.619524 | 0.623189 |
| 6 | `score_svd_recency_k64` | 0.616123 | 0.620185 |
| 7 | `score_svd_binary_k64` | 0.615623 | 0.619460 |
| 8 | `score_svd_hours_k64` | 0.609322 | 0.613481 |
| 9 | `score_time_affinity` | 0.547009 | 0.556914 |
| 10 | `score_text_len_affinity` | 0.543909 | 0.547903 |
| 11 | `score_hours_affinity` | 0.537307 | 0.544085 |
| 12 | `score_last_time_affinity` | 0.501300 | 0.501229 |
