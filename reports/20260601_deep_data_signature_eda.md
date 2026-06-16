# 2026-06-01 Deep Data-Signature EDA — KMU RecSys 26 Steam

분석 범위: 제공된 `train.json`, `pairs.csv`, 로컬 validation/score artifact만 사용했다. Kaggle 제출, hidden label 추정/외부 수집, submission CSV 생성은 수행하지 않았다.

## 1. 핵심 결론

1. 이 데이터는 **cold-start가 전혀 없는, 작은 item universe의 known-user/known-item constrained ranking**이다. bipartite graph의 largest component share가 1.0000이고 matrix density가 1.0702%라서 LightGCN/EASE류가 같은 공기(co-occurrence)를 거의 다 빨아먹기 쉽다.
2. `pairs.csv`의 모든 user candidate 수가 짝수이고 hidden positive 총량은 구조적으로 9,999개다. K=1 또는 2인 유저가 대부분이라, 긴 list 추천보다 **K/K+1 boundary** 문제가 본질이다.
3. 초반 EDA의 `candidate item popularity ≈ sqrt-pop` 결론은 절반만 맞다. 모든 candidate row를 negative처럼 맞추면 alpha≈0.5가 나오지만, **candidate는 positive 50% + negative 50% 혼합**이다. train-like held-out positive + uniform negative 혼합도 actual pair marginal을 강하게 설명한다. 따라서 public이 uniform surrogate를 따라간 기존 관찰과 모순되지 않는다.
4. pair-level side feature(`cooc`, `cos`, `hours`, `text`, `early_access`)는 대부분 item popularity와 confound되어 있다. raw AUC/상관이 있어 보여도 log-pop residual boundary AUC를 봐야 한다.
5. 남은 탐색은 새 encoder가 아니라 **K-aware objective / boundary-only fine-tune / residualized feature gate**처럼 데이터 구조를 직접 겨냥해야 한다.

## 2. Train/test 구조

- train rows/users/games: 175,000 / 6,710 / 2,437
- duplicate user-game rows: 0
- user×item density: 1.0702%
- user degree Gini / item degree Gini: 0.3131 / 0.5214
- graph components: 1개, largest component node share: 1.0000
- pairs rows/users/games: 19,998 / 4,737 / 2,429
- cold users/games rows: 0 / 0; train-pair overlap rows: 0
- candidate item count vs train item popularity corr(log-log): 0.6784

### K 분포

| K | users | user_share | median_train_n | mean_test_frac |
|---|---|---|---|---|
| 1 | 2141 | 0.4520 | 19.0000 | 0.0547 |
| 2 | 1307 | 0.2759 | 22.0000 | 0.0960 |
| 3 | 665 | 0.1404 | 28.0000 | 0.1200 |
| 4 | 296 | 0.0625 | 36.0000 | 0.1327 |
| 5 | 158 | 0.0334 | 50.0000 | 0.1289 |
| 6 | 74 | 0.0156 | 61.5000 | 0.1158 |
| 7 | 42 | 0.0089 | 86.0000 | 0.0905 |
| 8 | 18 | 0.0038 | 95.0000 | 0.1048 |
| 9 | 14 | 0.0030 | 106.0 | 0.0823 |
| 10 | 8 | 0.0017 | 125.0 | 0.0927 |
| 11 | 4 | 0.0008 | 83.0000 | 0.1205 |
| 12 | 2 | 0.0004 | 156.0 | 0.0908 |

해석: K=1/2가 지배적이므로, `NDCG@large K`식 방법보다 rank-`K`와 rank-`K+1` 사이를 직접 움직이는 방법이 더 적합하다.

## 3. Candidate generation signature

아래는 실제 `pairs.csv`의 item-pop/user-mean-pop/user-max-pop 분포를, `train history에서 K개 positive holdout + unseen negative K개` 시뮬레이션으로 맞춘 결과다. 낮을수록 actual pairs와 가깝다.

| mode | negative_alpha | ks_item_pop_all_rows | ks_user_mean_item_pop | ks_user_max_item_pop | mean_three_ks | corr_log_candidate_item_count_log_pop |
|---|---|---|---|---|---|---|
| random_positive | 0.0000 | 0.0155 | 0.0139 | 0.0165 | 0.0153 | 0.7191 |
| recent_positive | 0.0000 | 0.0162 | 0.0316 | 0.0347 | 0.0275 | 0.5154 |
| random_positive | 0.2500 | 0.0605 | 0.0645 | 0.0309 | 0.0520 | 0.7847 |
| recent_positive | 0.2500 | 0.0549 | 0.0648 | 0.0531 | 0.0576 | 0.6417 |
| recent_positive | 0.5000 | 0.0990 | 0.1213 | 0.0798 | 0.1000 | 0.7161 |
| random_positive | 0.5000 | 0.1066 | 0.1276 | 0.0712 | 0.1018 | 0.8327 |
| recent_positive | 0.7500 | 0.1500 | 0.1910 | 0.1164 | 0.1525 | 0.7758 |
| random_positive | 0.7500 | 0.1537 | 0.1984 | 0.1182 | 0.1568 | 0.8618 |

비교용으로 **모든 candidate row를 negative라고 잘못 가정**하면 다음처럼 alpha≈0.5가 best처럼 보인다.

| negative_alpha | ks_if_all_rows_were_negative |
|---|---|
| 0.5000 | 0.0366 |
| 0.7500 | 0.1087 |
| 0.2500 | 0.1132 |
| 0.0000 | 0.1974 |
| 1.0000 | 0.2014 |
| 1.2500 | 0.2913 |

해석: actual pair marginal이 sqrt-pop처럼 보였던 것은 positive half의 popularity skew가 섞인 효과다. 실제 negative half가 uniform-like일 가능성을 배제하지 않으며, 이미 public LB가 uniform split과 가장 가까웠던 경험적 사실을 유지해야 한다.

## 4. Current-best validation error anatomy

분석 score: `artifacts/last_slot_rankagg/rankagg_seed42_scores.csv` / `base_emb128_raw_mean`
- row accuracy: 0.76505
- per-user mean accuracy: 0.79024

### Accuracy by K

| bucket | rows | accuracy |
|---|---|---|
| (0.999, 2.0] | 9508 | 0.7966 |
| (2.0, 3.0] | 3990 | 0.7489 |
| (3.0, 4.0] | 2368 | 0.7568 |
| (4.0, 19.0] | 4130 | 0.7128 |

### Actual pairs vs calibrated uniform validation feature KS

| feature | ks_actual_pairs_vs_uniform_val | actual_median | val_median |
|---|---|---|---|
| candidate_count | 0.0001 | 6.0000 | 6.0000 |
| user_n | 0.0789 | 27.0000 | 25.0000 |
| item_n | 0.0209 | 62.0000 | 59.0000 |
| hist_cos_top3 | 0.0464 | 0.0589 | 0.0567 |
| hist_cooc_sum | 0.0619 | 78.0000 | 64.0000 |
| candidate_pop_z_vs_user_hist | 0.0120 | -0.6578 | -0.6553 |

### Feature-only signal and base-score correlation

| feature | row_auc_label | corr_with_base_score | feature_tophalf_acc |
|---|---|---|---|
| item_n | 0.7571 | 0.7370 | 0.7206 |
| item_log_n | 0.7571 | 0.8047 | 0.7206 |
| hist_cos_top3 | 0.7458 | 0.7612 | 0.7213 |
| hist_cooc_sum | 0.7308 | 0.6373 | 0.7331 |
| candidate_pop_z_vs_user_hist | 0.7441 | 0.5919 | 0.7206 |
| item_htr_mean | 0.6337 | 0.4682 | 0.6100 |
| item_text_mean | 0.4300 | -0.2541 | 0.4389 |
| item_early_rate | 0.5094 | 0.0856 | 0.5040 |

### K/K+1 boundary delta AUC

| boundary_delta_feature | raw_auc_top_candidate_is_true_positive | logpop_residual_auc |
|---|---|---|
| d_item_n | 0.6639 | 0.4602 |
| d_item_log_n | 0.6775 | None |
| d_hist_cos_top3 | 0.6864 | 0.5719 |
| d_hist_cooc_sum | 0.6610 | 0.4824 |
| d_candidate_pop_z_vs_user_hist | 0.6560 | 0.4220 |
| d_item_htr_mean | 0.5753 | 0.5002 |
| d_item_text_mean | 0.4548 | 0.4998 |
| d_item_early_rate | 0.5136 | 0.4147 |

해석: base score 자체가 이미 popularity/co-occurrence 계열을 흡수했다. feature-only top-half accuracy가 높더라도 base와 상관이 높으면 새 축이 아니다. boundary에서 log-pop residual AUC가 0.55 부근 이하인 feature는 마지막 슬롯 실험으로 승격하지 않는다.

## 5. 이 EDA가 탐색 전략을 어떻게 바꾸는가

### 유지해야 할 것
- 1차 gate는 계속 `val_random_uniform_seed42` 및 3-split uniform panel이다. 전체 candidate marginal만 보고 sqrt-pop을 primary로 되돌리면 안 된다.
- 모든 검증은 user별 `K_u = candidate_count_u / 2`를 고정한 top-half decoding으로 해야 한다.
- popularity/co-occurrence/text/hour feature는 raw gain이 아니라 **base 대비 residual + paired McNemar**로만 판단한다.

### 새로 구체화된 probe
1. **SL@K / TopKGAT-lite K-aware boundary fine-tune**: K가 작고 구조적으로 알려져 있으므로 가장 데이터 구조에 직접 맞는다. 단, old-loss continuation(A) vs new-loss fine-tune(B)로 분기해 `B-A`만 gate한다.
2. **Ambiguity-only objective**: 모든 pair를 다시 학습하지 말고 rank `K/K+1` 근방, boundary margin 하위 분위 유저만 loss에 크게 반영한다. DNS와 달리 hard-negative pool이 아니라 실제 metric boundary에 조건을 건다.
3. **Mixture-faithful validation refresh**: 새 probe마다 random-positive+uniform-negative split뿐 아니라 recent-positive+uniform-negative stress도 같이 본다. positive holdout mode가 바뀌어도 sign-stable한 축만 남긴다.
4. **Residualized multi-interest cheap probe**: multi-interest를 하더라도 raw cooc/cos 말고 log-pop residualized history-similarity로 segment를 나눠야 한다. residual boundary AUC가 낮으면 구현하지 않는다.

### 더 이상 우선하지 않을 것
- 전체 candidate item-pop 분포를 맞추기 위한 pop-bias 추가/감산: positive+negative mixture confound 때문에 public-transfer trap이다.
- 긴 sequence/transformer/diffusion 모델: K=1/2 유저가 대부분이고 no-cold small graph라, 새 encoder는 base score와 상관만 올라갈 가능성이 크다.
- item-level global quota: candidate item marginal은 label marginal이 아니고, positive half와 negative half가 섞여 있어 per-user decision에 균일 적용하면 회귀하기 쉽다.

## 6. Safety

- `kaggle competitions submit` 호출 없음
- submission CSV 생성 없음
- hidden label 외부 수집/추론 없음
- 산출물: JSON/Markdown EDA report only
