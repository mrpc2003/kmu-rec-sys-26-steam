# KMU RecSys 26 Steam — Deep EDA Report

이 리포트는 제공 데이터만 사용했으며 Kaggle 제출/외부 Steam 정답 수집은 수행하지 않았다.

## 1. Candidate/test 구조
- pairs rows: 19,998, users: 4,737, games: 2,429
- cold users/games: 0 / 0; train에 이미 있는 user-game pair: 0
- 유저별 후보 수는 모두 짝수이며 top-half label=1 총합이 9,999개로 정확히 50%다.
- 후보 수 분포: {'2': 2141, '4': 1307, '6': 665, '8': 296, '10': 158, '12': 74, '14': 42, '16': 18, '18': 14, '20': 8, '22': 4, '24': 2, '26': 4, '30': 2, '36': 1, '38': 1}
- 구조상 유저별 hidden positive 수는 `candidate_count/2`로 알려져 있다: 평균 2.11, 중앙값 2, p95 5, max 19.
- 이 hidden positive 수와 train user degree의 상관은 0.647로 높다. 단순 `floor(train_n * frac)` 규칙만으로는 최대 match가 약 47%라서, holdout 수는 활동량 기반이되 노이즈/샘플링 규칙이 섞인 것으로 보인다.
- pairs.csv row order는 유저별로 거의 섞여 있다: 직전 row와 같은 user인 경우 4회뿐이고, 유저별 ID span 중앙값은 11,143이다.

## 2. Train 분포의 강한 신호
- train rows/users/games: 175,000 / 6,710 / 2,437
- item popularity Gini: 0.5214; user activity Gini: 0.3131
- 상위 게임 coverage: {'25%': {'n_games': 109, 'pct_games': 0.044727123512515385}, '50%': {'n_games': 351, 'pct_games': 0.1440295445219532}, '75%': {'n_games': 887, 'pct_games': 0.3639720968403775}, '90%': {'n_games': 1568, 'pct_games': 0.6434140336479278}, '95%': {'n_games': 1926, 'pct_games': 0.7903159622486664}}
- 결론: popularity는 단일 feature가 아니라 모든 CF/graph/text score에 들어갈 calibration 축이다.

## 3. Pair users/games selection shift
- pair users median train degree: 22.0, non-pair users median: 18.0
- pair games median train popularity: 38.0, non-pair games median: 27.5
- actual candidate pair item_n median: 62.0
- user-conditioned uniform unseen negative item_n median: 38.0
- popularity-weighted unseen negative item_n median: 123.0
- 유저별 unseen item을 `popularity^alpha`로 샘플링한다고 가정해 후보 item popularity 분포에 맞춰보면 alpha≈0.5가 가장 가까웠다(KS 0.0366, median 62). 따라서 validation negative는 uniform(too easy)과 popularity-weighted alpha=1(too popular) 사이의 sqrt-popularity / bin-matched hard negative를 포함해야 한다.
- 결론: validation negative를 uniform만 쓰면 실제 pairs보다 너무 쉬울 가능성이 높다. popularity-matched 또는 mixed sampler가 필요하다.

## 4. Candidate pair score feature 관찰
- `item_n` median/mean/p95: 62 / 129.5 / 483
- `hist_item_cos_max` median/mean/p95: 0.06855 / 0.07481 / 0.1376
- `hist_item_cos_top3_mean` median/mean/p95: 0.05889 / 0.06469 / 0.1187
- `hist_item_cooc_sum` median/mean/p95: 78 / 188.7 / 741
- `candidate_pop_z_vs_user_hist` median/mean/p95: -0.6578 / -0.3159 / 1.72
- `hist_htr_weighted_cos` median/mean/p95: 0.0238 / 0.0293 / 0.06761
- 결론: 후보 pair마다 item popularity, user-history item similarity, hours-weighted similarity를 모두 만들 수 있다. 이들은 BPR/LightGCN 외의 strong rank features로 쓸 수 있다.

## 5. Temporal/text feature 관찰
- train date range: 2010-10-15 ~ 2018-01-05
- year counts: {'2010': 503, '2011': 2659, '2012': 3376, '2013': 8910, '2014': 27357, '2015': 37646, '2016': 43667, '2017': 49848, '2018': 1034}
- user degree vs last-date correlation: 0.1569
- item degree vs last-date correlation: 0.2564
- 결론: test에 date가 없어도 user/item aggregate recency, active_days, recent-popularity weighting은 validation에서 실험 가치가 있다.
- blank review rows: 838; text length median/mean/p95: 130.0 / 391.0 / 1637.0
- 텍스트는 pair별 직접 텍스트가 없으므로 user profile/item profile aggregate embedding 또는 topic vector 방식으로만 쓰는 것이 안전하다.

## 6. 검증 설계 권고
1. 기본 LOO: 유저별 마지막 또는 무작위 positive 1개 holdout + 같은 유저의 unseen negative 1개.
2. 실제 pairs 난이도 반영: negative sampler는 uniform, popularity-weighted, popularity-bin matched, item-similarity hard negative를 모두 만들고 split별 rank correlation을 본다.
3. scoring은 반드시 per-user top-half accuracy로 통일한다. 후보 수가 모두 짝수이므로 validation도 유저별 짝수 후보 구조로 만든다.
4. public/private mismatch 방지를 위해 seed 5개 이상의 repeated LOO와 user-group split을 병행한다.

## 7. 실험 우선순위
- Tier 0: baseline 재현, filename mismatch 수정, deterministic pipeline/sha 기록.
- Tier 1: popularity variants + per-user rank, item co-occurrence/cosine/KNN, EASE/SLIM/ALS/BPR 튜닝.
- Tier 2: LightGCN/NGCF, recency-weighted graph, hours_transformed confidence.
- Tier 3: FM/GBM ranker with user/item/pair features, text/user-profile embeddings, rank aggregation ensemble.
- 제출은 후보 CSV 생성 후에도 우현 승인 전까지 금지.

## 8. Generated plots
- `reports/deep_eda/train_interactions_by_year.png`
- `reports/deep_eda/item_pop_all_vs_pair_games.png`
- `reports/deep_eda/user_degree_all_vs_pair_users.png`
- `reports/deep_eda/candidate_count_per_user.png`
- `reports/deep_eda/candidate_pop_vs_history_similarity.png`
- `reports/deep_eda/within_user_similarity_half_pop_gap.png`
