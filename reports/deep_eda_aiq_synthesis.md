# KMU RecSys 26 Steam — Deep EDA + AI-Q Synthesis

작성 기준: 제공된 Kaggle 데이터와 AI-Q deep research 결과를 결합했다. Kaggle 제출은 수행하지 않았다. 외부 Steam 정답 수집/스크래핑/리버스엔지니어링도 수행하지 않았다.

## 1. 실행 산출물

- 로컬 deep EDA 리포트: `reports/deep_eda/deep_eda_report.md`
- 로컬 deep EDA JSON: `reports/deep_eda/deep_eda_summary.json`
- 후보 pair feature preview: `reports/deep_eda/candidate_pair_engineered_features_preview.csv`
- 구조 진단 JSON: `reports/deep_eda/deep_structure_diagnostics.json`
- AI-Q deep research raw JSON: `reports/20260530T091933KST_aiq_recsys_played_research.json`
- AI-Q deep research rendered report: `reports/aiq_recsys_played_research_report.md`

AI-Q deep job id: `d10a5222-fb05-4c94-8d62-a0e73fdd0f61`, status `success`. AI-Q source count: found 47, cited 30, report citations 21.

## 2. 로컬 EDA 핵심 발견

### 2.1 Test/candidate 구조

- `pairs.csv`: 19,998 rows, 4,737 users, 2,429 games.
- cold user/game: 0 / 0.
- train에 이미 존재하는 user-game pair: 0.
- 모든 user의 candidate 수가 짝수다.
- 대회 설명의 per-user pos:neg=1:1 구조 때문에 user별 hidden positive 수는 `candidate_count / 2`로 정해진다.
  - mean 2.11, median 2, p95 5, max 19.
  - hidden positive count와 train user degree 상관: 0.647.
- `pairs.csv` row order는 user별로 거의 shuffle되어 있다.
  - 직전 row와 같은 user: 4회뿐.
  - user별 ID span median: 11,143.

해석: 이 대회는 global binary classifier가 아니라 **known-user/known-item constrained ranking** 문제다. 최종 출력은 반드시 user별 candidate score 정렬 후 top-half label=1로 만들어야 한다.

### 2.2 Popularity 및 negative-generation 힌트

후보 pair의 item train popularity 분포:

- 실제 candidate pair `item_n` median: 62.
- user-conditioned uniform unseen negative sample median: 38.
- user-conditioned popularity-weighted unseen negative sample median: 123.
- `popularity^alpha` negative sampler fitting 결과:
  - alpha≈0.5가 가장 유사: KS 0.0366, median 62.
  - alpha=0은 너무 쉬운 uniform negative.
  - alpha=1은 너무 head-heavy.

해석: validation negative를 uniform으로만 만들면 test보다 쉬워져 popularity/CF가 과대평가될 수 있다. 실제 후보 분포는 대략 **sqrt-popularity negative**에 가깝다. 따라서 validation은 `uniform`, `sqrt-popularity`, `popularity-bin matched`, `hard negative`를 모두 가져가야 한다.

### 2.3 Candidate pair engineered feature 신호

후보 pair별로 train만으로 만들 수 있는 주요 feature:

- `item_n` median/mean/p95: 62 / 129.5 / 483.
- `hist_item_cos_max` median/mean/p95: 0.0686 / 0.0748 / 0.1376.
- `hist_item_cos_top3_mean` median/mean/p95: 0.0589 / 0.0647 / 0.1187.
- `hist_item_cooc_sum` median/mean/p95: 78 / 188.7 / 741.
- `hist_htr_weighted_cos` median/mean/p95: 0.0238 / 0.0293 / 0.0676.

상관 구조:

- `item_n` vs `hist_item_cos_top3_mean`: 0.821.
- `item_n` vs `hist_htr_weighted_cos`: 0.862.
- `user_n` vs `candidate_count`: 0.749.

해석: candidate similarity feature는 상당 부분 popularity와 엮여 있다. 그래서 raw score를 그대로 쓰기보다 **user-wise rank/z-score normalization + user historical popularity mismatch penalty**가 필요하다.

### 2.4 Temporal/text

- train date range: 2010-10-15 ~ 2018-01-05.
- year counts는 2017이 가장 큼: 49,848 rows.
- user degree vs last-date correlation: 0.1569.
- item degree vs last-date correlation: 0.2564.
- blank text rows: 838.
- text length median/mean/p95: 130 / 391 / 1,637.

해석: test에 date/text가 직접 없으므로 pair-level feature는 user/item aggregate로만 가능하다. recency-weighted item-item, recent popularity, user/item text profile embedding은 2차 신호로 유효하지만, 초반 주력은 CF/EASE/ALS/BPR가 맞다.

## 3. AI-Q가 제안한 연구/모델 방향

AI-Q deep research는 이 문제를 “binary classification disguised as implicit-feedback ranking”으로 규정했다. 주요 권고는 다음과 같다.

1. Metric이 Accuracy라도 최종 의사결정은 per-user constrained ranking이다.
2. BPR+popularity baseline은 좋은 출발점이지만, item universe가 2,437개로 작기 때문에 EASE/SLIM/item-item이 매우 유망하다.
3. `hours_transformed`는 label이 아니라 implicit confidence로 사용해야 한다.
4. validation은 random LOO 하나로 부족하며, temporal split과 test-like negative sampler를 같이 써야 한다.
5. 최종 모델은 raw score보다 within-user rank, percentile, z-score, robust z-score를 blend하는 방식이 안전하다.
6. feature-rich 모델은 backbone이 아니라 reranker로 사용한다.
7. Kaggle public LB overfitting 방지를 위해 제출은 모델 계열별 다양성을 기준으로 제한해야 한다.

AI-Q가 우선순위 높게 추천한 모델군:

- Calibrated popularity.
- Item-item cosine/co-occurrence/BM25 KNN.
- EASE.
- SLIM.
- BPR variants.
- Implicit ALS.
- LightGCN.
- Random-walk proximity.
- FM/GBM/logistic/ridge reranker.
- Recency-weighted item-item.
- Review-text embedding profile similarity.

## 4. 검증 설계 권고

Primary local validation은 다음 구조로 만든다.

1. user별 holdout positive 수를 실제 `candidate_count/2`와 유사하게 맞춘다.
2. 해당 positive 수만큼 negative를 user별 unseen game에서 샘플링한다.
3. negative sampler portfolio:
   - uniform unseen,
   - popularity^0.5 unseen,
   - popularity-bin matched,
   - baseline-score hard negative,
   - mixed: 50% sqrt-pop + 25% uniform + 25% hard.
4. 모든 validation score는 row-level Accuracy보다 **per-user top-half Accuracy**를 우선한다.
5. 같은 candidate rows를 모든 모델에 재사용해 모델 간 비교가 흔들리지 않게 한다.
6. temporal validation에서는 feature 계산도 반드시 train period 내부로 제한한다.

## 5. 실험 우선순위

### Phase 0 — 재현/안전장치

- `pairs.csv` vs baseline의 `pairs_Played.csv` filename mismatch 처리.
- deterministic ID map, sparse matrix, candidate scorer, top-half writer 구현.
- 제공 baseline popularity/BPR 재현.
- submission CSV 생성은 가능하지만 Kaggle 제출은 우현 승인 전 금지.

### Phase 1 — 가장 싸고 강한 모델

- Calibrated popularity:
  - raw count, log count, recent count, user historical popularity preference.
- Item-item:
  - cosine, Jaccard, co-occurrence sum/max/top-k,
  - `hours_transformed` weighted,
  - BM25/TF-IDF style normalization.
- EASE:
  - binary / log-hour / BM25 matrix,
  - lambda broad sweep: 50, 100, 300, 1000, 3000.

### Phase 2 — factor models

- BPR sweep:
  - factors 32/64/128,
  - negative sampler uniform / pop^0.5 / pop^0.75 / hard,
  - popularity hybrid alpha tuning.
- Implicit ALS:
  - rank 64/128,
  - confidence `1 + alpha * hours_transformed`,
  - alpha 5/20/40/80.

### Phase 3 — ensemble/reranker

- Base model score files를 candidate row order 기준으로 저장.
- 각 score의 user-wise rank / percentile / z-score 생성.
- Weighted normalized blend, Borda/average rank, RRF 실험.
- OOF candidate table 기반 logistic/ridge/GBM/FM reranker.

### Phase 4 — recency/text/graph

- Recency-weighted item-item, recent popularity.
- Text embedding은 user profile/item profile 평균 embedding cosine으로 제한.
- LightGCN은 2~3 layers + BPR loss로 ensemble diversity 확인.

## 6. 당장 다음 실행 제안

다음 작업은 “제출”이 아니라 **validation harness + baseline 재현 + EASE/item-item prototype**이 좋다.

구체적으로 다음 파일을 만들면 된다.

- `scripts/build_validation_splits.py`
- `scripts/score_popularity_itemknn_ease.py`
- `scripts/evaluate_tophalf.py`
- `reports/validation_split_diagnostics.md`

성공 기준:

1. validation candidate가 실제 pairs의 user degree/candidate count/item popularity 분포를 따라가는지 확인.
2. popularity baseline과 BPR baseline을 top-half evaluation으로 재현.
3. EASE/item-item이 BPR+pop baseline보다 local blended validation에서 우세한지 확인.
4. public submission은 아직 하지 않는다.

## 7. 주요 참고문헌/출처

AI-Q report의 citation 중 이번 대회에 특히 직접적인 것:

- BPR: Bayesian Personalized Ranking from Implicit Feedback — https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf
- Collaborative Filtering for Implicit Feedback Datasets — http://yifanhu.net/PUB/cf.pdf
- EASE: Embarrassingly Shallow Autoencoders for Sparse Data — https://arxiv.org/pdf/1905.03375
- SLIM: Sparse Linear Methods for Top-N Recommender Systems — https://ohiostate.elsevierpure.com/en/publications/slim-sparse-linear-methods-for-top-n-recommender-systems
- LightGCN — https://arxiv.org/abs/2002.02126
- Graph Neural Networks in Recommender Systems Survey — https://arxiv.org/pdf/2011.02260
- Amazon item-to-item collaborative filtering — https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
- Factorization Machines — http://d2l.ai/chapter_recommender-systems/fm.html
