# KMU RecSys 26 Steam — 최신 추천시스템 논문 기반 적용 탐색

작성일: 2026-05-30 KST
범위: Steam `userID, gameID` played 여부 예측 과제에 대해 최신 추천시스템 연구를 조사하고, **제출 없이** 로컬 validation-only probe로 적용 가능성을 점검한다.

## 0. 현재 과제에 맞춘 핵심 전제

- 문제는 명시 평점 예측이 아니라 **implicit feedback binary recommendation**이다.
- Test label은 played/non-played가 정확히 1:1이고, 현재 로컬 관찰상 user별 후보 수가 짝수라서 **user별 candidate ranking → top-half `Label=1` 변환**이 자연스럽다.
- Public LB는 test의 절반만 반영하므로, 논문 방법을 바로 제출 후보로 승격하지 않고 `sqrt-pop`, `recent`, `pop-bin` stress split을 함께 통과해야 한다.
- 사용할 수 있는 안전한 신호는 train의 `userID`, `gameID`, `text`, `date`, `hours`, `hours_transformed`, 그리고 train interaction graph뿐이다.
- 금지: 외부 Steam 리뷰 수집, hidden label 역추적, private test 복원, Public LB 반복 과최적화.

## 1. 조사한 최신/관련 paper family

| 우선순위 | paper / family | 핵심 아이디어 | 이 대회 적용 해석 | 1차 결론 |
|---:|---|---|---|---|
| 1 | **PURL: Unbiased Recommender Learning from Implicit Feedback via Weakly Supervised Learning** — ICML 2025 / OpenReview | unobserved item을 곧바로 negative로 보지 않고 positive-unlabeled/weak supervision 문제로 재정의. class prior를 추정해 negative sampling 의존을 줄임. | train에는 positive만 있고 후보 negative는 샘플링된 unobserved이므로 가장 직접적으로 맞는 family. 이 대회는 test 후보가 1:1이라 class-prior/threshold calibration을 user별 top-half와 결합해야 함. | **최우선**. 새로운 모델보다 validation/negative construction을 paper-grade로 개선하는 방향. |
| 2 | **ICPNS: In-Community Popularity Negative Sampling** — arXiv 2026 | user community를 찾고, 그 community 안에서 인기 있지만 해당 user가 보지 않은 item을 더 신뢰도 높은 negative로 사용. | 기존 sqrt-pop/pop-bin보다 더 test-like negative를 만들 수 있음. user embedding/ALS/SVD로 community를 만들고 community-pop unseen negatives로 validation split 재구성 가능. | **최우선 validation 개선 후보**. |
| 3 | **TFPS: Temporal Filtration-enhanced Positive Sample Set Construction** — arXiv 2026 | implicit CF에서 negative만이 아니라 positive set도 시간 간격/decay로 필터링해 현재 선호에 가까운 positive를 강화. | test에는 date가 없지만 train history에는 date가 있으므로 edge weight, recent-positive filtering, recent holdout gate로 적용 가능. | recent split에서 `item_recency_log_pop`가 best라서 **약한 양의 근거 있음**. |
| 4 | **Correct and Weight loss for Implicit Feedback Recommendation** — arXiv 2026 | false negative 영향을 objective에서 보정하고 sample별 weight를 부여. | BPR/ALS/LightGCN 학습 시 unobserved negative의 신뢰도를 user/item popularity, community exposure, hours/recency로 weighting. | 모델 구현 비용 중간, sampler 개선 후 적용. |
| 5 | **MDCNS / RoDPO / DynamicPO** — arXiv 2026 | hard negative를 deterministic하게 고정하지 않고 teacher/peer/self, dynamic top-K, DPO류 preference objective로 안정화. | 후보 set 내 ranking과 잘 맞지만 구현 비용이 큼. 현재는 candidate-level pairwise reranker로 축소 적용 가능. | 2단계 후보. 먼저 cheap pairwise reranker로 검증. |
| 6 | **LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation** — ICLR 2023 | user-item graph에 SVD 기반 global structure contrastive augmentation을 사용해 sparse/popularity bias를 완화. | 현재 EASE/ALS/ItemKNN 다음 단계의 graph CF 후보. 다만 cheap SVD proxy는 기존 blend보다 약했음. | full GCL은 바로 우선순위 아님. feature/blend 후보로 보류. |
| 7 | **RaDAR / GDA4Rec / graph diffusion-contrastive family** — arXiv 2025–2026 | random edge perturbation보다 relation-aware/diffusion/generative augmentation으로 graph noise를 줄임. | sparse Steam interaction graph에 맞지만 구현·튜닝 비용이 큼. Public half split overfit 위험도 높음. | 강한 validation signal 확보 전까지 보류. |
| 8 | **SymCERE: Robust Review-Enhanced Recommendation** — arXiv 2025 | behavior graph와 review text fusion에서 false negative/popularity bias/fusion gap을 contrastive loss로 완화. | train review text가 있어 적용 가능. 단 test에는 text가 없으므로 user/item text profile embedding만 사용해야 함. | TF-IDF probe가 약해 단독 scorer는 부적합, blend/regularizer로만. |
| 9 | **Toward User Preference Alignment in LLM Recommendation via Explicit Context Feedback** — arXiv 2026 | review/comment 같은 explicit contextual feedback을 LLM 기반 preference modeling에 적극 활용. | 공개 pretrained embedding/LLM으로 train review profile을 만들 수 있으나 eCampus 재현성과 비용이 관건. | item semantic cluster/user profile feature로 제한 적용. |
| 10 | **CPGRec+ / CPGRec video game recommendation** — TOIS/arXiv 2026 | game recommendation에서 category와 popularity를 균형 있게 사용해 정확도-다양성 trade-off를 다룸. | 외부 game category 수집은 금지에 가깝다. 대신 train review text나 interaction cluster에서 pseudo-category를 만들어 popularity와 결합 가능. | domain-specific 아이디어는 좋지만 외부 metadata 없이 내부 cluster로만 적용. |

## 2. 실제 로컬 적용 probe 결과

이번 라운드에서 논문 family를 바로 대형 모델로 구현하지 않고, **저비용 proxy**로 먼저 검증했다.

생성/실행한 스크립트:

- `scripts/paper_guided_feature_probe.py`
  - SVD/global graph proxy: `score_svd_binary_k64`, `score_svd_hours_k64`, `score_svd_recency_k64`
  - temporal/positive filtration proxy: `score_item_recency_log_pop`, `score_time_affinity`
  - hours/intensity proxy: `score_hours_affinity`, hours-weighted SVD
  - text coverage proxy: text length/count
- `scripts/review_tfidf_probe.py`
  - train review만 사용해 user profile과 item profile TF-IDF cosine을 계산
  - validation-only, no-submit
- `reports/20260530_arxiv_paper_search_results.json`
  - arXiv API 기반 검색 결과 원본
- `reports/20260530_opencode_paper_exploration_advice.log`
  - OpenCode advisory memo, no file edits

### 2.1 Paper-guided feature probe 요약

출처: `reports/20260530_paper_guided_feature_probe.md`

| split | best probe score | row acc | per-user mean acc | 해석 |
|---|---|---:|---:|---|
| `val_random_sqrtpop_seed42` | `score_item_log_pop` | 0.614323 | 0.630155 | popularity가 여전히 강함. SVD/time 단독은 기존 Stage2보다 약함. |
| `val_recent_sqrtpop_seed42` | `score_item_recency_log_pop` | 0.587417 | 0.587820 | 시간 가중 item popularity가 recent holdout에서 약하게 우세. TFPS류 근거. |
| `val_random_popbin_seed42` | `score_blend_graph_time_hours` | 0.550410 | 0.549900 | popularity가 통제된 split에서는 graph/time/hour blend가 가장 좋음. |
| `val_random_uniform_seed42` | `score_item_log_pop` | 0.720644 | 0.742005 | uniform split은 너무 쉬우며 실제 제출 판단에 단독 사용하면 위험. |

중요 비교:

- 기존 Stage2 blend는 `random_sqrtpop` 0.659732, `recent` 0.626025, `popbin` 0.590818 수준이다.
- 이번 cheap SVD/time/text proxy는 **단독으로는 기존 Stage2를 넘지 못했다**.
- 그러나 `popbin`에서 graph/time/hour blend가 popularity보다 높아, community-aware negative sampling 또는 calibrated blend에는 쓸 수 있다.

### 2.2 Review TF-IDF probe 요약

출처: `reports/20260530_review_tfidf_probe.md`

| split | best text-related score | row acc | per-user mean acc | 해석 |
|---|---|---:|---:|---|
| `val_random_sqrtpop_seed42` | `score_item_review_count` | 0.613923 | 0.629356 | text count는 사실상 item popularity proxy. |
| `val_recent_sqrtpop_seed42` | `score_item_review_count` | 0.586017 | 0.586347 | TF-IDF cosine은 0.585917로 근소하게 비슷하나 강하지 않음. |
| `val_random_popbin_seed42` | `score_review_tfidf_user_item_cosine` | 0.533807 | 0.544136 | popularity 통제 시 semantic profile이 약간 살아남음. |
| `val_random_uniform_seed42` | `score_item_review_count` | 0.720044 | 0.741645 | 쉬운 split에서는 popularity 계열이 압도. |

결론:

- review text는 **단독 ranking backbone으로는 부족**하다.
- 그래도 pop-bin처럼 popularity가 통제된 환경에서 TF-IDF cosine이 item review count보다 높으므로, LLM/review embedding을 완전히 버릴 필요는 없다.
- 적용 방식은 “LLM이 직접 정답 추론”이 아니라 **train review 기반 item semantic cluster / user profile feature / graph regularizer**로 제한해야 한다.

## 3. 다음 탐색 단계: paper → 구현 후보

### A. PURL/PU learning 계열 — 최우선

목표: unobserved를 모두 negative로 보는 현재 BPR/ALS 관성을 줄이고, test의 1:1 candidate 구조와 맞는 confidence calibration을 만든다.

구현 제안:

1. positive set: train observed interactions.
2. unlabeled set: user별 unseen item sample.
3. class prior:
   - validation candidate에서는 user별 positive ratio를 known으로 사용.
   - test candidate에서는 top-half 제약 때문에 candidate-level prior = 0.5를 후처리로 강제.
4. 모델:
   - logistic MF / LightFM-style feature model / shallow MLP 중 가장 재현 쉬운 것부터.
   - loss는 PU-risk 또는 corrected weighted BCE로 시작.
5. 평가:
   - `random_sqrtpop` primary.
   - `recent`, `popbin` stress.
   - 기존 Stage2 score와 stacking/blending했을 때 gain이 있는지 확인.

승격 조건:

- 단독 또는 blend가 `random_sqrtpop`에서 기존 Stage2 0.659732를 넘거나, 비슷하더라도 `recent/popbin`에서 명확한 보완축이어야 함.
- Public 제출 후보로 보려면 top-half 변환 후 ID/Label preflight와 재현 script가 필요.

### B. ICPNS/community-aware negative sampling — 최우선 validation 개선

목표: test negative가 uniform보다 sqrt-pop에 가깝다는 기존 관찰을 더 정교화한다.

구현 제안:

1. 기존 ALS/SVD/user embedding으로 user community를 만든다.
2. 각 community별 item popularity를 계산한다.
3. validation negative를 “해당 user community에서 인기 있지만 user가 보지 않은 item”에서 뽑는다.
4. hardness를 3단계로 둔다.
   - community-pop easy
   - community-pop + global-pop matched
   - model-disagreement hard negative
5. 기존 sqrt-pop/pop-bin split과 함께 evaluation portfolio로 유지한다.

기대 효과:

- 논문 방법 자체보다도 **Public LB half split에 덜 흔들리는 local gate**를 만들 가능성이 높다.
- 현재 첫 제출 public 0.74594가 local sqrt-pop보다 높게 나온 상태라, test-like validation을 더 맞추는 것이 가장 ROI가 높다.

### C. TFPS/time-aware positive filtering — 중간 우선순위

목표: 오래된 플레이 기록과 최근 선호를 구분한다.

이번 probe 근거:

- `recent` split에서 `score_item_recency_log_pop`가 단순 `score_item_log_pop`보다 근소하게 좋았다.
- 단, `score_time_affinity` 단독은 recent split에서 오히려 나빴다. 즉 user-item absolute date matching보다 **item-level recent popularity**가 더 안전하다.

구현 제안:

1. time-decay edge weight를 ALS/EASE/ItemKNN에 넣는다.
2. half-life: 90/180/365/730일 sweep.
3. 최근 interaction만 positive로 쓰는 filtered graph와 전체 graph를 별도 score로 만든다.
4. full score에 blend할 때 active user와 sparse user를 나눠 weight를 다르게 준다.

승격 조건:

- recent split 개선이 있고 sqrt-pop을 크게 해치지 않아야 함.
- pop-bin에서 neutral 이상이어야 함.

### D. LightGCL/RaDAR graph contrastive 계열 — 보류 후 2단계

이번 cheap proxy:

- `score_svd_binary_k64`, `score_svd_recency_k64`는 모든 split에서 기존 Stage2보다 낮다.
- pop-bin에서는 SVD 계열이 popularity보다 좋아서, popularity가 통제된 hard candidate에서는 collaborative latent signal이 있다.

해석:

- full GCL을 바로 구현하기보다는, 기존 ALS/EASE/ItemKNN blend에 **SVD/graph latent feature를 auxiliary axis**로 넣고 correlation/gain을 본 뒤 결정한다.
- full graph contrastive는 구현비가 높고 eCampus 재현성 설명도 길어진다.

### E. Review/LLM text-enhanced 계열 — 제한 적용

이번 TF-IDF probe:

- user-item review cosine은 `random_sqrtpop` 0.579016, `recent` 0.585917, `popbin` 0.533807, `uniform` 0.632226.
- pop-bin에서만 item review count보다 좋다.

권장 적용:

1. 공개 pretrained `sentence-transformers`/E5 계열로 train review item embedding을 만든다.
2. user embedding은 사용자가 남긴 train reviews의 평균/attention pooling으로 만든다.
3. 직접 score보다 다음 feature로 사용한다.
   - item semantic cluster popularity
   - user semantic cluster preference
   - candidate score residual calibration
   - graph edge dropout/false-negative filter
4. LLM prompt로 hidden label을 추론하는 방식은 금지/비재현 위험이 크므로 사용하지 않는다.

승격 조건:

- text axis가 기존 score와 낮은 correlation이면서 pop-bin/recent에서 보완 gain을 보여야 한다.
- public model, version, cache regeneration command가 명확해야 한다.

### F. Video-game-specific CPGRec 계열 — 내부 pseudo-category만 사용

CPGRec+류는 game category와 popularity 균형을 강조한다. 하지만 이 대회에서는 외부 Steam metadata/category 수집이 위험하다.

안전한 대체:

- interaction graph cluster = pseudo genre/community
- review text embedding cluster = pseudo semantic category
- item popularity × pseudo-category diversity/affinity feature
- user가 과거에 플레이한 pseudo-category 분포와 candidate category의 cosine/KL score

이 방식은 외부 metadata 없이 train만 사용하므로 규칙상 안전하다.

## 4. 모델 승격 게이트

논문 novelty만으로 제출 후보가 되면 안 된다. 다음 조건을 만족해야 한다.

| gate | 기준 |
|---|---|
| Primary validation | `val_random_sqrtpop_seed42`에서 기존 Stage2 또는 현재 best blend 대비 개선 |
| Stress validation | `val_recent_sqrtpop_seed42`, `val_random_popbin_seed42` 중 최소 하나에서 보완 gain, 다른 하나는 큰 하락 금지 |
| Uniform split | 참고용만 사용. 단독 근거로 제출 금지 |
| Top-half constraint | user별 positive count/top-half 변환 위반 0 |
| Reproducibility | script, seed, package, data fingerprint, output SHA256 기록 |
| Leakage safety | 외부 Steam data/hidden label/reverse engineering 없음 |
| Submission policy | 우현 승인 전 Kaggle 제출 없음 |

## 5. 우선 실행 계획

1. **Community-aware validation split 추가**
   - ICPNS에서 가져온 아이디어.
   - ALS/SVD user embedding → community → in-community popular unseen negative.
   - 목적: 지금보다 Public transfer를 더 잘 예측하는 local gate 확보.

2. **Corrected/weighted implicit loss 또는 confidence-weighted MF**
   - PURL/CW loss 계열.
   - negative confidence = global popularity + community popularity + model disagreement 기반.
   - output은 candidate score axis로 저장하고 기존 Stage2와 blend.

3. **Time-decay graph score sweep**
   - TFPS 계열.
   - EASE/ItemKNN/ALS에 half-life edge weight를 넣고 recent split과 sqrt-pop split 동시 확인.

4. **Text embedding은 low-cost feature로만 추가**
   - TF-IDF 결과가 약하므로 heavy LLM fine-tuning은 뒤로 미룸.
   - 먼저 public sentence embedding으로 item/user semantic cluster를 만들고 pop-bin 보완축인지 확인.

5. **Graph contrastive full implementation은 마지막**
   - cheap SVD proxy가 약했으므로, LightGCL/RaDAR 구현은 다른 축에서 한계가 확인된 뒤 진행.

## 6. 현재 라운드 산출물

- Paper search raw JSON: `reports/20260530_arxiv_paper_search_results.json`
- Query for AI-Q/deep research: `reports/20260530_research_paper_exploration_query.md`
- Paper-guided feature probe script: `scripts/paper_guided_feature_probe.py`
- Paper-guided feature probe report: `reports/20260530_paper_guided_feature_probe.md`
- Review TF-IDF probe script: `scripts/review_tfidf_probe.py`
- Review TF-IDF probe report: `reports/20260530_review_tfidf_probe.md`
- OpenCode advisory log: `reports/20260530_opencode_paper_exploration_advice.log`

AI-Q backend은 처음에는 8101/8100 모두 내려가 있어 실패했으나, `scripts/hermes_start_backend_codex_proxy_detached.sh`로 8101을 기동했다. 다만 deep researcher job은 대화 내 검증 시간 안에 완료되지 않아 중단했으며, 이번 결론에는 arXiv/OpenReview/Web 검색, OpenCode advisory, 로컬 validation probe만 사용했다.

## 7. 최종 판단

현재 증거상 “최신 논문을 그대로 큰 모델로 구현”하는 것보다, **negative sampling / PU-learning / community-aware validation / time-decay confidence**가 이 대회에 가장 잘 맞는다. 특히 첫 제출 public 0.74594가 local보다 높게 나온 상황에서는 모델 복잡도보다 **test-like validation과 calibrated blend**가 더 중요하다.

따라서 다음 실험은 `ICPNS-style community split + PURL/CW-style weighted implicit model`을 먼저 진행하는 것이 가장 합리적이다.
