# KMU RecSys 26 Steam — paper-guided next-step run 해석

작성일: 2026-05-30 KST
원본 산출물:
- `scripts/paper_guided_next_steps.py`
- `reports/20260530_paper_guided_next_steps.md`
- `reports/20260530_paper_guided_next_steps.json`
- `reports/20260530_paper_guided_next_steps_run.log`
- `reports/20260530_aiq_deep_next_raw.json`
- `reports/20260530_aiq_deep_next_curated.md`

이 라운드는 AI-Q deep researcher 큐레이션 결과를 코드로 옮겨, 다음 4개 paper family를 **로컬 validation-only**로 실행했다. Kaggle 제출은 수행하지 않았다.

1. ICPNS-style community-aware negatives와 community-pop validation split (`val_*_communitypop_seed42`)
2. CW/PU-inspired weighted implicit logit (`score_cw_weighted_implicit_logit`)
3. TFPS-style time-decay ItemKNN/EASE (`score_time_itemknn_*`, `score_time_ease_*`)
4. Train-only review pseudo-category (`score_review_pseudocat_*`)

각 split에서 모든 score를 같은 candidate set/per-user top-half 변환으로 평가했다.

## 1. Split별 best와 Stage2 anchor 비교

기존 Stage2 anchor (이전 라운드 결과):
- `val_random_sqrtpop_seed42` 0.659732
- `val_recent_sqrtpop_seed42` 0.626025
- `val_random_popbin_seed42` 0.590818

이번 라운드 best:

| split | best score | row acc | Stage2 대비 | 해석 |
|---|---|---:|---:|---|
| `val_random_sqrtpop_seed42` | `score_next_blend_mean_z` | 0.645929 | -0.0138 | 새 blend는 Stage2를 넘지 못함 |
| `val_recent_sqrtpop_seed42` | `score_time_itemknn_hl90_top3` | 0.619524 | -0.0065 | time-decay ItemKNN이 Stage2에 거의 근접 |
| `val_random_popbin_seed42` | `score_next_blend_mean_z` | 0.563413 | -0.0274 | 새 blend는 Stage2 대비 손실 |
| `val_random_communitypop_seed42` | `score_review_pseudocat_affinity` | 0.535307 | (신규) | 매우 어려운 split, popularity는 무력화됨 |
| `val_recent_communitypop_seed42` | `score_review_pseudocat_blend` | 0.556111 | (신규) | recent + community에서 review pseudo-cat이 가장 강함 |

요약: 이번 라운드의 단일 또는 mean-z blend는 **Stage2를 깨지 못했다**. 하지만 paper family별로 의미 있는 신호가 분리되어 보였다.

## 2. Paper family별 신호 분석

### 2.1 ICPNS — community-aware negatives/validation

- 새로 만든 `val_random_communitypop_seed42`/`val_recent_communitypop_seed42`는 popularity를 거의 무력화한다. `score_item_log_pop`이 0.495/0.491로 떨어진다.
- 이는 ICPNS 의도와 정확히 맞다: community 내 인기 unobserved item이 negative이면 popularity-only 모델은 random에 가깝게 된다.
- ICPNS 자체 score(`score_icpns_comm_*`)는 sqrt-pop split에서는 popularity와 유사한 수준(0.619)이고, communitypop split에서는 오히려 random 이하(0.435~0.446)로 나왔다. 단독 scorer로는 그대로 쓰기 어렵다.
- 가치는 score보다 **새로운 stress gate** 그 자체에 있다. 이후 모델은 communitypop split에서 무너지지 않는지를 추가로 검증해야 한다.

### 2.2 TFPS-style time-decay ItemKNN

- `score_time_itemknn_*`가 가장 robust한 단일 axis다.
- `val_random_sqrtpop_seed42` 0.6431 (Stage2 0.6597과 비교, -0.017).
- `val_recent_sqrtpop_seed42` 0.6195 (Stage2 0.6260과 비교, -0.007). `hl=90/365/730`이 거의 동일.
- `val_random_popbin_seed42` 0.5583 (Stage2 0.5908과 비교, -0.033).
- communitypop split에서도 0.529~0.531로 random 이상.
- half-life 30~730은 거의 무차이. half-life 자체보다 **time-weighted graph weight + ItemKNN top-3 normalization**이 효과의 본체로 보인다.
- 결론: 이미 Stage2에 들어 있는 ItemKNN BM25에 time-decay edge weight를 주입하는 형태로 정식 통합 후보.

### 2.3 TFPS-style time-decay EASE

- `score_time_ease_hl365_lambda1000`가 모든 split에서 0.35~0.48로 약함. 단일 lambda/half-life 한 점만 본 것이라 결정적이지 않지만, 현재 구현/하이퍼는 부적절.
- 후속에서 EASE는 lambda sweep + 행/열 정규화 교차로 다시 검증해야 한다. 현재 코드는 시간 가중 행렬에 단일 lambda를 적용하는 단순 버전이며, 이 자체가 EASE 신호를 죽일 가능성이 있다.

### 2.4 CW/PU weighted implicit logit

- `score_cw_weighted_implicit_logit`가 모든 split에서 0.5 이하로 떨어졌다.
  - `val_random_sqrtpop` 0.4345
  - `val_recent_sqrtpop` 0.3983
  - `val_random_popbin` 0.5179
  - `val_random_communitypop` 0.5339
  - `val_recent_communitypop` 0.4543
- 이는 기능을 제대로 학습했다면 절대 안 나오는 수준이다. 가설:
  1. 학습에 사용한 feature 일부가 user별 정규화/재계산 없이 들어가 부호가 뒤집혔을 가능성.
  2. positive/negative weight 균형이 negative 쪽으로 과하게 기울어 모델이 popularity를 음의 방향으로 학습했을 가능성.
  3. train sample이 candidate-level이 아닌 row-level이라 candidate 내 ranking 정보를 잃었을 가능성.
- **결론: CW-lite 구현은 현재 형태로는 배제, 다음 실험 전 디버깅이 필요.** AI-Q 큐레이션 가이드의 reject 기준에도 명시한 항목이 그대로 발생했다.

### 2.5 Train-only review pseudo-category

- `score_review_pseudocat_affinity`/`score_review_pseudocat_blend`가 communitypop split에서 가장 좋다. `random_communitypop` 0.535, `recent_communitypop` 0.556로 popularity baseline(0.495, 0.491)보다 명확히 우위.
- sqrt-pop/popbin/recent에서는 단독 scorer로 popularity/time-decay ItemKNN보다 약하다.
- 결론: 단독 scorer가 아니라 **community/popularity가 통제된 환경의 보완축**으로 활용. 향후 EASE/ItemKNN과의 blend나 candidate-level rerank 후보.

## 3. 다음 라운드 우선순위 (검증 결과 기반)

1. **time-decay ItemKNN을 Stage2에 정식 통합**
   - 현재 ItemKNN BM25 노드에 edge time-decay weight를 주입하고 sqrt-pop/recent/popbin/communitypop 4-gate에서 z-blend gain을 측정.
2. **CW-lite 디버깅 후 재시도**
   - feature 부호/정규화/weight 분포 점검, candidate-level pairwise loss로 변경 검토.
3. **Review pseudo-category를 communitypop 스트레스 보완축으로 blend**
   - `score_review_pseudocat_affinity`만이라도 z-score blend에 추가했을 때 communitypop split에서 안정적인 gain이 있는지 확인.
4. **ICPNS validation을 표준 stress gate로 채택**
   - `val_*_communitypop_seed42`를 항상 평가하도록 evaluation harness에 등록.
5. **EASE time-decay는 lambda sweep + 정규화 교차**로 다시 시도. 현 구현은 inconclusive.

## 4. 게이트와 규칙 재확인

- 어떤 score도 **per-user top-half 제약**을 어기지 않았다.
- 모든 실험은 train-only 데이터로 작동한다(외부 Steam 수집/리버스 엔지니어링 없음).
- W&B에는 `no-submit` tag로만 기록 권장.
- 이 라운드 결과만으로 새로운 Kaggle 제출 후보를 만들지 않는다. Stage2 anchor를 넘기지 못했고, CW-lite는 디버깅이 필요하다.

## 5. 산출물 요약

| 종류 | 경로 | 설명 |
|---|---|---|
| 스크립트 | `scripts/paper_guided_next_steps.py` | ICPNS split + CW-lite + time-decay graph + review pseudo-cat 통합 실행기 |
| 실행 로그 | `reports/20260530_paper_guided_next_steps_run.log` | start/score/exit |
| 결과 표 | `reports/20260530_paper_guided_next_steps.md` | split별 full score table |
| 결과 JSON | `reports/20260530_paper_guided_next_steps.json` | split별 summary, community/text/cw meta |
| AI-Q 원본 | `reports/20260530_aiq_deep_next_raw.json` | deep_researcher job `23659d8d-...` 원본 |
| AI-Q 큐레이션 | `reports/20260530_aiq_deep_next_curated.md` | local 사실과 분리한 큐레이션 |
| 쿼리 | `reports/20260530_aiq_recsys_next_query.md` | next-step 전용 AI-Q 쿼리 |
| 산출물 디렉터리 | `artifacts/paper_guided_next_steps_20260530/` | split별 candidate score CSV (gitignored) |
| community split | `artifacts/validation/val_*_communitypop_seed42/` | ICPNS-style 신규 split (gitignored) |

큰 CSV/캐시는 `.gitignore`에 의해 커밋되지 않는다.
