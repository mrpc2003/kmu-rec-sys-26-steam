# KMU RecSys 26 Steam — AI-Q deep research 큐레이션 (next-step validation)

작성일: 2026-05-30 KST
원본: `reports/20260530_aiq_deep_next_raw.json` (AI-Q deep_researcher job `23659d8d-d79f-430e-89cb-c074d6a5093f`, status=success)
쿼리: `reports/20260530_aiq_recsys_next_query.md`

이 문서는 AI-Q deep researcher 결과를 **로컬 검증 사실과 분리해** 정리한 큐레이션이다. AI-Q 출력은 연구 보조 자료이며, 이 대회의 실제 코드/데이터/규칙에 대한 권위가 아니다. Kaggle 제출은 수행하지 않았다.

## 1. AI-Q가 제안한 핵심 프레임 (검증된 로컬 사실과 일치)

- 다음 단계는 새 모델 sweep이 아니라 **대회 메커니즘에 맞는 validation framework**다.
- 모든 실험은 global balanced Accuracy + `sqrt-pop` + `recent` + `pop-bin` + per-user top-half로 동시에 채점해야 한다.
- 기존 Stage2/z-score blend와 public anchor 0.74594를 비교 기준으로 유지한다.
- 핵심 위험은 잘못된 negative 분포에 대해 검증하는 것이다(MNAR: missing-not-at-random).

이 프레임은 우리 로컬 관찰(`sqrt-pop`/`recent`/`pop-bin` stress split, per-user top-half)과 정확히 일치한다.

## 2. AI-Q 제안 4개 액션 (우리 우선순위와 정합)

| 우선순위 | 액션 | AI-Q 권장 1차 실험 | 주 stress gate | reject 기준 |
|---:|---|---|---|---|
| 1 | ICPNS community-aware negatives/validation split | KMeans `k={16,32}`, smoothed community pop `pop_c(i)=count_c(i)+beta*global_pop(i)`, `gamma=0.5` sampling | sqrt-pop, pop-bin, top-half | cluster가 seed에 불안정하거나 high-pop bin에서만 gain |
| 2 | Time-decay matrix (TFPS-lite) | half-life `{90,180,365}`, binary/log-hours, ItemKNN BM25 먼저 | recent 우선, pop-bin neutral 이상 | global top-half가 0.002~0.003 이상 하락 |
| 3 | CW-lite / PU-inspired weighted implicit scorer | logistic/pairwise, `w_pos=1+alpha*log_hours_clipped`, `w_neg=clip(a*sqrt_pop+b*comm_pop+c*recent_pop)` | 4개 gate 전부 | AUC/logloss만 좋아지고 top-half Accuracy gain 없음 |
| 4 | Train-only review pseudo-category | TF-IDF→SVD64→KMeans64, `text_cos`/`same_cluster_count`만 먼저 blend | pop-bin 우선 | blend gain 없음 또는 재현 부담만 큼 |

AI-Q의 우선순위(ICPNS split → temporal ItemKNN → CW-lite → text)는 우리가 이미 세운 next-step 계획과 동일하다.

## 3. AI-Q가 제공한 신뢰할 만한 출처

AI-Q `sources.counts`: found=40, cited=21, report=20, artifact_entries=61.

핵심 인용(우리 검색과 교차 확인됨):

- MNAR/exposure-aware 평가: Yang et al. 2018 (Unbiased Offline Recommender Evaluation for MNAR Implicit Feedback)
- PU learning under MNAR implicit feedback (PMC12839574)
- ICPNS: In-Community Popularity Negative Sampling — arXiv 2602.18759
- PURL: Unbiased Recommender Learning from Implicit Feedback via Weakly Supervised Learning — OpenReview `0E5rZOGA13` (ICML 2025)
- Correct and Weight loss — arXiv 2601.04291
- Hu, Koren, Volinsky — Collaborative Filtering for Implicit Feedback Datasets (confidence-weighted ALS)
- EASE — arXiv 1905.03375
- TFPS — arXiv 2602.22521
- Steam recommendation 사례: LightFM Steam, thehir0/steam-recsys, arXiv 2305.04890
- 경고: "Do Recommender Systems Really Leverage Multimodal Content?" — arXiv 2508.04571 (content feature가 자동으로 도움 되지 않는다는 reject 근거)

이 출처들은 이전 라운드 arXiv/OpenReview 검색 결과와 겹치며, 핵심 paper(PURL, ICPNS, TFPS, CW loss, Hu-Koren, EASE)를 보강한다.

## 4. AI-Q 권장 중 우리가 채택/유보한 항목

채택:

- 모든 실험에 4-gate(global/sqrt-pop/recent/pop-bin) + per-user top-half 일괄 채점 강제.
- ICPNS smoothed community popularity 공식 `count_c(i)+beta*global_pop(i)`와 `gamma=0.5` sampling.
- CW-lite를 full PURL/PPT보다 먼저 시도(저비용, 재현 쉬움).
- Hu-Koren confidence `c_ui=1+alpha*r_ui`를 positive weight 기반으로 사용.
- 텍스트는 저차원 보조 feature로만, pop-bin gate 우선.

유보/주의:

- full PURL class-prior 추정은 user별 true positive rate 변동에 취약하므로 CW-lite 신호 확인 후로 미룸.
- sentence-transformers 임베딩은 eCampus 재현성(모델 revision/캐시 해시) 확보 전까지 TF-IDF 경로 우선.
- 과도한 time decay는 evergreen game과 sparse user를 해칠 수 있어 half-life sweep과 sparse-user 보호 필수.

거부 신호:

- 어떤 방법이든 top-half Accuracy gain 없이 AUC/logloss만 좋아지면 승격 금지.
- popularity에만 의존(=high-pop bin에서만 gain)하면 승격 금지.

## 5. 재현성/규칙 경계 (AI-Q와 우리 규칙 합치)

- `train.json`과 `pairs.csv`만 사용. Steam metadata 수집 금지.
- hidden label 추론/역추적 금지, public LB 피드백을 validation으로 쓰지 않음.
- 공개 pretrained 모델은 모델 identity/version/cache/추론 설정이 재현 가능할 때만 사용.
- 최종 evaluator는 per-user top-half를 강제하고 global/sqrt-pop/recent/pop-bin Accuracy를 출력하는 단일 결정적 스크립트로 둔다.

## 6. 후속

이 큐레이션의 권장은 `scripts/paper_guided_next_steps.py`로 코드화되어, ICPNS-style community split, CW-lite weighted implicit logit, time-decay graph score, review pseudo-category를 validation-only로 실행한다. 결과는 `reports/20260530_paper_guided_next_steps.md`/`.json`에 정리한다.
