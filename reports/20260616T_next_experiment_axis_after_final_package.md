# KMURecSys26 Steam — final package 이후 새 실험축 재검토

- 작성일: 2026-06-16 UTC
- 범위: `final_package/` 고정 이후, 새 public submit 없이 이어갈 수 있는 실험축 탐색
- 안전 상태: Kaggle submit 없음, `submissions/` 새 파일 생성 없음, full-test candidate materialization 없음

## 결론

지금 바로 이어서 돌릴 **safe internal candidate axis는 없다.** boundary v1 forced public fail까지 반영하면, 제공 데이터 내부에서 남은 축은 대부분 이미 닫힌 family의 retune이다. 새 실험을 계속하려면 아래 둘 중 하나여야 한다.

1. **공식 `gameID -> Steam appid` 매핑이 확보된 metadata 축**
2. **기존 LightGCN/CF retune이 아닌 완전 새 backbone의 validation-only smoke**

그 외 boundary/rankblend residual/ALS/OTTO/pseudo-label/SWA/ItemKNN/EASE/GF-CF/UserKNN/temporal/hours/semantic retune은 지금 새 축으로 보지 않는다.

## 근거 요약

현재 유지 후보는 이미 고정돼 있다.

| slot | 파일 | public | 의미 |
|---:|---|---:|---|
| 1 | `final_package/final_slot1_rank_blend_emb128_emb192_LABEL.csv` | 0.77825 | current public best 보존 |
| 2 | `final_package/final_slot2_lightgcn_emb128L4r3_4seed_LABEL.csv` | 0.77745 | 안정 재현 backbone |

닫힌 축은 다음 근거가 있다.

| family | 최신 근거 | 판정 |
|---|---|---|
| boundary v1 / ridge-fast row flip | `reports/20260615T_boundary_v1_forced_submission_closure.md`, public 0.77705, implied precision 약 0.432 | closed |
| pseudo-label transduction | `reports/20260613T0605KST_opencode_post_pseudo_margin_axis_raw_text.md`, margin filtering까지 평균 악화 | closed |
| checkpoint/SWA-like averaging | `reports/20260613T0106KST_checkpoint_avg_aggregate.md`, best mean Δ +0.000067 | closed |
| EASE/ItemKNN wide audit | `reports/20260612T213950KST_uniform_wide_ease_itemknn_aggregate.md`, best mean 0.742849 | closed |
| GF-CF spectral panel | `reports/20260612T214616KST_gfcf_uniform_panel_probe.md`, best blend mean 0.763786 < emb128 0.76505 | closed |
| DIN target-conditioned set encoder | `reports/din_set_encoder_closure_report.md`, d64 blend Δ -0.00230 vs emb128 4-seed | closed |
| temporal/hours/exact-K/UserKNN/jackknife | `reports/20260614T2102KST_progress_and_current_blockers.md` | weak, stalled, or no strict gate |

OpenCode adviser의 최신 판단도 `NO_SAFE_INTERNAL_AXIS`다.

- `reports/20260615T_opencode_after_boundary_forced_no_submit_raw_text.md`
- sentinel: `BOUNDARY_AFTERCARE_ADVISORY_DONE`

## 다음 실험축 후보

### 1. metadata mapping-gated content profile axis — 1순위, 현재 blocked

**조건:** 공식/과제 제공 `gameID -> Steam appid` 매핑이 있어야 한다. 역매핑, Steam user/profile/owned-games 조회, 리뷰 대조로 매핑을 찾는 방식은 하지 않는다.

허용·매핑이 확보되면 다음 validation-only probe가 가장 새롭다.

- item metadata: genre/tag/category/release-date 등 공개 item side information
- user profile: train positive item metadata 평균/TF-IDF profile
- score: candidate item vector와 user profile cosine 또는 small regularized linear score
- blend gate: emb128/rankblend와 within-user z/rank blend, 20 uniform panel 기준

**gate:** mean Δ >= +0.0040, worst split >= -0.0015, public-tested failed rows와 낮은 overlap. 통과 전 full-test candidate를 만들지 않는다.

현재는 `reports/20260613T082844KST_metadata_approval_feasibility.md`에서 확인했듯 `gameID`가 Steam appid가 아니고 repo에 매핑 파일이 없어 blocked다.

### 2. UltraGCN/constraint-loss backbone smoke — 2순위, low prior

기존 닫힌 LightGCN loss retune이 아니라, propagation-free constraint-loss backbone이면 아직 “완전 동일 family retune”은 아니다. 다만 구현 비용 대비 prior는 낮다.

**bounded design:**

- validation-only, `val_random_uniform_seed42` 1 split smoke부터 시작
- candidate scoring/decoding은 기존 `evaluate_tophalf`와 동일
- emb64 또는 emb128, 1 seed, 2시간 kill switch
- LightGCN emb128 single-seed 0.76205와 emb128 4-seed 0.76505를 둘 다 기준으로 비교

**kill gate:** solo가 0.760 이하이거나, emb128 4-seed와 50/50 z-blend가 +0.0007 미만이면 즉시 종료. seed42에서만 살아나면 3 split panel로 확장하되, full-test materialization은 금지.

이 축은 “해볼 수 있는 내부 실험”이지 제출 후보가 아니다. boundary/ALS/OTTO처럼 weak-positive residual을 다시 태우는 것보다 낫지만, 0.78795 gap을 설명할 가능성은 metadata보다 낮다.

### 3. validation-harness axis — 후보 생성이 아니라 리스크 절감용

새 점수 후보가 아니라, future probe를 걸러내는 도구로만 의미가 있다.

- 20 uniform panel에서 rankblend/emb128/boundary-failed rows의 overlap map 작성
- public-failed boundary rows와 새 probe flip rows overlap이 높으면 자동 reject
- same-family public negative transfer ledger를 guard로 쓰기

이 축은 final score를 올리는 실험이 아니라, 다음 실험을 덜 위험하게 만드는 preflight다.

## 지금 하지 말 것

- boundary v1/ridge-fast/boundary covariate/row flip 재시도
- rankblend + ALS/WMF residual weight retune
- OTTO coplay/reverse_recent weight retune
- pseudo-label top-k/top-margin 확장
- SWA/checkpoint averaging 단독 재시도
- EASE/ItemKNN/GF-CF/PPR/heat-kernel/BSPM 확장
- UserKNN broad fine-grid 재실행
- public LB 기반 threshold tuning
- `submissions/` 파일 생성 또는 Kaggle submit

## 추천 행동

1. 새 성능축을 진짜로 이어가려면 먼저 공식 매핑 여부를 확인한다.
2. 매핑이 없거나 “하지마” 결정이 유지되면, internal candidate search는 중단하고 final package를 유지한다.
3. 그래도 내부 실험을 하나만 더 한다면 UltraGCN/constraint-loss smoke를 validation-only로 제한하고, seed42 kill gate를 통과하지 못하면 바로 닫는다.

## 운영 메모

team-mode로 병렬 검토를 열려고 했지만, 현재 `team_create`가 mutually exclusive 필드 처리에서 빈 필드까지 제공된 것으로 판정해 생성이 거부됐다. 팀은 생성되지 않았으므로 닫을 팀도 없다. 이 문서는 직접 repo audit으로 작성했다.

NEXT_EXPERIMENT_AXIS_REVIEW_DONE
