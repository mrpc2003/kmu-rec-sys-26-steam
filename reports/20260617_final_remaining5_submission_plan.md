# 남은 5회 제출 소진 계획 — KMURecSys26 Steam

이 문서는 제출 실행이 아니라 최종 계획이다. live submission 목록을 다시 읽고, 이미 제출된 파일명과 exact SHA를 제외했다. 의미 있는 비중복 기준은 `min rowdiff vs live >= 500 labels`로 잡았다.

## 기본 원칙

- 목적은 “점수 개선”이 아니라 남은 5회를 중복 없이 소진하는 것이다.
- private 기대값 기준 final pair는 그대로 둔다: #1 emb128 4-seed, #2 rankblend emb128+emb192.
- 아래 5개는 exact SHA 중복이 아니고, live 제출 파일명과도 겹치지 않으며, 모두 live 제출물과 최소 500 label 이상 다르다.
- 결과가 나와도 다음 후보를 새로 최적화하지 않는다. 지금부터 결과 기반 재탐색을 하면 public overfit이다.

## 제출 순서

| 순서 | 파일 | SHA | min diff vs live | nearest live | diff vs rankblend | diff vs emb128 | 역할 |
|---:|---|---:|---:|---|---:|---:|---|
| slot1 | `submissions/candidate_smoke_tagcf_fulltest.csv` | `9d9c5b01` | 3144 | `candidate_autorun_tagcf_fulltest_seed2024_sym_a0.1_raw_zblend_bw0.5.csv` | 3280 | 3258 | TAGCF smoke: graph-family lottery. 기존 TAGCF 제출과 family는 같지만 fulltest smoke로 row-diff가 가장 큼. |
| slot2 | `artifacts/scores/test_pairs_full_train_stage2_cf/prediction_csv/candidate_score_als_f32_it30_alpha20_popa2.csv` | `dbeba05b` | 2604 | `candidate_score_blend_mean_z.csv` | 3264 | 3386 | Pure ALS CF baseline: rankblend/ALS-residual 제출과 다른 원본 ALS top-half 후보. |
| slot3 | `artifacts/scores/test_pairs_full_train_stage2_itemease/prediction_csv/candidate_score_itemknn_bm25_max.csv` | `16519ee1` | 3556 | `candidate_score_blend_mean_z.csv` | 4824 | 4888 | Pure itemKNN BM25 max baseline: item-neighborhood 계열의 강한 비상관 후보. |
| slot4 | `artifacts/scores/test_pairs_full_train_stage2_itemease/prediction_csv/candidate_score_rrf_pop_itemknn_ease.csv` | `644bc71b` | 2796 | `candidate_score_blend_mean_z.csv` | 3134 | 3142 | RRF pop+itemKNN+EASE hybrid: pure ALS/itemKNN과 다른 고전 CF ensemble 후보. |
| slot5 | `artifacts/scores/test_pairs_full_train_stage2_blend/prediction_csv/candidate_score_blend_median_z.csv` | `af5362b4` | 906 | `candidate_score_blend_mean_z.csv` | 2800 | 2908 | Stage2 median-z blend: 이미 제출한 mean-z와 다른 rank/score blend baseline. |

## 제출 전 공통 preflight

- schema: `ID,Label` 또는 `ID,Played`
- rows: 19,998
- labels: binary `{0,1}`
- per-user top-half: `bad_users_tophalf = 0`
- duplicate block: live filename 중복 없음, exact SHA 중복 없음, `min_rowdiff_vs_live >= 500` 유지

## 실행 규칙

1. slot1부터 slot5까지 한 번씩만 제출한다.
2. 제출 메시지에는 `FINAL5 non-duplicate burn`, slot 번호, SHA 앞 8자리, 후보 성격을 넣는다.
3. 각 제출 후 public score와 현재 best 대비 delta를 기록한다. 다만 중간 결과를 보고 새 후보를 끼워 넣지는 않는다.
4. 어떤 후보가 public best를 넘으면 그 파일만 final-pair 승격 후보로 표시하고, 그래도 즉시 추가 파생 제출은 하지 않는다.
5. 5개 중 하나가 preflight에서 막히면 그 slot은 건너뛰고 중단한다. 같은 family의 임의 대체 후보를 즉석에서 넣지 않는다.

## 해석

이 계획은 “남은 5회를 다 쓰되 중복 제출은 피한다”는 요구에 맞춘 burn plan이다. 점수 향상 기대값은 낮다. 최종 선택은 현재 결론대로 emb128 재현 후보와 rankblend hedge를 기본값으로 유지한다.
