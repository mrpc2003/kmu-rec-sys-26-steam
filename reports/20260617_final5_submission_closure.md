# 남은 5회 제출 실행 결과 — KMURecSys26 Steam

사용자 승인에 따라 남은 5회를 모두 사용했다. 새 모델을 만들거나 외부 메타데이터를 쓰지 않았고, 사전에 정한 비중복 후보 5개만 `ID,Label` 형식으로 materialize해 제출했다.

## 결과

| slot | upload | public | Δ vs 0.77825 | min diff vs prior live | 판정 |
|---|---|---:|---:|---:|---|
| slot1 | `final5_slot1_tagcf_smoke_LABEL.csv` | 0.72674 | -0.05151 | 3144 | close |
| slot2 | `final5_slot2_pure_als_popa2_LABEL.csv` | 0.73304 | -0.04521 | 2604 | close |
| slot3 | `final5_slot3_itemknn_bm25_max_LABEL.csv` | 0.69333 | -0.08492 | 3556 | close |
| slot4 | `final5_slot4_rrf_pop_itemknn_ease_LABEL.csv` | 0.73704 | -0.04121 | 2796 | close |
| slot5 | `final5_slot5_stage2_median_z_LABEL.csv` | 0.74174 | -0.03651 | 906 | close |

## 해석

- 5개 모두 `SubmissionStatus.COMPLETE`로 끝났고, UTC 기준 오늘 제출 카운트는 `5/5`가 됐다.
- exact filename/SHA 중복은 없었고, 각 후보는 기존 live 제출물과 최소 906~3556 label 이상 달랐다.
- 점수는 모두 현재 public best `0.77825`보다 크게 낮다. 즉 남은 5회는 중복 없이 소진됐지만, final pair를 바꿀 근거는 없다.
- 최종 선택은 그대로 유지한다: #1 emb128 4-seed 재현 후보, #2 rankblend emb128+emb192 hedge.
- 이번 결과는 고전 CF/TAGCF/Stage2 diversity만으로는 saturated LightGCN/rankblend 신호를 이기지 못한다는 실패축 기록으로 `reports/failed_axes.json`에 반영했다.
