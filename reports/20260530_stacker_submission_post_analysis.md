# Stacker 제출 사후 분석 — 실패 (2026-05-30 KST)

## 결과: 정직 OOF가 transfer되지 않음

| 후보 | Public | vs LightGCN(0.76245) |
|---|---:|---:|
| LightGCN (현 best, anchor 유지) | **0.76245** | — |
| stacker logreg | 0.75355 | **−0.00890** |

- 제출 파일: `artifacts/stacker_20260530/test_candidate/candidate_stacker_logreg_emb64_L3_reg1e-04.csv`
- SHA256: `ebd69b42548f4c48651905d54bd1e985d0bbbe57a7aa3db846b6ca05fdc637a0`
- 상태: `SubmissionStatus.COMPLETE`
- 제출 시각: 2026-05-30 12:03 UTC (21:03 KST)
- quota: 3/5 사용 (2 남음)

## 예측 vs 실제 — calibration 실패

| 지표 | 값 |
|---|---:|
| pooled GroupKFold OOF 예측 게인 | **+0.00907** |
| 실제 public 게인 | **−0.00890** |
| **transfer ratio** | **−0.981** |

정직한 user-level GroupKFold(누수 갭 3e-5)를 통과했음에도 public은 **거의 정확히 반대 방향**으로 움직였다. 이전 LightGCN의 transfer ratio가 +1.24로 건강했던 것과 극명히 대비된다.

## 실패 원인 가설

메타러너가 **검증 split의 negative 샘플러 인공물**을 학습했다:
- 최대 가중 피처가 `score_blend_mean_z`(+0.87)와 `log_pop`(−0.42), `wz_score_blend_mean_z`(−0.42).
- 우리 validation negatives는 sqrtpop/popbin/recent로 **인위적으로 구성**된 것이고, 이 분포에서 popularity·within-user Stage2 신호가 양성/음성을 가르는 데 유용했다.
- 그러나 hidden test의 실제 negative 분포는 이 구성과 다르므로, 메타러너가 학습한 "보정"이 test에서는 오히려 해가 됐다.
- LightGCN 단독은 이런 split-specific 보정을 하지 않아 더 robust하게 일반화.

## 핵심 교훈 (skill 정정 필요)

**정직한 GroupKFold OOF만으로는 충분한 제출 게이트가 아니다.** 누수(within-user feature leak)는 막았지만, **검증 negative 분포 자체에 메타러너가 과적합**하는 더 미묘한 문제는 GroupKFold가 잡지 못한다. 이 task에서 신뢰할 수 있는 게이트는:
- 단일 robust 모델(LightGCN)을 기본값으로 유지
- 메타러너/blend는 **negative 샘플러를 바꿔가며(out-of-distribution)** 검증해 모든 샘플러에서 일관되게 이겨야만 신뢰
- 또는 transfer ratio가 검증된 축(LightGCN 같은)만 제출

## 후속 조치

1. ✅ `reports/failed_axes.json`에 실패 축 등록 (correlation policy 포함)
2. ✅ anchor는 LightGCN 0.76245로 유지 (회귀 없음)
3. ⬜ skill `binary-played-prediction-and-stacking-honesty.md`의 "GroupKFold 통과 = 신뢰" 주장 정정
4. ⬜ 추가 제출 없음 — protocol상 chain-submit 금지, 신규 파일은 명시적 재승인 필요
5. sweep 완료(~01:40 KST) 후 best-config LightGCN 단독이 다음 제출 후보 (메타러너 아님)
