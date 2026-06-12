# KMURecSys26 Steam Kaggle 제출 결과 — final slot 1

제출 시각: 2026-06-12 23:08 KST

## 제출 파일

- file: `artifacts/final_ecampus_package_20260612T2257KST/csv/final_slot1_rank_blend_emb128_emb192_LABEL.csv`
- candidate: `final_slot1_rank_blend_emb128_emb192_LABEL`
- SHA256: `1d38c3edf7afae2bbcde9f3d548aa92276323379af2c0c143b66844f25cef052`
- schema: `ID,Label`
- rows: 19,998
- Label=1 / Label=0: 9,999 / 9,999

## Kaggle 결과

- status: `SubmissionStatus.COMPLETE`
- public score: `0.77825`
- leaderboard position after submission: 4위, team `mrpc2003`
- submission message: `FINAL slot1 normalized rank-blend emb128+emb192; same predictions as public-best candidate_rank_blend_emb128_emb192, header ID/Label for eCampus; SHA 1d38c3ed; no external metadata/hidden labels.`

이번 제출은 eCampus 패키지의 Slot 1 파일이다. 기존 public-best `candidate_rank_blend_emb128_emb192.csv`와 예측값은 같고, 제출 형식에 맞추기 위해 두 번째 컬럼명만 `Played`에서 `Label`로 바꾼 파일이다. 점수도 기존 public-best와 같은 `0.77825`로 확인됐다.

## 안전 상태

- 사용자 제출 지시: `[우현] kaggle에 제출을 해야돼`
- 외부 Steam metadata 사용: 없음
- hidden label 사용: 없음
- 제출 전 preflight: `reports/20260612T2308KST_final_slot1_kaggle_submission_preflight.json`

## 다음 확인 사항

Slot 2인 emb128 4-seed 파일은 이미 과거 Kaggle 제출 기록에 있고 public score `0.77745`가 확인되어 있다. 같은 SHA의 파일을 다시 올리는 것은 새 정보가 거의 없으므로, Kaggle UI의 final selection 단계에서는 다음 두 제출을 선택하면 된다.

1. `final_slot1_rank_blend_emb128_emb192_LABEL.csv` — public `0.77825`
2. `candidate_lightgcn_emb128L4r3_seed_ens.csv` — public `0.77745`
