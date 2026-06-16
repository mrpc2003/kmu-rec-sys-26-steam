# BSVD Boundary Seed-Vote Decoder — 3-Split Panel Aggregate

## 결론
- **판정: REJECT / NO-SUBMIT.** fixed BSVD variant가 predeclared panel gate를 통과하지 못했습니다.
- mean Δ=+0.000100, fixes/breaks=143/137, p=0.765149, direction=2/3.
- 범위: validation-only. candidate/test/submission CSV 생성 및 Kaggle submit 없음.

## Split Results
| split | base acc | BSVD acc | Δ | fixes | breaks | p | changed | boundary oracle UB | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| val_random_uniform_seed42 | 0.765053 | 0.765353 | +0.000300 | 42 | 36 | 0.571587 | 78 | +0.200740 | REJECT |
| val_random_uniform_seed7 | 0.760952 | 0.760752 | -0.000200 | 50 | 54 | 0.768792 | 104 | +0.204241 | REJECT |
| val_random_uniform_seed123 | 0.759952 | 0.760152 | +0.000200 | 51 | 47 | 0.762036 | 98 | +0.203441 | REJECT |

## Gate Policy
- fixed variant only: `score_bsvd_boundary_vote_w10cap20`
- mean Δ ≥ +0.00355
- pooled fixes > breaks and exact paired/binomial p < 0.05
- direction split ≥ 2/3
- split별 best, width tuning, tie-break tuning 금지
- no candidate/test/submission CSV; no public LB probing

## 해석
- BSVD는 seed-level top-half vote를 boundary band에만 적용하는 독립적인 decoder probe였지만, panel gate를 통과하지 못했다면 마지막 제출권을 쓸 근거가 없다.
- boundary oracle UB는 라벨을 사용한 진단 상한일 뿐이며, variant 선택/튜닝에 사용하지 않았다.
