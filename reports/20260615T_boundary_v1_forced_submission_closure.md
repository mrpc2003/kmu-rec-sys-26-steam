# boundary v1 강제 제출 후속 정리

## 현재 결론

boundary v1은 더 밀지 않는다. scored panel20 gate가 이미 실패했고, 강제 public probe도 current best를 넘지 못했다.

## 제출 결과

| 항목 | 값 |
|---|---:|
| 제출 파일 | `submissions/candidate_boundary_v1_ridge_fast_panel20_forced.csv` |
| SHA256 | `ebca52d0ea2c185c3dc4770dbf471553213b1fd1cf6a0a5d5a2d22668117f4a6` |
| public score | 0.77705 |
| current best | 0.77825 |
| delta | -0.00120 |
| current best 대비 row diff | 176 |
| public changed rows 추정 | 88 |
| public net rows 추정 | -11.9988 |
| public implied precision 추정 | 0.431825 |

## 제출 전 gate

| 항목 | 값 |
|---|---:|
| scored panel20 complete splits | 20 |
| mean flip precision | 0.54537 |
| mean net gain rows | +6.3 |
| positive split ratio | 0.70 |
| worst split net gain rows | -14 |
| top2 gate pass | false |
| top1 gate pass | false |

이 후보는 원래 제출 대상이 아니었다. 사용자가 강제 public probe를 요청해서 `forced manual-risk`로 제출했다. 결과는 gate 판단과 같은 방향이었다.

## 운영 결정

- Slot A는 `candidate_rank_blend_emb128_emb192.csv` / `final_slot1_rank_blend_emb128_emb192_LABEL.csv` 0.77825를 유지한다.
- Slot B는 boundary v1로 교체하지 않는다.
- boundary proximity, ridge-fast row utility, 기존 boundary-family row flip은 닫는다.
- 같은 family를 다시 열려면 새 독립 신호가 먼저 필요하다. 조건은 scored panel20 gate 통과, worst split 급락 없음, 기존 public-failed boundary rows와 낮은 overlap이다.

## 기록 반영

- `reports/failed_axes.json`에 `boundary_v1_ridge_fast_panel20_forced_20260615` 항목을 추가했다.
- W&B run: `https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/h0qutxss`
