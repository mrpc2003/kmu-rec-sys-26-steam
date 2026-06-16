# KMURecSys26 Steam — pseudo-label 결과와 다음 SWA 축 실행

작성 시각: 2026-06-13 01:08 KST

## 1. pseudo-label transduction top-1 결과

- run root: `artifacts/pseudolabel_transduction_20260612T2312KST`
- aggregate: `reports/20260612T2312KST_pseudolabel_transduction_aggregate.md`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

| 설정 | runs | splits | mean student | mean teacher | mean Δ | min Δ | max Δ | positive runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top_n=1, margin=0 | 12 | 3 | 0.761927 | 0.761986 | -0.000058 | -0.001500 | +0.001900 | 6 |

Gate는 통과하지 못했다.

- required: mean Δ >= +0.0050, min Δ >= -0.0015, splits >= 3
- actual: mean Δ -0.000058, min Δ -0.001500, splits 3

해석: top-1 pseudo-positive의 label precision은 평균 0.8442로 높았지만, student 재학습으로는 안정적인 이득이 나오지 않았다. seed42 split에서는 평균 +0.000725가 나왔지만, seed7 split에서 평균 -0.000875가 발생했다. 이 정도면 1등 gap(+0.00970)을 닫는 축으로 보기 어렵다.

## 2. top_n 확장 보류 사유

사전 precision 진단상 top_n이 커질수록 pseudo-positive 오염이 빠르게 늘었다.

- top1 precision: 약 0.837~0.855
- top2 precision: 약 0.801~0.808
- top3 precision: 약 0.783~0.789

top1도 multi-split gate를 통과하지 못했으므로 top2/top3를 바로 full sweep하는 것은 기대값이 낮다. 다음 축은 OpenCode가 2순위로 둔 checkpoint prediction averaging/SWA 쪽으로 넘겼다.

## 3. 새로 시작한 축: LightGCN checkpoint prediction averaging

- background process: `proc_d052737234b8`
- run root: `artifacts/lightgcn_checkpoint_avg_20260613T0106KST`
- logs: `logs/lightgcn_checkpoint_avg_20260613T0106KST/`
- status: `reports/20260613T0106KST_checkpoint_avg_status.txt`
- summary: `reports/20260613T0106KST_checkpoint_avg_status.jsonl`
- aggregate 예정:
  - `reports/20260613T0106KST_checkpoint_avg_aggregate.json`
  - `reports/20260613T0106KST_checkpoint_avg_aggregate.md`

실험 구조:

- splits: `val_random_uniform_seed42`, `val_random_uniform_seed7`, `val_random_uniform_seed123`
- seeds: 42, 123, 2024, 7
- model: LightGCN emb128 L4 reg1e-3
- checkpoints: 120, 140, 160, 180, 200 epoch
- variants: each checkpoint score, last2 average, last3 average, all-checkpoint average

SWA/checkpoint averaging gate:

- mean Δ >= +0.0015
- min Δ >= 0.0000
- splits >= 3
- full-test materialization remains blocked until later explicit approval

## 4. 안전 상태

- Kaggle submit 추가 실행 없음
- `submissions/*.csv` count: 23 유지
- full-test scoring 없음
- candidate CSV 생성 없음
- 외부 Steam metadata 사용 없음

## 5. 다음 판정

`proc_d052737234b8` 완료 후 aggregate를 확인한다. checkpoint averaging이 gate를 통과하면 pseudo-label과 결합하기 전에 adversarial review를 먼저 둔다. gate를 통과하지 못하면 남은 내부 축은 margin-filtered pseudo-label 또는 구조가 다른 backbone으로 제한된다.
