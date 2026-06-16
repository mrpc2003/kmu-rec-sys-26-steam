# KMURecSys26 Steam — checkpoint averaging 결과와 margin pseudo-label 재실행

작성 시각: 2026-06-13 02:53 KST

## 1. checkpoint prediction averaging 결과

- run root: `artifacts/lightgcn_checkpoint_avg_20260613T0106KST`
- aggregate: `reports/20260613T0106KST_checkpoint_avg_aggregate.md`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

Best row:

```text
variant: score_avg_last3_160_200
splits: 3
mean acc: 0.762052
baseline: 0.761986
mean Δ: +0.000067
min Δ: -0.000300
max Δ: +0.000500
positive splits: 1 / 3
gate pass: false
```

Gate 기준은 mean Δ >= +0.0015, min Δ >= 0, splits >= 3이었다. 실제 best는 평균 +0.000067에 그쳤고 한 split만 양수였다. epoch 200 단독은 기준선과 동일하게 재현됐고, 더 이른 checkpoint나 넓은 평균은 오히려 성능을 깎았다. SWA/checkpoint averaging 축은 submit 후보가 아니다.

Overall table image: `reports/20260613T0106KST_checkpoint_avg_aggregate_table.png`

## 2. 다음 refill: margin-filtered pseudo-label

SWA 축이 닫혔으므로, pseudo-label 축에서 아직 남은 “score margin으로 boundary 근처를 버리는” 설정만 좁게 확인한다. top1 margin0은 평균 이득이 없었지만, margin threshold를 넣으면 pseudo-positive precision이 올라가고 coverage가 줄어든다.

사전 진단:

```text
margin 1.5: edges 약 3161~3218, precision 약 0.907~0.918
margin 2.5: edges 약 1756~1778, precision 약 0.942~0.946
```

새 background process:

```text
proc_bb273b9b6bd5
```

실행 경로:

```text
run root: artifacts/pseudolabel_margin_transduction_20260613T0246KST
logs: logs/pseudolabel_margin_transduction_20260613T0246KST/
status: reports/20260613T0246KST_pseudolabel_margin_status.txt
summary: reports/20260613T0246KST_pseudolabel_margin_status.jsonl
aggregate 예정: reports/20260613T0246KST_pseudolabel_margin_aggregate.md
```

실험 구조:

```text
margins: 1.5, 2.5
splits: val_random_uniform_seed42, val_random_uniform_seed7, val_random_uniform_seed123
student seeds: 42, 123, 2024, 7
jobs: 24
```

## 3. 현재 live 상태

첫 wave는 `val_random_uniform_seed42 / margin=1.5 / 4 seeds`로 올라갔다.

```text
GPU0: seed42
GPU1: seed123
GPU2: seed2024
GPU3: seed7
```

로그상 각 job은 epoch 1까지 정상 진입했다. GPU2에는 이전부터 PID 없는 stale 1274MiB가 남아 있지만, 현재 작업은 32GB V100에서 정상 실행 중이다.

## 4. 안전 상태

- Kaggle submit 추가 실행 없음
- `submissions/*.csv` count: 23 유지
- full-test scoring 없음
- candidate CSV 생성 없음
- 외부 Steam metadata 사용 없음

`proc_bb273b9b6bd5`가 끝나면 margin pseudo-label aggregate를 gate로 판정한다. 여기서도 mean Δ가 작거나 split 간 방향이 갈리면, 제공 데이터만 쓰는 local improvement axis는 더 좁아진다.
