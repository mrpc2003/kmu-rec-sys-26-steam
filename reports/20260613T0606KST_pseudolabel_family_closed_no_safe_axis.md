# KMURecSys26 Steam — pseudo-label family closure and no-safe-axis verdict

작성 시각: 2026-06-13 06:06 KST

## 1. margin-filtered pseudo-label 결과

- run root: `artifacts/pseudolabel_margin_transduction_20260613T0246KST`
- aggregate: `reports/20260613T0246KST_pseudolabel_margin_aggregate.md`
- status: `reports/20260613T0246KST_pseudolabel_margin_status.txt`
- validation only: true
- Kaggle submit: false
- full-test scoring: false
- candidate CSV written: false
- external metadata: false

Best row는 `top_n=1, min_margin=1.5`였지만 gate를 통과하지 못했다.

```text
margin 1.5
mean student: 0.761236
mean teacher: 0.761986
mean Δ: -0.000750
min Δ: -0.002801
max Δ: +0.001700
positive runs: 4 / 12
pseudo precision: 0.9109

gate pass: false
```

`min_margin=2.5`는 precision이 0.9444까지 올라갔지만 평균 성능은 더 나빠졌다.

```text
margin 2.5
mean Δ: -0.001134
min Δ: -0.004701
positive runs: 1 / 12
pseudo precision: 0.9444
```

해석은 명확하다. pseudo-label precision을 0.84에서 0.91/0.94로 끌어올려도 validation lift가 나오지 않는다. coverage가 줄고, seed7 split은 계속 음수다. 이건 pseudo label이 새 신호를 주는 게 아니라 teacher bias를 재주입하거나 graph training boundary를 흔드는 쪽에 가깝다.

## 2. OpenCode 후속 판정

- prompt: `reports/20260613T0605KST_opencode_post_pseudo_margin_axis_prompt.md`
- raw output: `reports/20260613T0605KST_opencode_post_pseudo_margin_axis_raw_text.md`
- log: `logs/20260613T0605KST_opencode_post_pseudo_margin_axis.jsonl`
- sentinel: `OPENCODE_POST_PSEUDO_MARGIN_AXIS_DONE`

OpenCode verdict:

```text
NO_SAFE_INTERNAL_AXIS
```

요지는 다음과 같다.

- pseudo-label transduction은 margin0, margin1.5, margin2.5까지 닫혔다.
- checkpoint/SWA averaging도 noise-scale이라 닫혔다.
- 이미 닫힌 내부 축이 model capacity, seed expansion, CF variants, graph filters, sequence models, stackers, exact-K losses, text/date/hours/popularity residuals까지 넓다.
- 현 public #1 gap +0.00970은 남은 작은 내부 tweak로 설명하기 어렵다.
- 남는 분기는 외부 Steam metadata 허용 여부다. 승인 전에는 수집/학습하지 않는다.

## 3. ledger와 monitor 반영

`reports/failed_axes.json`에 아래 세 축을 추가했다.

```text
pseudolabel_top1_margin0_transduction_20260612T2312KST
lightgcn_checkpoint_prediction_averaging_20260613T0106KST
pseudolabel_margin_filtered_transduction_20260613T0246KST
```

JSON parse 검증도 통과했다.

paused cron job `4d627b59804f`도 최신 상태로 업데이트했다. 나중에 다시 켜져도 pseudo-label/SWA/닫힌 CF 계열을 반복하지 않고, 새 승인 정보가 없으면 final packaging/status 쪽으로 가도록 바꿨다.

## 4. 현재 안전 상태

```text
Kaggle submit: 추가 실행 없음
submissions/*.csv count: 23
live project process: 없음
GPU: idle, 단 GPU2에 PID 없는 stale 1274MiB만 남음
candidate_csv_written true: 0
full_test_scored true: 0
```

## 5. 다음 선택지

이 상태에서 제공 데이터만 쓰는 내부 no-submit 탐색은 더 돌릴수록 중복 실험이 될 가능성이 높다. 현실적인 다음 선택지는 둘이다.

1. 교수자에게 Steam item metadata 사용 가능 여부를 확인한다.
2. 허용이 안 되거나 시간이 부족하면 current safe final fallback을 고정하고 eCampus 재현성/최종 제출 패키징으로 돌아간다.

외부 metadata가 허용되면 그때도 바로 full-test 후보를 만들지 않고, 먼저 validation-only metadata side-information probe부터 돌리는 게 맞다.
