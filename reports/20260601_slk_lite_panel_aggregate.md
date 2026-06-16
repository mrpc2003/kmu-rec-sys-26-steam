# SL@K-lite Top-K Objective Probe — 3-Split Panel Aggregate

## 결론
- **판정: REJECT.** best fixed epoch 기준 평균 Δ가 음수이며, fixes>breaks 방향성도 0/3으로 실패했습니다.
- Best fixed epoch by mean delta: **epoch 1**, mean Δ=-0.009035, aggregate exact p=8.59e-19, fixes/breaks=1603/2145.
- 제출 파일/hidden test 접근 없이 validation-only summary 및 validation score CSV만 생성했습니다.

## Gate Policy
- mean Δ(control 대비) ≥ +0.00355
- aggregate exact binomial/McNemar 방향 p < 0.05 and fixes > breaks
- split별 fixes > breaks가 2/3 이상
- fixed epoch panel 기준, split cherry-pick 금지

## Epoch Aggregate
| epoch | control mean | variant mean | mean Δ | fixes | breaks | exact p | direction splits | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.761052 | 0.752017 | -0.009035 | 1603 | 2145 | 8.59e-19 | 0/3 | REJECT |
| 2 | 0.760919 | 0.739248 | -0.021671 | 2288 | 3588 | 6.24e-65 | 0/3 | REJECT |

## Per-Split Best Fixed Epoch (epoch 1)
| split | control acc | variant acc | Δ | fixes | breaks | p | split gate |
|---|---:|---:|---:|---:|---:|---:|---|
| val_random_uniform_seed42 | 0.762953 | 0.753851 | -0.009102 | 519 | 701 | 2.2e-07 | REJECT |
| val_random_uniform_seed7 | 0.759652 | 0.748350 | -0.011302 | 512 | 738 | 1.97e-10 | REJECT |
| val_random_uniform_seed123 | 0.760552 | 0.753851 | -0.006701 | 572 | 706 | 0.000199 | REJECT |

## Safety / Artifact Check
- validation_only flags: True
- candidate_csv_written flags any: False
- kaggle_submit_executed flags any: False
- 생성된 주요 파일: `artifacts/slk_lite_probe/emb128_L4_r3_seed*/summary.json`, `slk_lite_validation_scores.csv`, 본 aggregate report/json.

## 해석
- SL@K-lite는 Top-K/accuracy 구조를 직접 맞추려는 신규 objective였지만, 동일 pretrain checkpoint에서 BPR continuation control 대비 일관되게 악화했습니다.
- epoch 1이 epoch 2보다 덜 나쁘지만, 세 split 모두 negative delta이며 exact paired 방향도 control 우세입니다.
- 따라서 남은 마지막 제출권을 이 축에 배정할 근거는 없습니다. 추가 grid 확장은 preset gate상 중단합니다.
