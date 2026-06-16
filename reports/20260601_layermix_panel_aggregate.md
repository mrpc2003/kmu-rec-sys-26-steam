# 2026-06-01 LightGCN++ Layer-Mixture Probe Panel Aggregate

## 결론

- 판정: **REJECT**
- 범위: validation-only 3-split panel. candidate/test CSV 생성 및 Kaggle submit 없음.
- 기준: fixed variant 기준 평균 Δ ≥ 0.00355, 2/3 split 이상 양수, pooled paired McNemar p<0.05, fixes>breaks.
- 최상 fixed non-base: `score_blend_shallow_heavy_l0p25` mean Δ=+0.000200, positive_splits=2/3, fixes/breaks=154/142, pooled p=0.5227.
- 해석: seed42의 late-heavy blend는 약한 양수였지만 seed7/seed123에서 역전/소멸했고, fixed-grid 평균 Δ가 MDE의 6% 수준에 불과하다. 레이어 가중치 자체가 새로운 public-transferable 축이라는 증거 없음.

## 상위 fixed variant 집계

| rank | score | mean Δ | splits + | fixes | breaks | net | pooled p | fisher p | per-split Δ(seed42/7/123) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `score_blend_shallow_heavy_l0p25` | +0.000200 | 2/3 | 154 | 142 | +12 | 0.5227 | 0.5814 | +0.000600 / -0.000300 / +0.000300 |
| 2 | `score_blend_tail_cut_l0p25` | +0.000067 | 2/3 | 119 | 115 | +4 | 0.8446 | 0.06146 | +0.000800 / -0.000900 / +0.000300 |
| 3 | `score_layermix_shallow_heavy` | -0.000100 | 2/3 | 801 | 807 | -6 | 0.9008 | 0.5536 | +0.000800 / +0.000400 / -0.001500 |
| 4 | `score_blend_mid_heavy_l0p25` | -0.000200 | 1/3 | 241 | 253 | -12 | 0.6207 | 0.5595 | +0.000500 / -0.000300 / -0.000800 |
| 5 | `score_blend_late_heavy_l0p25` | -0.000467 | 1/3 | 603 | 631 | -28 | 0.4421 | 0.03112 | +0.001700 / -0.000500 / -0.002601 |
| 6 | `score_blend_no_deep_l0p25` | -0.000667 | 0/3 | 515 | 555 | -40 | 0.2331 | 0.6634 | -0.001000 / -0.000600 / -0.000400 |
| 7 | `score_layermix_mid_heavy` | -0.000934 | 1/3 | 1256 | 1312 | -56 | 0.2778 | 0.09561 | +0.001500 / -0.003201 / -0.001100 |
| 8 | `score_blend_no_ego_l0p25` | -0.001067 | 1/3 | 564 | 628 | -64 | 0.06799 | 0.06986 | +0.000500 / -0.001400 / -0.002300 |
| 9 | `score_layermix_tail_cut` | -0.001400 | 0/3 | 544 | 628 | -84 | 0.0153 | 0.0725 | -0.000500 / -0.001900 / -0.001800 |
| 10 | `score_blend_ego_heavy_l0p25` | -0.001567 | 0/3 | 671 | 765 | -94 | 0.01409 | 0.0933 | -0.001500 / -0.001800 / -0.001400 |
| 11 | `score_blend_decay_half_l0p25` | -0.003201 | 0/3 | 911 | 1103 | -192 | 2.053e-05 | 0.0001677 | -0.002100 / -0.004301 / -0.003201 |
| 12 | `score_layermix_no_ego` | -0.009469 | 0/3 | 2732 | 3300 | -568 | 2.756e-13 | 3.256e-12 | -0.006801 / -0.009802 / -0.011802 |

## Best-per-split oracle diagnostic — 제출 후보 아님

split별로 다른 variant를 고르는 것은 사후 cherry-pick이므로 금지한다. 아래는 headroom 진단용이다.

| split | best score | Δ | fixes | breaks | McNemar p |
|---|---|---:|---:|---:|---:|
| seed7 | `score_layermix_shallow_heavy` | +0.000400 | 272 | 264 | 0.7624 |
| seed42 | `score_blend_late_heavy_l0p25` | +0.001700 | 226 | 192 | 0.1065 |
| seed123 | `score_blend_shallow_heavy_l0p25` | +0.000300 | 60 | 54 | 0.6396 |

Oracle mean Δ=+0.000800, fixes/breaks=558/510, pooled p=0.1503. 이조차 MDE에 크게 미달한다.

## 다음 단계

- Layer-mixture는 REJECT로 닫고, 기존 자문서의 백업축인 **SL@K-lite / Top-K metric-aligned continuation probe**로 넘어간다.
- continuation probe도 동일하게 validation-only, same checkpoint, old-loss continuation control vs hybrid top-K objective variant 비교로 제한한다.
