# Aggressive rank/z frontier scan — validation only

## Verdict context
- split: `val_random_uniform_seed42`
- base emb128 acc: `0.765053`
- variants scanned: `140`
- seed42-only strict gate pass count: `0`
- safety: no hidden/test read, no candidate CSV, no Kaggle submit.

## Top variants
| rank | variant | family | Δ vs emb128 | fixes | breaks | p | changed |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `z_w128_1_w192_2_w64_0.5` | weighted_z_sum | +0.001300 | 247 | 221 | 0.2478 | 468 |
| 2 | `rrf_128_192_k5` | rrf | +0.001300 | 191 | 165 | 0.1851 | 356 |
| 3 | `rrf_128_192_k10` | rrf | +0.001300 | 191 | 165 | 0.1851 | 356 |
| 4 | `rrf_128_192_k20` | rrf | +0.001300 | 191 | 165 | 0.1851 | 356 |
| 5 | `rrf_128_192_k50` | rrf | +0.001300 | 191 | 165 | 0.1851 | 356 |
| 6 | `boundary_rank_128_192_frac0.2_cap5` | boundary_only | +0.001300 | 185 | 159 | 0.1776 | 344 |
| 7 | `boundary_rank_128_192_frac0.2_cap10` | boundary_only | +0.001300 | 185 | 159 | 0.1776 | 344 |
| 8 | `boundary_rank_128_192_frac0.2_cap20` | boundary_only | +0.001300 | 185 | 159 | 0.1776 | 344 |
| 9 | `boundary_rank_128_192_frac0.05_cap5` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 10 | `boundary_rank_128_192_frac0.05_cap10` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 11 | `boundary_rank_128_192_frac0.05_cap20` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 12 | `boundary_rank_128_192_frac0.1_cap5` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 13 | `boundary_rank_128_192_frac0.1_cap10` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 14 | `boundary_rank_128_192_frac0.1_cap20` | boundary_only | +0.001300 | 182 | 156 | 0.1738 | 338 |
| 15 | `rank_w128_1_w192_3_w64_0.25` | weighted_rank_sum | +0.001200 | 339 | 315 | 0.3685 | 654 |
| 16 | `z_w128_1_w192_3_w64_0.5` | weighted_z_sum | +0.001200 | 273 | 249 | 0.3141 | 522 |
| 17 | `z_w128_1_w192_1_w64_0.25` | weighted_z_sum | +0.001200 | 177 | 153 | 0.2054 | 330 |
| 18 | `rank_w128_1_w192_3_w64_0` | weighted_rank_sum | +0.001100 | 340 | 318 | 0.413 | 658 |
| 19 | `rank_w128_1_w192_3_w64_0.5` | weighted_rank_sum | +0.001100 | 337 | 315 | 0.4109 | 652 |
| 20 | `rank_w128_1_w192_3_w64_1` | weighted_rank_sum | +0.001100 | 311 | 289 | 0.3913 | 600 |

## Interpretation
공격 후보는 양수지만 MDE 미만이다. rank-blend public 개선처럼 manual-risk 후보는 될 수 있으나, 통계적으로는 noise-chasing 범주다.
