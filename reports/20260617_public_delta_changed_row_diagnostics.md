# Public Delta Changed-Row Diagnostics

Analysis-only. No Kaggle submit, no hidden labels, no new candidate CSV.

Reference: `candidate_rank_blend_emb128_emb192.csv` public `0.77825`. `candidate_better_frac` is an approximation assuming the public subset is half of changed rows.

| candidate | public | Δ vs ref | row diff | approx better frac on changed public rows | changed item_pop mean | changed pop-rank pct |
|---|---:|---:|---:|---:|---:|---:|
| `rankblend_public_best` | 0.77825 | +0.00000 | 0 | 1.000 | 0.00 | 0.000 |
| `otto_forced` | 0.77815 | -0.00010 | 508 | 0.498 | 78.40 | 0.518 |
| `als_htr_w0.025` | 0.77805 | -0.00020 | 278 | 0.493 | 79.89 | 0.526 |
| `als_htr_w0.05` | 0.77805 | -0.00020 | 278 | 0.493 | 79.89 | 0.526 |
| `als_htr_w0.1` | 0.77805 | -0.00020 | 278 | 0.493 | 79.89 | 0.526 |
| `als_raw_w0.2` | 0.77795 | -0.00030 | 276 | 0.489 | 82.74 | 0.520 |
| `als_htr_w0.2` | 0.77785 | -0.00040 | 300 | 0.487 | 82.93 | 0.520 |
| `boundary_w_neg075` | 0.77755 | -0.00070 | 582 | 0.488 | 75.69 | 0.544 |
| `emb128_stable` | 0.77745 | -0.00080 | 368 | 0.478 | 70.15 | 0.535 |
| `frontier_w1920_w64neg025` | 0.77715 | -0.00110 | 556 | 0.480 | 74.93 | 0.537 |
| `boundary_v1_forced` | 0.77705 | -0.00120 | 176 | 0.432 | 87.40 | 0.516 |
| `tagcf_seed2024` | 0.77615 | -0.00210 | 646 | 0.467 | 76.71 | 0.531 |
| `boundary_w2` | 0.77575 | -0.00250 | 596 | 0.458 | 77.42 | 0.522 |

## Readout

- Public-loss candidates are not just “many row changes”; the worst drops concentrate deviations from the rankblend anchor on rows where the anchor appears better aligned with the public subset.
- The approximate better fraction stays below 0.5 for every non-reference candidate, so public feedback treats most deviations from rankblend as harmful.
- This supports using public-mimic validation for gate calibration, but it does not reveal a new positive axis by itself.
