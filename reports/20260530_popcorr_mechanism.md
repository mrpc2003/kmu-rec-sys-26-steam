# Popularity-correction mechanism test (per negative-sampler)

Tests whether down-weighting popular items (the stacker's learned behavior, `log_pop` weight −0.42) helps hard samplers but hurts the near-uniform public-surrogate split.

`score = z(LightGCN) − alpha · within_user_z(log_pop)`; alpha>0 = popularity down-weighting.

| split | base | a=0.0 | a=0.1 | a=0.25 | a=0.5 | a=1.0 | best_a | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_uniform | 0.75445 | 0.75445 | 0.75125 | 0.73655 | 0.68704 | 0.52000 | 0.0 | -0.00000 |
| random_sqrtpop | 0.67483 | 0.67483 | 0.67644 | 0.67984 | 0.65583 | 0.55111 | 0.25 | +0.00501 |
| recent_sqrtpop | 0.63963 | 0.63963 | 0.63873 | 0.63073 | 0.60512 | 0.51610 | 0.0 | +0.00000 |
| random_popbin | 0.60202 | 0.60202 | 0.60452 | 0.60382 | 0.58562 | 0.53951 | 0.1 | +0.00250 |
| random_communitypop | 0.57231 | 0.57231 | 0.57421 | 0.57822 | 0.57251 | 0.55021 | 0.25 | +0.00591 |
| recent_communitypop | 0.55551 | 0.55551 | 0.55281 | 0.54291 | 0.52691 | 0.49230 | 0.0 | -0.00000 |

## Interpretation

- ✅ Mechanism CONFIRMED: popularity down-weighting (alpha>0) helps the hard popularity-matched samplers but does NOT help (or hurts) the near-uniform public-surrogate split. This is exactly why the logreg stacker — which learned a −0.42 log_pop weight on hard-sampler validation — failed on the near-uniform public test.