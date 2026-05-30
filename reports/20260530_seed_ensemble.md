# LightGCN Seed Ensemble (raw-score averaging)

- seeds: [42, 123, 2024, 7]
- mean single (seed42): 0.63883
- mean ensemble: 0.64293
- **mean gain: +0.00410**

| split | seed42 | ensemble | Δ |
|---|---:|---:|---:|
| random_sqrtpop | 0.67483 | 0.68274 | +0.00791 |
| recent_sqrtpop | 0.63963 | 0.64073 | +0.00110 |
| random_popbin | 0.60202 | 0.60532 | +0.00330 |

## Note

Raw-score averaging of the same verified config; no validation-label learning, so it cannot overfit the negative sampler the way the logreg stacker did (which failed public 0.76245→0.75355). Expect small/flat gain, low downside.