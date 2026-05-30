# LightGCN ⟂ Stage2 Complementarity (validation)

- mean LightGCN: **0.63883**
- mean Stage2: 0.62563
- mean best-blend: 0.64106 (beats LightGCN)
- mean oracle upper bound: 0.72893

## Per-split crosstab (LightGCN vs Stage2)

| split | LG | S2 | both✓ | both✗ | LG-only✓ | S2-only✓ | oracle | best-blend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_sqrtpop | 0.6748 | 0.6597 | 11643 | 4953 | 1851 | 1549 | 0.7523 | 0.6770 (w0.8/global) |
| recent_sqrtpop | 0.6396 | 0.6261 | 10878 | 5564 | 1912 | 1642 | 0.7217 | 0.6405 (w0.7/global) |
| random_popbin | 0.6020 | 0.5910 | 9604 | 5744 | 2434 | 2214 | 0.7127 | 0.6056 (w0.8/global) |

## Interpretation

- Stage2-only-right rows (LightGCN missed, Stage2 caught): 5405
- LightGCN-only-right rows (Stage2 missed, LightGCN caught): 6197
- A blend improves mean validation by +0.00224 → worth materializing a blended candidate.