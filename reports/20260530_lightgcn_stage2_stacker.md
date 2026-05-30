# LightGCN + Stage2 Stacking Meta-Learner (OOF validation)

- mean LightGCN: **0.63883**
- mean fixed-blend: 0.64106
- mean stack-logreg (OOF): 0.64933
- mean stack-lightgbm (OOF): 0.64729
- **best method: mean_stack_logreg = 0.64933**

| split | LightGCN | fixed-blend | stack-logreg | stack-lgbm |
|---|---:|---:|---:|---:|
| random_sqrtpop | 0.67483 | 0.67704 | 0.67944 | 0.66963 |
| recent_sqrtpop | 0.63963 | 0.64053 | 0.64973 | 0.65493 |
| random_popbin | 0.60202 | 0.60562 | 0.61882 | 0.61732 |

## Interpretation

- Stacking beats LightGCN by +0.01050 on mean OOF → a full-data stacker is worth materializing as a submission candidate (pending approval).