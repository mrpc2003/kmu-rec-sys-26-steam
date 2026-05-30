# LightGCN + Stage2 Stacking Meta-Learner (OOF validation)

Honest evaluation uses **user-level GroupKFold** (group); row-level StratifiedKFold (strat) is shown only as a leakage diagnostic — within-user z/rank features can leak across rows of the same user.

- mean LightGCN: **0.63883**
- mean fixed-blend: 0.64106
- mean stack-logreg [group/honest]: 0.6493
- mean stack-lightgbm [group/honest]: 0.64703
- mean stack-logreg [strat/diag]: 0.64933
- mean stack-lightgbm [strat/diag]: 0.64729
- logreg strat−group leakage gap: 3e-05
- **best honest method: mean_stack_logreg_group = 0.6493**

| split | LightGCN | fixed | lr-strat | lr-group | lgbm-strat | lgbm-group |
|---|---:|---:|---:|---:|---:|---:|
| random_sqrtpop | 0.67483 | 0.67704 | 0.67944 | 0.67914 | 0.66963 | 0.67013 |
| recent_sqrtpop | 0.63963 | 0.64053 | 0.64973 | 0.64923 | 0.65493 | 0.65423 |
| random_popbin | 0.60202 | 0.60562 | 0.61882 | 0.61952 | 0.61732 | 0.61672 |

## Interpretation

- Honest (GroupKFold) stacking beats LightGCN by +0.01047 → a full-data stacker is worth materializing as a submission candidate (pending approval).