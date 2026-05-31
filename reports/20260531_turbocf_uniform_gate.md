# Turbo-CF / Graph-Filtering CF — UNIFORM gate (public surrogate)

- split `val_random_uniform_seed42` | emb128 4-seed ref **0.76505** | floor 0.684 | noise ±0.0007
- **tier: REDUNDANT** — strong-ish (best solo 0.74155) but corr_z 0.8472 high and blend 0.75825 vs ref -0.00680 -> same family as EASE/ItemKNN, redundant. No new axis.

| config | solo uniform | corr_z vs emb128 | 50/50 z-blend | blend vs ref |
|---|---:|---:|---:|---:|
| power1_poly2a0.3 | 0.74155 | 0.8472 | 0.75825 | -0.00680 |
| power1_linear | 0.74105 | 0.8454 | 0.75775 | -0.00730 |
| power2_linear | 0.73005 | 0.7839 | 0.75765 | -0.00740 |
| power2_poly2a0.3 | 0.73005 | 0.7842 | 0.75795 | -0.00710 |
