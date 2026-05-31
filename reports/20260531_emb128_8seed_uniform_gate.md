# emb128_L4_reg1e-3 — 4->8 Seed Expansion (UNIFORM gate)

- split: `val_random_uniform_seed42` (rows=19996)
- seeds present: [42, 123, 2024, 7, 11, 99, 2025, 314]
- 4-seed ref (public 0.77745): **0.76505**
- 4-seed (recomputed here): 0.76505
- **8-seed uniform: 0.76465**
- **Δ vs 4-seed ref: -0.00040**
- **tier: TIED** — 8-seed (0.76465) vs 4-seed (0.76505) = -0.00040, within noise 0.0007 -> statistically tied; 4-seed stays primary. 8-seed only as a marginally lower-variance alternate, not a clear upgrade.

| seed | uniform acc |
|---|---:|
| 42 | 0.76205 |
| 123 | 0.76275 |
| 2024 | 0.76435 |
| 7 | 0.76325 |
| 11 | 0.76475 |
| 99 | 0.76405 |
| 2025 | 0.76345 |
| 314 | 0.76515 |
| **4-seed ens** | **0.76505** |
| **8-seed ens** | **0.76465** |
