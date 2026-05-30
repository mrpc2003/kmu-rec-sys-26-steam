# Seed Ensemble — UNIFORM Gate (real public surrogate)

- split: `val_random_uniform_seed42`  (rows=19996)
- seeds present: [42, 123, 2024, 7]
- seed42 uniform ref: **0.75445**
- ensemble uniform: **0.76145**
- **Δ vs seed42 on uniform: +0.00700**
- **verdict: ROBUST_GAIN** — Ensemble beats seed42 on the real public surrogate -> strong submit candidate.

| seed | uniform row acc |
|---|---:|
| 42 | 0.75445 |
| 123 | 0.75735 |
| 2024 | 0.75775 |
| 7 | 0.75805 |
| **ensemble** | **0.76145** |

## Why this gate
The +0.0041 mean ensemble gain was measured on hard samplers only. The OOD gate proved public (0.76245) tracks the uniform split, so the gain must be confirmed HERE before the ensemble can be treated as a stronger candidate than the submitted seed42 LightGCN (public 0.76245).
