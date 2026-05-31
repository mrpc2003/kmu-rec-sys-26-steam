# emb192 L4 reg1e-3 — 4-seed Ensemble UNIFORM gate (public surrogate)

- split `val_random_uniform_seed42` (rows=19996) | seeds [42, 123, 2024, 7]
- emb128 4-seed ensemble ref (public 0.77745): **0.76505** | emb128 single 0.76205 | noise ±0.0007
- **emb192 4-seed ensemble: 0.76615**  (vs emb128 ens +0.00110)
- **tier: UPGRADE_MATERIALIZE** — emb192 4-seed ensemble 0.76615 beats emb128 ensemble 0.76505 by +0.00110 > noise 0.0007 -> GENUINE backbone upgrade. Materialize the emb192 4-seed full-test candidate and gate it to 우현 for submission approval.

| seed | emb192 uniform acc |
|---|---:|
| 42 | 0.76665 |
| 123 | 0.76425 |
| 2024 | 0.76365 |
| 7 | 0.76325 |
| **4-seed ens** | **0.76615** |
