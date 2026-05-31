# DNS (Dynamic Negative Sampling) pool-sweep — UNIFORM gate

- backbone emb128 L4 reg1e-3 | split `val_random_uniform_seed42`
- emb128 4-seed ensemble ref (public 0.77745): **0.76505** | single-seed plain-BPR ~0.762 | noise ±0.0007
- control (pool=1, plain random neg): **0.76565**
- **best: pool=1 = 0.76565  (vs ensemble +0.00060)**
- **tier: TIED_ENSEMBLE_LEVEL** — DNS pool=1 single model 0.76565 ~ emb128 4-seed ensemble 0.76505 (Δ+0.00060); a SINGLE DNS model matching a 4-seed ensemble is promising -> a DNS seed ensemble may exceed it; consider escalation.

| DNS pool M | uniform acc | train s |
|---|---:|---:|
| 1 (control) | 0.76565 | 1044.6 |
| 8 | 0.67734 | 1051.7 |
| 16 | 0.65243 | 1029.8 |
| 32 | 0.62633 | 1054.3 |
