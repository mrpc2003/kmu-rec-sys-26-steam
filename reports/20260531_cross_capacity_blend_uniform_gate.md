# Cross-capacity blend emb128⊕emb192 — UNIFORM gate + paired McNemar

- split `val_random_uniform_seed42` rows=19996
- emb128 ens 0.76505 | emb192 ens 0.76615 | **corr_z(128,192) = 0.9864**
- 50/50 z-blend uniform: **0.76545** (raw-8-mean 0.76555) vs emb128 ref 0.76505 → **+0.00040**
- paired McNemar: blend right/emb128 wrong = 157, emb128 right/blend wrong = 149, p = **0.6891**
- **tier: NO_GAIN** — blend_z 0.76545 vs ref 0.76505 = +0.00040 (<= noise). corr_z(128,192)=0.986; cross-capacity blend gives no uniform gain. Keep emb128 4-seed (public 0.77745).

## Why this axis

Post-mortem: emb128 (0.77745) and emb192 (0.77715) tie on real LB but disagree on 3.40% of rows -> two strong, partly-decorrelated models. A cross-capacity blend is the one diversity play not yet tested (vs weak side-axes and same-config seed averaging). Gated with paired McNemar because the emb192 submission proved surrogate Δ < 0.003 is unreliable.
