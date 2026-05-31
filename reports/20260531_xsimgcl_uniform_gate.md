# XSimGCL λ-sweep — UNIFORM gate (public surrogate)

- backbone: emb128 L4 reg1e-3 + XSimGCL CL term | split `val_random_uniform_seed42`
- emb128 4-seed ref (public 0.77745): **0.76505** | pop floor 0.684 | noise ±0.0007
- **best: lam0.02_eps0.1 = 0.74145  (vs ref -0.02360)**
- **tier: TIED_OR_WEAK** — best XSimGCL (lam0.02_eps0.1=0.74145) >= floor but vs emb128 ref = -0.02360, within/under noise 0.0007 -> not a parameter-free upgrade; contrastive term gives no clear gain over the strong backbone.

| config | uniform acc | train s |
|---|---:|---:|
| lam0.02_eps0.1 | 0.74145 | 1936.5 |
| lam0.05_eps0.1 | 0.70614 | 1931.1 |
| lam0.1_eps0.1 | 0.66743 | 1932.5 |
| lam0.2_eps0.2 | 0.62382 | 1943.2 |
