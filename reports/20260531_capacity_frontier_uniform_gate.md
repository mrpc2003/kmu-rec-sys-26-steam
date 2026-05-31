# LightGCN Capacity Frontier — UNIFORM gate (public surrogate)

- split `val_random_uniform_seed42` | emb128 single-seed **0.76205** | emb128 4-seed ens **0.76505** (public 0.77745) | noise ±0.0007
- **best: emb192_L4_r3 = 0.76665  (vs emb128 single +0.00460, vs ens +0.00160)**
- **tier: ESCALATE_SEED_ENSEMBLE** — best emb192_L4_r3 single 0.76665 beats emb128 single-seed 0.76205 by +0.00460 > noise 0.0007 -> capacity frontier still rising. Worth a 4-seed ensemble of emb192_L4_r3; its ensemble would likely exceed the emb128 ensemble 0.76505. Then gate the candidate to 우현.

| config | uniform acc | vs emb128 single | train s |
|---|---:|---:|---:|
| emb192_L4_r3 | 0.76665 | +0.00460 | 1823.7 |
| emb256_L4_r3 | 0.76215 | +0.00010 | 1917.5 |
| emb320_L4_r3 | 0.76145 | -0.00060 | 1899.5 |
| emb256_L4_r2 | 0.73625 | -0.02580 | 1844.2 |

## Why this gate was needed

The original hparam sweep DID try emb256 but evaluated it ONLY on the hard samplers (sqrtpop/recent/popbin). We later proved the public LB tracks the UNIFORM split, so those emb256 numbers were measured on the wrong distribution. This probe gates capacity on the actual public surrogate for the first time.
