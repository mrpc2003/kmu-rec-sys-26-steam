# emb128_L4_reg1e-3 ensemble vs emb64 ensemble — UNIFORM (public surrogate)

- emb64 ensemble (submitted, public 0.77125): **0.76145**
- emb128 single-seed42 (sweep gate): 0.76205
- anchor: 0.75445  |  single-seed noise band: ±0.0007
- **emb128 4-seed ensemble: 0.76505**
- **vs emb64 ensemble: +0.00360** → **UPGRADE**
- verdict: emb128_L4_reg1e-3 ensemble (0.76505) beats emb64 ensemble (0.76145) by +0.00360 > noise 0.0007 -> genuine upgrade, consider as new primary.

| seed | emb128 uniform acc |
|---|---:|
| 42 | 0.76205 |
| 123 | 0.76275 |
| 2024 | 0.76435 |
| 7 | 0.76325 |
| **ensemble** | **0.76505** |

## Interpretation
The sweep gate's single-seed 0.76205 vs the emb64 ensemble 0.76145 was an unfair comparison (single seed vs 4-seed mean). This evaluates emb128 as its own 4-seed ensemble so both sides have equal variance reduction. A within-noise result means the two configs are interchangeable on the surrogate; the emb64 ensemble stays primary because it is already validated on the real public LB (0.77125).
