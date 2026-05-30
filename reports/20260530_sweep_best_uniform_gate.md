# Sweep-best configs — UNIFORM gate (public surrogate)

- anchor uniform: **0.75445**  |  seed ensemble uniform: **0.76145**
- all configs evaluated
- **verdict: emb128_L4_reg1e-03 BEATS the seed ensemble on uniform (0.76205 > 0.76145) — new strongest candidate.**

| config | sweep hard Δmean | uniform acc | vs anchor | vs ensemble |
|---|---:|---:|---:|---:|
| emb128_L4_reg1e-03 | +0.00326 | 0.76205 | +0.00760 | +0.00060 |
| emb128_L3_reg1e-03 | +0.00223 | 0.75815 | +0.00370 | -0.00330 |
| emb128_L4_reg1e-04 | +0.00310 | 0.75775 | +0.00330 | -0.00370 |

## Interpretation
A config that gained on the hard-sampler mean but does not beat the anchor on uniform is an artifact of the wrong surrogate (see mechanism test: reg/arch changes that suppress popularity help on hard samplers, hurt on uniform). Only a config that beats the seed ensemble on uniform would displace the current candidate.
