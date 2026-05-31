# Hyperbolic (Lorentz) Geometry — Track Closure

**Status: CLOSED — GEOMETRY_REDUNDANT (independently verified)**

## Hypothesis (A.2)
The one structurally-untested combination: keep the strong ranking (BPR/triplet) loss and the
exact uniform-unseen negative sampler, but swap ONLY the decision geometry from Euclidean
inner-product to negative Lorentz geodesic distance (geoopt RiemannianAdam, HGCF pattern).

## Why early-stop (epoch 130)
The full 200-epoch probe hit a **NaN embedding collapse at epoch ~150-160** — the 3rd recurrence
at the same window. The root-cause fix (global grad-norm clip + non-finite-grad batch skip)
prevented the crash but **masked the collapse as loss=0.000000**, producing 100% NaN scores
(`solo_acc=0.4986` was an arbitrary tie-break on NaN, not a real ranking). The Lorentz distance
gradient `1/sqrt(arg²-1)` explodes for rare near-coincident pairs even with the distance value
clamped. `es130` captures the **last healthy point before collapse** for a clean read.

## Independent verification (not trusting summary.json)
Recomputed every metric directly from the score CSVs:

| metric | recomputed | summary claim | match |
|---|---:|---:|---|
| hyp score CSV validity | 0 NaN, 19988 unique finite | — | valid, not degenerate |
| solo_acc | **0.71734** | 0.71734 | exact |
| corr_z(emb128, hyp) | **0.7467** | 0.7467 | exact |
| eq_blend (50/50 z) | 0.74805 | 0.75055 | ~0.0025 diff, same direction |

## Result

| model | uniform acc |
|---|---:|
| popularity floor | 0.684 |
| **hyperbolic emb64 (es130)** | **0.71734** |
| emb128 4-seed ensemble | 0.76505 |
| 50/50 z-blend (emb128 ⊕ hyp) | 0.74805 |

## Verdict
Hyperbolic geometry **IS** a real ranking (0.717 > floor 0.684) and **IS** genuinely decorrelated
from the Euclidean base (corr_z 0.747 < 0.9 — a truly different model). **But** it is **weaker**
than emb128 (0.717 vs 0.765), and the parameter-free 50/50 blend **drags emb128 down by −0.017**.
Diversity does not compensate for the weaker base → **no orthogonal value**.

## Implication — structural exploration exhausted
Every axis is now closed with evidence:
- SOTA families: graph, item-CF, LM, VAE, capacity, hard-negative, ensemble
- GPT-5.5-Pro levers: exact-K subset loss, temporal compatibility, candidate-marginal residual, hours-confidence
- boundary co-error: intrinsic Bayes ceiling (21.4% simultaneous, non-recoverable)
- 8-seed expansion: TIED (Δ −0.0004)
- cross-capacity blend: TIED / NO_GAIN (public-confirmed −0.0003 vs emb128)
- **hyperbolic geometry: redundant (this report)**

**emb128 4-seed ensemble (public 0.77745) remains the final #1.** No submission from this track.
