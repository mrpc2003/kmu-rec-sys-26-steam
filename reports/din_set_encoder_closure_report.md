# DIN-style Target-Conditioned Set Encoder — Closure Report

**Paradigm:** GPT-5.5 research direction #1 (target-attention, candidate-specific user representation)
**Date:** 2026-05-31 KST
**Verdict:** **REDUNDANT — set-prediction paradigm closed.**
**Submission impact:** none. Validation-only. No Kaggle submission made.

---

## 1. Motivation (why this was the single most promising untested axis)

LightGCN gives each user **one static embedding**. Its score for candidate game *g* is
`<e_u, e_g>` — a fixed user vector dotted with each item. It therefore *cannot* represent
"user looks different depending on which candidate we score": e.g. "having played A **and** B
makes C likely, but A alone does not." This is a genuine higher-order conditional
set-interaction that graph-CF is structurally blind to.

A DIN-style (Zhou et al., KDD 2018) **target-conditioned set encoder** addresses exactly this:
the candidate game is the *query*, the user's interaction history is the *key/value* set, and
target-attention builds a **candidate-specific** user representation. It is a **set** model (no
positional embedding), avoiding SASRec's order-bias failure, and it does **not** consume
LightGCN scores as input, avoiding the GBDT/stacker reconstruction trap.

GPT-5.5 ranked this #1 of its 5 proposed directions; independent analysis agreed it was the
only structurally-new axis among the five.

## 2. Implementation

`scripts/din_set_probe.py` — DIN target-conditioned set encoder.
- **Set, not sequence:** no positional embeddings (SASRec proved order = noise here).
- **Target attention:** candidate embedding = query; user history embeddings = keys/values;
  attention weights → attended history vector.
- **Anti-collapse interaction features:** final scorer takes
  `[q, attended, q⊙attended, mean_pool, max_pool]` (a plain pooling head collapses to MF).
- **Leakage control:** history drawn only from `train_interactions` (held-out excluded,
  identical to LightGCN); uniform-unseen negatives; BPR ranking objective.
- **Training:** leave-one-out — a random train positive is held out as target, the rest form
  the history; mirrors the validation random-mask structure.
- **Gate:** identical solo_acc / corr_z / eq_blend harness, uniform-seed42 split (= public LB
  surrogate).

Variants run on idle V100s (parallel): `d64_L64` (GPU0), `d128_L64` (GPU1), 120 epochs each.

## 3. Results (uniform_seed42 gate; emb128 4-seed ref = 0.76505, noise = 0.0007)

| Variant | solo_acc | corr_z (vs 4-seed) | eq_blend | Δ vs ref | Tier |
|---|---|---|---|---|---|
| DIN d64  | 0.74275 | 0.8478 | 0.76275 | **-0.0023** | REDUNDANT |
| DIN d128 | 0.73955 | 0.8218 | 0.75725 | **-0.0078** | REDUNDANT |

**Independent re-computation** (`scripts/verify_din_independent.py`, does not trust summary.json):
- Integrity clean: NaN=0, inf=0, 19991/19996 unique scores, pred_pos = true_pos (per-user
  exact-half satisfied). No degeneracy (contrast: hyperbolic 200ep = 100% NaN).
- solo_acc reproduced **exactly** (0.74275 / 0.73955).
- The only positive blend Δ (+0.0006 for d64) appears **only** against a weaker single-config
  emb128 (0.76205), is **sub-noise** (< 0.0007), and is **-0.0023 against the true best
  4-seed**. Not a signal under any honest reference.

## 4. Why it failed — the strong-OR-orthogonal dilemma, again

The training trajectory is the decisive evidence:

| Training stage | solo_acc | corr_z (4-seed ref) |
|---|---|---|
| 5-epoch (untrained) | 0.717 | 0.766 |
| 120-epoch d128 | 0.740 | 0.822 |
| 120-epoch d64  | 0.743 | **0.848** |

As the model **gets stronger, correlation to LightGCN rises** (0.766 → 0.85). The early
orthogonality was untrained noise, not signal. This is precisely the risk GPT-5.5 named for
direction #1: *"attention may simply relearn co-occurrence and correlate highly with LightGCN."*
Empirically confirmed. It is the **same collapse mode** as:
- Hyperbolic Lorentz LightGCN (corr 0.75, weaker → GEOMETRY_REDUNDANT)
- GBDT/FM multi-model fusion (corr 0.97, reconstructs LightGCN)

The target-conditioned higher-order set signal, when actually learned to competitive strength
(~0.76), **is** the co-occurrence signal LightGCN already captures. Decorrelation only survives
while the model is too weak to matter. Capacity scaling (d64 → d128) made the blend worse,
ruling out an "under-parameterized" escape.

## 5. GPT-5.5 research — all 5 directions resolved

| # | Direction | Disposition |
|---|---|---|
| **1** | **Target-conditioned set encoder (DIN)** | **REDUNDANT (this report) — strengthens into LightGCN, corr→0.85** |
| 2 | Energy/DPP subset model | Adjacent to exact-K subset loss (Δ=0.00000, closed); collapses to graph-CF when compatibility = co-occurrence |
| 3 | Symmetric review-text dual-tower | MiniLM already REJECT_WEAK (solo 0.639) |
| 4 | Diffusion / denoising recommender | GPT itself flagged collapse to MultiVAE smoothing (MultiVAE already redundant) |
| 5 | Boundary-only uncertainty reranker | Equivalent to boundary covariate expansion (all → chance, 21.4% intrinsic Bayes); also requires set-encoder scores as input (sub-#1) |

The **set-prediction paradigm is now fully closed.**

## 6. Cumulative paradigm closure (validation-first, uniform gate)

| Paradigm | Best representative | Status |
|---|---|---|
| Graph-BPR CF | LightGCN emb128 4-seed | **#1, public 0.77745** |
| Graph SSL/contrastive | SGL, DirectAU, xSimGCL | weaker / monotonic worsening |
| Hard-negative | DNS pool1/8/16 | monotonic collapse |
| Item-item linear | itemKNN, EASE, ALS, Turbo-CF | corr ~0.97, weaker |
| Latent generative | MultiVAE | redundant w/ EASE |
| Text / LM semantic | MiniLM, AlphaRec | REJECT_WEAK |
| Capacity frontier | emb64–320 | plateau |
| Seed / cross-cap ensemble | 4/8-seed, emb128⊕192 | tied, all ≤ 4-seed |
| Stage-2 stacking | logreg, GBDT/FM | FAILED / corr 0.97 |
| Geometry | hyperbolic Lorentz | REDUNDANT (corr 0.75) |
| Sequence | SASRec (L20/50/100, d128) | REJECT_FLOOR (longer = worse) |
| **Set-prediction** | **DIN target-attention, exact-K, DPP, boundary** | **REDUNDANT / closed (this report)** |

## 7. Conclusion

Every structurally distinct paradigm — collaborative-filtering, item-item linear, latent,
text/LM, geometry, sequence, tree-stacking, and now target-conditioned set encoders — has been
validation-first exhausted. No method has been found that is **both** competitively strong
(uniform solo ~0.76) **and** genuinely decorrelated from LightGCN (within-user corr < 0.9);
every decorrelated candidate was decorrelated only because weaker or solving a different
problem, and converges to LightGCN as it is trained to strength. Combined with the three
independent ceiling proofs (21.4% intrinsic Bayes simultaneous-error floor, scorer redundancy
corr ~0.97, orthogonal-models-solve-a-different-problem), the signal space is **saturated**.

**emb128 4-seed (uniform 0.76505 / public 0.77745) is the confirmed peak.** Final-2 reproducible
bundle (emb128 SHA 7e3191de, emb64 SHA dcc578de) remains locked and byte-identical. The honest,
disciplined action is to **stop here.**

---

*Scripts: `scripts/din_set_probe.py`, `scripts/verify_din_independent.py`.
Artifacts: `artifacts/din_set/`, `artifacts/din_set_variants/`.
GPT-5.5 research log: `logs/gpt55_research.log`.*
