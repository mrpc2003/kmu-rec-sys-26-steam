# Multi-Sampler Cross-Fitted Residual Invariance Audit — Final Closure

**Date:** 2026-05-31 KST
**Origin:** GPT-5.5 Round-2's single recommended experiment ("identify whether any residual
feature is real under uniform-gate validation or just sampler leakage").
**Verdict:** **NO train-only symmetric prior survives. Signal space exhausted. Stop confirmed.**
**Submission impact:** none. Validation-only, CPU. (`scripts/multisampler_residual_invariance_audit.py`)

---

## 1. Why this is the strongest possible test

Earlier we rejected hours / temporal / text / popularity features **individually on the uniform
split**. A skeptic could argue: maybe each is weak alone but a *combination* helps, or maybe we
were just unlucky on one split. This audit closes both loopholes:

- **Four negative samplers** (uniform / sqrtpop / popbin / communitypop) that share **exactly the
  same held-out positives** (verified 9998/9998 identical) — only the negatives differ. The same
  LightGCN embedding scores all four (inner product is sampler-invariant). A feature reflecting
  *true preference* must help regardless of negative composition; a feature reflecting the
  *sampler* (popularity) helps on uniform but flips on hard samplers.
- **Sign-invariance gate:** a feature PASSES only if uniform Δ ≥ +0.0007 **AND** Δ > 0 on all four
  samplers at the same λ. This structurally kills the pop-trap that defeated the candidate-marginal
  quota and the logreg/GBDT stackers.
- **Integrated cross-fitted model:** all 11 features in one ridge probe, weights fit on half the
  uniform users, evaluated on the **held-out** other half — the honest out-of-sample number.

## 2. Single-feature result (11 train-only symmetric priors)

Base LightGCN solo: uniform 0.76205 / sqrtpop 0.67483 / popbin 0.60202 / communitypop 0.57231.

| feature | best λ (uniform) | Δ uniform | Δ sqrtpop | Δ popbin | Δ comm | result |
|---|---|---|---|---|---|---|
| it_logcount | +0.30 | **+0.00220** | -0.00450 | -0.00490 | -0.00450 | uniform+ but FLIPS (pop-trap) |
| it_date_mean | -0.05 | +0.00180 | +0.00030 | -0.00070 | -0.00180 | flips |
| it_textlen_mean | -0.10 | +0.00170 | -0.00170 | -0.00100 | 0.00000 | flips |
| inter_date_absdiff | +0.05 | +0.00160 | -0.00010 | -0.00090 | -0.00080 | flips |
| it_funny_rate | -0.10 | +0.00150 | +0.00020 | 0.00000 | -0.00220 | flips |
| it_hours_std | -0.05 | +0.00140 | +0.00110 | 0.00000 | -0.00030 | flips |
| it_hours_med | +0.02 | +0.00020 | +0.00060 | +0.00080 | +0.00040 | sign-stable but **sub-noise** |
| inter_hours_prod | +0.05 | +0.00040 | +0.00060 | +0.00090 | -0.00010 | sub-noise + flips |
| (others) | — | < +0.0007 | — | — | — | no uniform gain |

**Zero features pass.** The features with the *largest* uniform gains are precisely the ones that
flip sign on hard samplers (popularity, date, text-length) — they encode the sampler, not
preference. The only sign-stable features are sub-noise.

## 3. Integrated cross-fitted model — the trap demonstrated live

A ridge probe over `[base_z, 11 features]`, fit on half the uniform users:

- Largest learned weight: **it_logcount = +0.0567** — the model spends its capacity on popularity.
- In-sample full uniform: +0.00040 (the seductive false positive)
- **Cross-fitted held-out B: −0.00119** (base_B 0.76364 → regresses)
- sqrtpop −0.00140, popbin −0.00180, communitypop −0.00210 (regresses everywhere)

This reproduces, in miniature and on purpose, the exact failure of the earlier logreg/GBDT
stackers (public 0.77745 → 0.75355 / 0.76245). Any model given these features learns popularity
down-weighting, which is an in-sample uniform mirage that does not survive cross-fitting and
inverts on the real (harder-than-uniform-in-places) negative structure.

## 4. Conclusion — three independent paths converge

| Path | Method | Conclusion |
|---|---|---|
| Modeling | 12 paradigms (CF/linear/latent/text/geometry/sequence/set/stacking) | all collapse to LightGCN when strong |
| Ceiling probe | 9-family + 7-seed agreement, covariate separation | ~19% correlated error, non-addressable, no covariate flags hard rows |
| GPT-5.5 ×2 rounds | independent RecSys/PU/noise reasoning | "space is probably exhausted," irreducible-ceiling prior 70-85% |
| **This audit** | **multi-sampler sign-invariance + cross-fit** | **no symmetric prior survives; trap demonstrated live** |

All four converge on the same answer. There is no train-only signal — modeling or feature-based —
that is **both** real on the public-LB-surrogate uniform split **and** robust to the negative
sampler. Every apparent gain is either sub-noise or a popularity artifact.

**emb128 4-seed (uniform 0.76505 / public 0.77745) is the saturated peak.**
Final-2 reproducible bundle (emb128 SHA 7e3191de, emb64 SHA dcc578de) remains locked and
byte-identical. The honest, disciplined action is to **stop here.**

---

*Script: `scripts/multisampler_residual_invariance_audit.py`. No Kaggle submission.*
