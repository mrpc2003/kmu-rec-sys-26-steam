# New-Paradigm Exploration — Capstone Synthesis (2026-06-01)

User request: "완전 새로운 방법론을 탐색해" (explore a completely new methodology).
Two genuinely new paradigms beyond the order-free CF-scoring family were built, run, and
independently verified. Both close with **mechanistic** diagnoses, not bare null results.

## Paradigm A — SASRec (self-attentive sequential)  →  REJECT_FLOOR
First non-CF-scoring model: encodes user **play order** via causal self-attention.

| variant | solo_acc | corr_z | eq_blend Δ |
|---|---:|---:|---:|
| maxlen 20 | 0.61922 | 0.5326 | −0.01420 |
| maxlen 50 (base) | 0.53331 | 0.3218 | −0.00080 |
| d128 maxlen 50 | 0.52971 | 0.3085 | −0.00090 |
| maxlen 100 | 0.50750 | 0.1661 | +0.00030 |

**Monotonic pattern:** the more genuine long-range sequence modelling (longer context), the
WORSE the fit. Root cause = **objective mismatch**: this is a *set-membership* task (held-out
positive is randomly masked, not chronological "next"), so SASRec's autoregressive objective is
orthogonal to it — and that orthogonality is **noise, not signal** (blend never beats emb128).
Conclusion: order-free CF is the *correct* inductive bias because the task is not sequential.

## Paradigm B — GBDT / FM model collaboration (user-requested)  →  REDUNDANT
Non-linear LightGBM fusion, cross-split (train seed123+seed7, eval seed42), anti-stacker-trap
(uniform splits only, eval sampler never seen, symmetric within-user-varying features only).

| feature set | solo_acc | corr_z vs lgcn | Δ solo | tier |
|---|---:|---:|---:|---|
| LightGCN-only (+pop, interactions) | 0.76555 | 0.9734 | +0.0005 | REDUNDANT |
| Multi-model {lgcn, itemknn, ease, pop} | 0.76515 | 0.9681 | +0.0001 | REDUNDANT |

**Both essentially RECONSTRUCTED LightGCN** (corr ~0.97). The tree's gain is dominated by
`z_lightgcn` (83k–94k) while itemknn/ease/pop combined contribute little. This reproduces the
boundary-analysis finding (item popularity residualized against LightGCN = chance) and matches
public reality (Stage2 alone 0.74594, logreg stacker 0.75355 — both far below emb128 0.77745).
Model collaboration cannot beat LightGCN because every available scorer is a ~0.97-correlated,
individually-weaker view of the same ranking.

## Why no paradigm beats emb128 — unified explanation
Three independent lines of evidence now converge on a **structural ceiling**:
1. **Intrinsic Bayes ceiling** — 21.4% of candidates are simultaneously mis-ranked by every
   model; boundary covariate expansion showed this is irreducible (all covariates → chance).
2. **Scorer redundancy** — every non-LightGCN scorer (item-CF, EASE, ALS, popularity, even a
   GBDT fusion of them) correlates ~0.97 with LightGCN and is individually weaker.
3. **Wrong-paradigm orthogonality** — the only genuinely *decorrelated* models (hyperbolic
   geometry corr 0.75, SASRec corr 0.16–0.53) are decorrelated because they are solving a
   *different* problem; their orthogonality is noise, so blends drag emb128 down, not up.

## Exhausted axes (all evidence-closed)
CF families (LightGCN/SGL/DirectAU/DNS/xSimGCL/item-CF/EASE/ALS/MultiVAE) · capacity
(emb64→256 plateau) · hard-negatives · seed ensembling (4/8) · cross-capacity blend ·
logreg stacker · hyperbolic geometry · **SASRec sequential** · **GBDT/FM collaboration**.

## Standing conclusion
**emb128 4-seed ensemble (uniform 0.76505, public 0.77745) is the definitive final #1.**
No new paradigm produces a candidate that beats it on the public-LB surrogate. The remaining
gap to a higher score is the intrinsic Bayes ceiling of the data, which is non-recoverable by
any model-side method. Recommended next focus: lock the eCampus one-file reproducible bundle
for emb128 4-seed; treat further model exploration as exhausted unless new external data or a
rule change appears.
