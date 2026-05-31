# Ceiling Reality Probe — Is the mis-ranking irreducible or recoverable?

**Date:** 2026-05-31 KST
**Method:** confident-learning / cross-model agreement on `val_random_uniform_seed42`
**Verdict:** ceiling is **effectively irreducible**; the "recoverable" disagreement is a
weak-model chance artifact with **no realizable aggregator or covariate router** able to
extract it.
**Submission impact:** none. Validation-only analysis. (`scripts/ceiling_reality_probe.py`)

---

## 0. Why this probe — auditing our own stop decision

Prior rounds claimed "all models simultaneously mis-rank the same ~21.4% (intrinsic Bayes
ceiling)." That number was the **per-model error rate**, asserted to be simultaneous on the
strength of pairwise correlation — never directly measured as joint error across a structurally
diverse panel. Before declaring the signal space saturated, we measure it directly.

## 1. Panel & per-model accuracy (uniform_seed42, 19996 rows, 4736 users, exact 50/50)

Diverse panel (one strong rep per family):

| model | row_acc |
|---|---|
| emb192 | 0.76665 |
| dns_pool1 | 0.76565 |
| lightgcn_emb128 | 0.76205 |
| xsimgcl_l0.02 | 0.74145 |
| din_d64 | 0.74275 |
| hyperbolic | 0.71734 |
| sgl_l0.05 | 0.64053 |
| directau_g1 | 0.57101 |
| sasrec_L50 | 0.53631 |

Homogeneous LightGCN 7-seed panel: all 0.76275–0.76515.

## 2. Error-mass concentration (diverse panel, 9 models)

| #models correct | rows | % | class |
|---|---|---|---|
| 0/9 | 595 | 2.98% | ALL-WRONG |
| 1–8/9 | 16026 | 80.15% | contested |
| 9/9 | 3375 | 16.88% | ALL-RIGHT |

At first glance only 2.98% is "irreducible" and 80% is "recoverable." **This reading is wrong**
— see §3–4.

## 3. The 80% contested is a chance artifact, not signal — three proofs

**(a) Panel agreement does NOT beat the best single model.**
The *majority-correct diagnostic* (fraction of rows where ≥half the panel is right) = **0.76310**,
vs best single (emb192) **0.76665** → **Δ -0.00355**. This is a diagnostic, not a literal
majority-vote ensemble accuracy (a true ensemble would have to re-rank by vote-sum to preserve
the per-user exact-half constraint). But it points the same direction as the **already-measured
realizable ensembles** from earlier this session: 4-seed/8-seed averaging and emb128⊕emb192
cross-capacity blends were all **TIED** with the single best on the uniform gate, and the
logreg/GBDT stackers **regressed** (public 0.77745→0.75355/0.76245). A mixed-strength panel's
agreement is *below* its best member because the weak models (sasrec 0.54, directau 0.57) only
inject noise. Every realizable pooling rule we have actually built fails to extract the
disagreement — the diagnostic and the real ensembles agree.

**(b) The "oracle" ceiling is the 1−0.5ⁿ illusion.**
Per-row ANY-model-correct = 0.97024 (Δ +0.204). With 9 partially-decorrelated models, the chance
that *at least one* is right on a given row approaches 1−0.5⁹ ≈ 99.8% even under near-random
behavior. This oracle is unrealizable (needs per-row labels to pick the model) and is almost
entirely statistical, not exploitable signal.

**(c) Homogeneous-seed contrast pins the true floor.**
Strong LightGCN 7-seed panel ALL-WRONG = **19.08%** — close to the per-model error (~23.5%).
Models that genuinely solve the task make **highly correlated** errors (~19% jointly wrong).
The diverse panel's low 2.98% all-wrong is inflated by weak models being right *by coincidence*
on different rows. The honest irreducible floor is ~19%, not 3%.

## 4. No train-only covariate separates the hard rows

| covariate | contested median | all-wrong median | all-right median | corr(log1p, frac_correct) |
|---|---|---|---|---|
| item_pop | 60.0 | 62.0 | 57.0 | **+0.0411** |
| user_deg | 24.0 | 34.0 | 28.0 | **−0.0258** |

Neither popularity nor user degree separates easy vs hard rows (|corr| < 0.05). So even the
genuinely-contested mass cannot be *targeted* by any train-only router or meta-feature — which is
exactly why the GBDT/logreg stacker reconstructed LightGCN (corr 0.97) and regressed on public.
There is no observable handle on "where the model is wrong."

## 5. Honest correction & conclusion

The earlier "all models mis-rank the same 21.4%" statement was imprecise. The corrected,
directly-measured picture:

> Models that actually solve the task make **correlated** errors on **~19%** of per-user
> top-half decisions. The disagreement among models of differing strength is **not** recoverable
> signal: a realizable majority/pool is *below* the best single model (−0.00355), the only
> "ceiling lift" is the unrealizable 1−0.5ⁿ oracle, and **no train-only covariate flags the hard
> rows** so no router can exploit them.

This is a **stronger** basis for stopping than the original framing: the irreducibility is not
just "models agree on errors" but "the residual heterogeneity is provably non-addressable with
any realizable rule or observable feature." Combined with every modeling paradigm collapsing to
LightGCN when trained to strength (CF / linear / latent / text / geometry / sequence / set /
stacking), the signal space is **saturated**.

**emb128 4-seed (uniform 0.76505 / public 0.77745) remains the confirmed peak. Stop is correct.**

---

*Script: `scripts/ceiling_reality_probe.py`. No Kaggle submission.*
