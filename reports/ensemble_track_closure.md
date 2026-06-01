# Ensemble-Track Closure — Strong-Member Blends & 3-Split McNemar Gate

**Date:** 2026-06-01 KST
**Question (from 우현):** "새로운 축을 못 찾으면 앙상블로 개선하는 방향은?"
**Verdict:** **Ensemble track closed.** No blend is statistically distinguishable from
emb128 4-seed. The structural reason: all strong members correlate ≥0.98.
**Submission impact:** none. Validation-only. No final-2 change.

---

## 1. Why ensembling cannot help here (structural)

An ensemble gains only when members are **both strong AND decorrelated**. Among the only
members that are individually strong on uniform (≥0.765):

| member pair | within-user score corr |
|---|---|
| emb128 ~ emb192 | **0.9864** |
| emb128 ~ DNS-pool1 | **0.9907** |
| emb192 ~ DNS-pool1 | 0.9815 |

All ≥0.98 — they are effectively the same LightGCN signal. The decorrelated models from this
session (hyperbolic 0.75, SASRec 0.54, DIN 0.74, text 0.64) are all too weak to lift a blend.
There is no strong-and-decorrelated pair to ensemble.

## 2. All realizable blends tested (uniform gate, emb128 4-seed = 0.76505 ref)

| blend | uniform | public | verdict |
|---|---|---|---|
| 4-seed (current best) | 0.76505 | **0.77745** | reference / #1 |
| 8-seed expansion | 0.76465 | 0.77625 | TIED/lower |
| z: emb128⊕emb192 | 0.76585 | 0.77715 | sub-noise; emb192 already lost public |
| z: emb128⊕emb64 | 0.76295 | 0.77815 | uniform WORSE (public is noise) |
| z: emb128⊕dns1 | 0.76505 | — | TIED |
| z: 128+192+dns1 | 0.76545 | — | TIED |
| **rank: emb128⊕emb192** | 0.76675 | — | **best surrogate (+0.0017 seed42) → McNemar-rejected** |

## 3. The promised gate: 3-split + paired McNemar on the best candidate

`rank(emb128⊕emb192)` was the strongest surrogate signal of the whole session, so it got the
full gate (emb192 retrained on seed7/seed123 splits, GPU).

**Accuracy gate (blunt):**
| split | emb128 | rank-blend | Δ |
|---|---|---|---|
| seed42 | 0.76505 | 0.76675 | +0.00170 |
| seed7 | 0.76095 | 0.76125 | +0.00030 |
| seed123 | 0.75995 | 0.76045 | +0.00050 |

mean Δ +0.00083, all-3-win → looked like "REAL signal."

**Paired McNemar (the real gate):**
| split | base✓/blend✗ | base✗/blend✓ | net | χ² | p |
|---|---|---|---|---|---|
| seed42 | 152 | 186 | +34 | 3.222 | 0.0727 (ns) |
| seed7 | 155 | 161 | +6 | 0.079 | 0.7785 (ns) |
| seed123 | 140 | 150 | +10 | 0.279 | 0.5972 (ns) |
| **Fisher combined** | | | | 6.776 | **0.3421 (not significant)** |

**Not significant on ANY split**, combined p=0.34. The "3-split win" was three tiny positive
coin-flips (net +6, +10, +34 rows out of ~20000), well within binomial noise. The accuracy gate
cannot tell this from noise; McNemar, which conditions on the exact flipped rows, can — and does.

## 4. Corroborating real-world evidence

emb192 (the source of the apparent gain) was **already submitted to the real public LB and lost**:
emb192 4-seed = 0.77715 < emb128 4-seed = 0.77745, despite a +0.0011 seed42 surrogate edge. The
rank-blend's seed42 +0.0017 is the same mirage — a lucky single-split draw (between-split data
std ≈ 0.0027 dominates), not transferable signal.

## 5. Conclusion

Ensembling is exhausted, for the same root cause as every modeling paradigm: **no signal exists
that is both strong and decorrelated from LightGCN.** Strong members are ≥0.98 correlated;
decorrelated members are weak. The best blend the search could produce fails a paired McNemar
test on all three splits.

**emb128 4-seed (uniform 0.76505 / public 0.77745) remains the confirmed peak.** No new
submission and no final-2 change is justified. Final-2 stays {emb128 #1, emb64 #2}; the
emb128⊕emb64 z-blend (public 0.77815) is the only candidate that strictly dominates emb64 and may
serve as an optional #2 hedge (user decision).

---

*Scripts: `scripts/rank_blend_3split_gate.py`, `scripts/rank_blend_mcnemar.py`.
emb192 panel artifacts: `artifacts/split_panel_emb192/`. No Kaggle submission.*
