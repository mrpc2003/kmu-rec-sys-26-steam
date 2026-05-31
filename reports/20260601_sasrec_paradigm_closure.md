# SASRec Sequential Paradigm — Track Closure

**Status: CLOSED — REJECT_FLOOR (all variants, independently verified)**

## What was tested
The first **non-CF-scoring** paradigm. Every previously-closed axis (LightGCN / SGL /
DirectAU / DNS / xSimGCL / item-CF / EASE / ALS / MultiVAE / capacity / hyperbolic) is an
**order-free** collaborative scorer. SASRec (Kang & McAuley, ICDM 2018) instead encodes the
user's **play order** with a causal self-attention Transformer and predicts the next item
autoregressively — a fundamentally different inductive bias. EDA confirmed viability (median
sequence length 21, 97.7% users ≥10 plays, dates 100%).

## Independent verification (recomputed from score CSVs, not trusting summary.json)

| run | solo_acc | corr_z vs emb128 | eq_blend (50/50 z) | Δ vs ref | tier |
|---|---:|---:|---:|---:|---|
| emb128 4-seed (REF) | 0.76505 | 1.0000 | — | — | REF |
| SASRec d64 maxlen20 | 0.61922 | 0.5326 | 0.75085 | −0.01420 | REJECT_FLOOR |
| SASRec d64 maxlen50 (base) | 0.53331 | 0.3218 | 0.76425 | −0.00080 | REJECT_FLOOR |
| SASRec d128 maxlen50 | 0.52971 | 0.3085 | 0.76415 | −0.00090 | REJECT_FLOOR |
| SASRec d64 maxlen100 | 0.50750 | 0.1661 | 0.76535 | +0.00030 | REJECT_FLOOR |

All four trained cleanly (loss 1.34 → 0.69, no NaN collapse) yet **solo_acc never reached the
popularity floor 0.684**. The single positive blend delta (d64_L100, +0.00030) is within noise
(0.0007) and below MDE (0.00355) — not a signal.

## Diagnosis — objective mismatch, not a bug
A **monotonic pattern** proves the root cause:

```
context length ↓  ⇒  solo_acc ↑  AND  corr_z ↑
  maxlen 20  : solo 0.619, corr 0.533   (short = recency / co-play, closer to the task)
  maxlen 50  : solo 0.533, corr 0.322
  maxlen 100 : solo 0.508, corr 0.166   (long = pure next-item, farthest from the task)
```

The **more** the model does genuine long-range sequence modelling, the **worse** it fits this
task. The task is **set-membership** ("is game *i* in user *u*'s played set?"), where the
held-out positive is a *randomly* masked item, not the chronological "next" item. SASRec's
autoregressive "what comes next" objective is therefore largely orthogonal to set-membership
(corr 0.16–0.53), and that orthogonality is **noise, not complementary signal** — confirmed by
the blend never beating emb128.

## Conclusion
**Order-free collaborative filtering is the correct inductive bias for this competition**, not
because sequence models are weak, but because the task is not sequential. This is a positive,
generalizable finding, not just another null result.

emb128 4-seed ensemble (public 0.77745) **remains the final #1**. No submission from this track.

## Implication for remaining exploration
The only genuinely-untried paradigm left is **GBDT / Factorization-Machine over engineered
symmetric features** (user activity, item popularity, co-play counts, hours stats, LightGCN
score as a feature) — explicitly suggested by 우현. High popularity-trap risk (the logreg
stacker already regressed 0.76245→0.75355), so it must be cross-split trained on symmetric
features only and gated identically.
