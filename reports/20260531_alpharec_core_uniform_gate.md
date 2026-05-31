# AlphaRec-core (ICLR'25) — UNIFORM gate (LM item-rep behavior aggregation)

- split `val_random_uniform_seed42` rows=19996 | text known 1.0000
- **solo uniform: 0.64473** (pop floor 0.684)
- corr(within-user z) vs emb128 = **0.4856**
- **50/50 z-blend: 0.73025** vs emb128 ref 0.76505 → **-0.03480**
- **tier: REJECT_WEAK** — AlphaRec-core solo 0.64473 < floor 0.684; corr_z 0.486; blend50 0.73025 (vs ref -0.03480). LM-representation CF axis is orthogonal-ish but too weak on noisy review text -> closed, like MiniLM/TF-IDF.

## vs prior MiniLM probe

MiniLM probe used the user's OWN review text (solo 0.639, corr_z 0.461). AlphaRec-core uses behavior aggregation of item language reps. Both test the LM-semantic axis.
