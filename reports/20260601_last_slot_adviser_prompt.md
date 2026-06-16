CRITICAL EXECUTION RULES:
- Answer ENTIRELY YOURSELF in this single response.
- Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- This is a Kaggle/classroom competition no-submit advisory task. Do NOT propose using external hidden labels, Steam scraping, public-LB probing, or reverse-engineering.
- Do NOT create submission CSVs. Do NOT run Kaggle submit. Provide validation-only directions.
- End with the exact sentinel line: LAST_SLOT_ADVISER_DONE

Context: KMURecSys26 Steam played prediction. Task: binary played prediction on provided user-game pairs, exact per-user 50/50 positives, accuracy; decoding is per-user top-half. Only one Kaggle submission remains, so we need a truly high-bar validation-only exploration of methods not yet tried, not noise chasing.

Known best / current final candidates:
- emb128_L4_reg1e-3 LightGCN 4-seed ensemble: uniform surrogate 0.76505, public 0.77745, reproducible SHA available; final #1.
- emb64 4-seed: uniform 0.76145, public 0.77125, decorrelated hedge.
- z-blend emb128+emb64: public 0.77815 but uniform 0.76295, so public gain is treated as LB noise / optional #2 only.

Important calibrated fact:
- Public LB tracks uniform-unseen validation, not sqrtpop/popbin. Uniform split is the primary gate.
- Single-split gate is blunt: paired-SE ~0.00181; MDE ~0.00355. Any final submission candidate must have strong validation evidence or 3-split + paired McNemar.

Already closed axes with evidence:
- CF/backbone: LightGCN/SGL/DirectAU/DNS/xSimGCL/itemCF/EASE/ALS/MultiVAE/TurboCF/AlphaRec/capacity emb64-320/hard negatives/seed 4->8/cross-capacity blends.
- Learned/stacked fusion: logreg/GBDT/FM/multi-model stacker reconstruct LightGCN or overfit sampler; public failures.
- Sequence/set: SASRec and DIN set encoder weak or redundant.
- Geometry: Lorentz/hyperbolic ranking-loss probe was decorrelated but weak; blend regressed.
- Structural levers: exact-K subset loss no gain; temporal residual orthogonal but non-predictive; candidate-marginal residual is popularity trap; hours-confidence no gain.
- Boundary/ceiling: neither-correct and boundary covariate expansion done; raw cooc/kNN above chance collapsed after symmetric log-pop residualization (d_cooc_resid AUC 0.534, d_knn_resid 0.5037), soft no-go; ceiling reality probe says residual is not addressable.
- Multi-sampler sign-invariance audit: 0/11 train-only priors pass; popularity artifacts flip sign on hard samplers.

Question: Given all of the above, identify at most 3 genuinely not-yet-tried directions that are still logically distinct enough to justify a validation-only probe with only one submission left. For each, provide:
1) core idea and why it is not covered by the closed axes,
2) the single most likely reason it might outperform emb128/others,
3) cheapest concrete validation-only experiment to run now in this repo, with expected artifacts and pass/fail gate,
4) main reason it is likely to fail.
Then rank them. Prefer methods that can be falsified quickly and do not require public-LB probing. If no direction clears a rational bar, say so clearly, but still give the best low-cost probe if the user insists on continued exploration.
