We are in the last-submission phase of a Kaggle/classroom recommender competition: KMU RecSys 26 Steam played prediction.

Task facts:
- Binary played/not-played prediction for userID-gameID candidate pairs.
- Public metric: Accuracy.
- Test candidates are known users/items and are exactly 50/50 played/non-played per user, so inference is per-user top-half ranking.
- Our strongest verified model family is LightGCN trained on binary implicit graph with BPR/uniform negatives.
- Current public best by our team: 0.77815; current leaderboard #1 appears 0.78065, so the remaining public gap is about +0.00250.
- No Kaggle submission may be made from this research. We only want validation-only ideas.

Known closed or weak axes; do not recommend these again unless you can identify a materially different mechanism:
- ALS, BPR MF, itemKNN, EASE, MultiVAE.
- LightGCN capacity sweeps: emb64/128/192/256/320; emb128 is the sweet spot, emb192 lost on public.
- Seed ensembles beyond 4 seeds; 8 seeds tied/worse.
- Cross-capacity blends and rank/z blends mostly noise or public-LB noise.
- SGL, DirectAU, xSimGCL, DNS/hard negatives.
- SASRec/DIN/sequential or set-order models; they mismatched the set-membership task.
- Review text / MiniLM / TF-IDF text features; weak.
- Temporal/hour features; weak or non-transferable.
- Exact-K subset loss tried and net-zero.
- Hyperbolic/Lorentz geometry; redundant/weak.
- Candidate-marginal/global quota and popularity debiasing; public surrogate trap.
- Co-occurrence/kNN residual boundary covariates after log-pop residualization; near chance.
- Multi-sampler sign-invariance audit killed popularity-like priors.

Current live signal:
- TAG-CF-style parameter-free test-time neighbor aggregation on uniform seed42 produced a weak but nontrivial signal:
  best score_tag_sym_a0p25_raw, row_accuracy 0.765253, delta_vs_base +0.0032006, flips 582, candidate_fixes 323, base_breaks 259, McNemar p=0.0090.
- However, it missed the single-split MDE threshold +0.00355, so we are running uniform split seed7 and seed123 as a 3-split panel.

Research request:
1. Search arXiv, OpenReview, RecSys/KDD/WWW/CIKM, GitHub, and competition writeups for 2024-2026 implicit-feedback recommendation ideas that are genuinely distinct from the closed axes above.
2. Prioritize ideas that can be reduced to a cheap validation-only probe in this repo before a final submission slot.
3. Focus especially on mechanisms aligned with per-user top-half/top-K boundary accuracy: top-K metric-aligned objectives, test-time/inference aggregation, boundary-aware preference optimization, multi-interest routing, PU/denoising methods that are not just hard-negative/popularity traps.
4. For each candidate, provide: source URL, mechanism, why it is distinct from closed axes, minimal validation-only probe, cost estimate, expected sign/risk, and kill/escalation gate.
5. Be harsh: novelty alone is not enough. Reject expensive full-model papers if they are just new encoders over the same saturated binary graph.
6. Explicitly call out whether TAG-CF's seed42 signal should justify a full-test candidate only after 3-split paired McNemar, and what additional anti-overfit checks are needed.

Return a concise but source-cited research memo in Korean or English. Do not suggest Kaggle submission without explicit validation gates and human approval.