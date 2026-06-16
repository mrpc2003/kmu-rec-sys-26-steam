# Web/arXiv 신규축 탐색 — KMURecSys26 Steam last-slot triage

**Date:** 2026-06-01 KST
**Safety:** `validation_only=true`, `candidate_csv_written=false`, `kaggle_submit_executed=false`
**Raw arXiv search artifact:** `reports/20260601_web_arxiv_search_raw.json` (46 unique arXiv hits)

## 0. Search scope

Searched arXiv API, OpenReview, ACM/RecSys pages, GitHub/community repositories, and competition solution repositories for 2024–2026 recommender-system methods. AI-Q backend (`127.0.0.1:8101/8100`) was unavailable, so this scan used direct arXiv/web search. Parallel delegated web agents timed out and were not used as evidence.

Primary query clusters:

- graph collaborative filtering, ranking-objective architecture, top-K optimization
- test-time aggregation / TAG-CF
- Rankformer / TopKGAT / SL@K
- preference optimization / DynamicPO / DPO for recommendation
- diffusion / flow matching / generative recommendation
- PU / weak supervision / Correct-and-Weight / negative sampling
- denoising noisy implicit feedback
- RecSys Challenge 2025 GitHub solutions and generative-rec community lists

## 1. Important source-backed findings

| Source | What it proposes | Status vs our closed axes | Last-slot relevance |
|---|---|---|---|
| TopKGAT, WWW 2026 — `arXiv:2601.18432`, GitHub `StupidThree/TopKGAT` | Architecture derived from differentiable top-K/Precision@K gradient; band-pass activation focuses near top-K boundary | **Not exactly tried.** Closer to metric-aligned architecture than BPR/LightGCN; distinct from hyperbolic and TAG-CF. | **Highest novelty** but implementation cost high; cheap approximation possible as top-K boundary loss/fine-tune. |
| SL@K, KDD 2025 — `arXiv:2508.05673`, GitHub `Tiny-Snow/IR-Benchmark` | SoftmaxLoss@K using quantile/truncation-aware smooth upper bound for NDCG@K | **Not exactly tried.** We tested exact-K subset loss, but not SL@K/quantile Top-K objective. | Most plausible immediate objective-level probe because task is per-user top-half/top-K. |
| Rankformer, WWW 2025 — `arXiv:2503.16927`, GitHub `StupidThree/Rankformer` | Graph Transformer inspired by ranking objective gradients; global information + ranking-guided attention | Partially related to our untried aggregation/operator axis; not covered by sequence SASRec or hyperbolic. | Could be tried only if TAG-CF/operator axis shows signal; otherwise likely redundant/too expensive. |
| TAG-CF, NeurIPS 2024 — `arXiv:2404.08660`, GitHub `snap-research/Test-time-Aggregation-for-CF` | Plug-and-play test-time aggregation to enhance MF/DirectAU/BPR; reported gains on ML/Yelp/Gowalla/Amazon | Already in progress locally as `scripts/tagcf_testtime_aggregation_probe.py`. | Live validation-only probe; not a submission candidate unless it clears MDE+McNemar and then 3-split. |
| DynamicPO, DASFAA 2026 — `arXiv:2605.00327` | Dynamic Boundary Negative Selection + dynamic beta for multi-negative preference optimization; prevents multi-negative collapse | Conceptually new vs vanilla BPR, but overlaps with our DNS/hard-negative and exact-K closures. | Possible **2nd-tier** fine-tune probe only; must include old-loss continuation control. |
| MASS-DPO — `arXiv:2605.10784` | Multi-negative active sample selection for DPO/PL objective | Same family as DynamicPO; not directly recsys-CF-specific enough. | Lower priority than DynamicPO/SL@K. |
| FlowCF, KDD 2025 — `arXiv:2502.07303` | Flow matching for collaborative filtering | Not tried; true generative/flow axis | Too heavy for last slot; likely slow and may not beat saturated graph CF quickly. |
| PreferDiff, ICLR 2025 OpenReview | Diffusion recommender with BPR/log-likelihood preference objective and multi-negatives | Not tried; generative diffusion axis | Heavy implementation; better only if we had days, not a final-slot fast probe. |
| Correct and Weight / CW, `arXiv:2601.04291` | Correct false negatives and weight loss for implicit feedback | The idea was known in earlier paper-guided exploration; not clearly fully implemented. | Low expected transfer because real public follows uniform negatives and false-negative correction can become popularity artifact. |
| ICPNS, `arXiv:2602.18759`; TFPS, `arXiv:2602.22521` | community negative sampling / temporal positive filtering | Already conceptually explored; hard/community/temporal gates failed. | Closed or low priority. |
| Debiased message passing / popularity-bias papers, e.g. `arXiv:2605.11145` | Mitigate popularity amplification in GNN-CF | Popularity debiasing is a known trap in this competition; public surrogate favors uniform, not fairness/long-tail. | Reject for final slot unless multi-sampler sign-invariance passes, which prior audits did not. |
| RecSys Challenge 2025 GitHub solutions | Large transformer/LLM/stacking/user-profile embeddings | Different challenge: universal behavioral profiles, multi-task logs, LLM embeddings. | Mostly non-transferable; text/stacking already failed here. |

## 2. Ranked 신규축 candidates after filtering

### #1 — Top-K metric-aligned objective probe (SL@K / TopKGAT-lite)

**Why this is the strongest new axis:**

- Our scoring rule is not ordinary full-ranking NDCG; it is per-user exact top-half. However, this is still a hard top-K boundary problem.
- TopKGAT and SL@K are explicitly designed around the top-K boundary, not just pairwise BPR over random negatives.
- This is distinct from the closed hyperbolic, sequence, stacker, temporal, hours, text, candidate-marginal, and seed/capacity axes.
- It is closest to the live failure mode: remaining errors sit at rank-K/K+1 boundary.

**Why it may fail:**

- We already tested an exact-K conditional subset loss and it was net-zero.
- BPR with uniform negatives is already well aligned with the public surrogate.
- Single-split MDE is ~0.00355, so small top-K objective gains are likely unresolved noise.

**Cheapest validation-only experiment:**

Implement a *TopK boundary fine-tune* rather than full TopKGAT:

1. Start from canonical emb128/L4/reg1e-3 BPR configuration on `val_random_uniform_seed42`.
2. Confound-control from the same pretrain checkpoint:
   - A: +N epochs old BPR loss
   - B: +N epochs TopK/SL@K-style boundary loss
3. Training batches are sampled as per-user candidate mini-sets: `r` positives + `r` uniform unseen negatives, with K=r.
4. Loss approximates top-K truncation with a user-local threshold/quantile and weights samples near that threshold.
5. Gate on isolated `B - A`, not just B vs pretrain.

**Pass gate:** `Δ >= +0.00355` and McNemar p<0.05 on seed42; otherwise no escalation. If passed, expand to 3-split + paired McNemar.

### #2 — DynamicPO / boundary-hard multi-negative preference fine-tune

**Why this is new enough to test only after #1:**

- DynamicPO specifically attacks the collapse where easy negatives dominate and boundary-critical negatives are under-optimized.
- This maps naturally to our boundary-pair error structure.

**Overlap / risk:**

- DNS/hard-negative axes and exact-K loss already failed or tied.
- Hard-negative training can easily overfit to non-public negative distributions.
- Needs strict old-loss continuation control and uniform-only candidate sampler.

**Cheapest probe:**

Same pretrain/control pattern as #1, but the variant samples a random negative pool, chooses boundary negatives by current model score, and applies dynamic-beta/margin only for near-boundary negatives. Reject unless isolated gain exceeds MDE.

### #3 — Multi-interest / item-to-interest routing probe

**Why it appeared from communities:**

- RecSys 2025 community papers include collaborative interest modeling / item-to-interest routing, where one user embedding is replaced by multiple interest prototypes.
- This differs from a single LightGCN user vector and could help users with mixed genres.

**Why likely to fail here:**

- ItemKNN max/top-k, boundary cooc, and kNN consensus already mostly collapsed after popularity residualization.
- SASRec/DIN set-style attempts did not reach emb128 strength.

**Cheap probe:**

Use frozen item embeddings or itemKNN similarity to build per-user K-means/medoid interest prototypes and score candidates by max prototype similarity; evaluate alone and 50/50 z-blend. This is CPU/low-GPU and should be rejected if it only produces popularity/cooc-like behavior.

## 3. Rejected / not worth last-slot escalation

- Full FlowCF / PreferDiff / diffusion recommender: novel but too expensive; no clear reason to beat saturated LightGCN on a 20k candidate top-half task before deadline.
- LLM/review-augmented/generative recommendation: text axis and stackers already failed; no hidden textual signal strong enough.
- Popularity-bias mitigation / debiased message passing: directly conflicts with the public-surrogate lesson and previous pop-trap audits.
- Community negative sampling / temporal positive filtering: already explored via ICPNS-like community splits, temporal compatibility, hours confidence, and multi-sampler audits.

## 4. Recommended next action

Continue the already-running TAG-CF probe. In parallel or immediately after, implement **TopK boundary fine-tune (SL@K-lite)** as the next research-backed validation-only candidate.

Priority order:

1. Finish TAG-CF validation-only result.
2. If TAG-CF rejects, run SL@K/TopK boundary fine-tune with BPR continuation control.
3. Run DynamicPO-lite only if #1 shows a non-noise boundary gain or if implementation is cheap after reusing #1 infrastructure.

**No Kaggle submission is justified from web search alone.** Any candidate must first clear uniform + paired McNemar, then 3-split panel before asking for explicit final-slot approval.
