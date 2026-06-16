# KMURecSys26 Steam — latest methodology untried-axis scan

- Date: 2026-06-16 UTC
- Scope: latest RecSys / graph-CF / semantic-CF methods after final package lock
- Safety: no Kaggle submit, no `submissions/` write, no full-test candidate materialization
- Current locked references:
  - `final_package/final_slot1_rank_blend_emb128_emb192_LABEL.csv`, public `0.77825`
  - `final_package/final_slot2_lightgcn_emb128L4r3_4seed_LABEL.csv`, public `0.77745`

## Decision

There is still **no submission-ready internal axis**. The only route worth a small validation-only probe is a materially stronger **train-review semantic topology/profile axis** that uses the existing `train.json` text and does not require Steam appid mapping. Everything else in the fresh paper scan is either already closed by local evidence, blocked by missing safe metadata joins, or research-only because it is another graph-CF retune with high correlation risk.

Recommended status:

| Rank | Axis | Classification | Why |
|---:|---|---|---|
| 1 | Train-review semantic topology/profile probe inspired by TAGCF / modern embedding profiles | **TRY, validation-only** | It is the only remaining path that can add non-adjacency signal from allowed local data. Prior TF-IDF/MiniLM-style text probes were weak, but a topology/profile formulation is different enough to merit one bounded smoke if more work is desired. |
| 2 | NT-SSM / neighbor-type-aware contrastive objective | **research-only** | New 2026 CL analysis, but local SGL/DirectAU/xSimGCL/DNS evidence already says contrastive/hard-negative graph objectives damage this small balanced top-half task. |
| 3 | TopKGAT / full top-K objective architecture | **reject-same-family** | Latest and relevant, but local SL@K-lite and exact-K probes are already negative; full architecture is a high-cost graph-CF objective retune. |
| 4 | RaDAR diffusion-asymmetric graph contrastive CF | **reject-same-family** | Mostly GCL/diffusion augmentation for sparse/noisy graphs; local graph-SSL and UltraGCN-style constraint smoke failed the strong+orthogonal condition. |
| 5 | StageCF / burn-down diffusion CF | **research-only / reject for this sprint** | Diffusion CF is structurally different, but heavy and still CF-only; prior diffusion scans warned that strong tuned baselines often beat diffusion recommenders. |
| 6 | Metadata/appid content profile | **blocked** | Still the highest-information path if official `gameID -> Steam appid` mapping appears; blocked without that mapping and should not be reverse-mapped. |

## Local constraints that dominate the literature scan

The competition is not generic open-world recommendation. It is a small, dense, user-local top-half reranking problem:

- train: about 175k interactions, 6,710 users, 2,437 games;
- test pairs: about 20k rows, per-user positive:negative is 1:1;
- canonical decode: per-user top half gets `Label=1`;
- calibrated public surrogate: uniform validation, not popularity-debiased / long-tail / temporal stress splits;
- current public best is already a LightGCN capacity/rank blend at `0.77825`.

Closed local families matter more than paper leaderboard claims. The current `reports/failed_axes.json` and closure reports already close:

- LightGCN capacity/seed retune, cross-capacity frontier, rank-blend residuals;
- SGL / DirectAU / xSimGCL / DNS / hard-negative variants;
- EASE / ItemKNN / BM25 / ALS / WMF / GF-CF / PPR / heat-kernel / Turbo-CF-style item filters;
- MultiVAE / latent generative smoothing;
- SASRec / sequential next-item framing;
- DIN target-conditioned set encoder and set-prediction family;
- pseudo-label / transduction;
- SWA / checkpoint averaging;
- boundary/ridge-fast row flips and boundary feature factories;
- TAG-CF/test-time aggregation full-test family after public-negative transfer;
- UltraGCN-style propagation-free constraint-loss smoke at current design.

## Fresh-source scan

### TopKGAT — arXiv:2601.18432, WWW 2026

Source: https://arxiv.org/abs/2601.18432

TopKGAT derives a graph-attention architecture from a differentiable approximation of top-K metrics. Its abstract says the layer is aligned with Precision@K gradient dynamics and uses a graph-attention-like structure.

**Local disposition: reject-same-family.** This was the right academic direction before local top-K objective probes were run, but the project now has stronger negative evidence:

- `reports/20260601_slk_lite_panel_aggregate.json` is referenced by later axis-loop reports as all three splits negative;
- `scripts/slk_lite_continuation_probe.py` already implemented an old-BPR-control vs SL@K-lite continuation harness;
- exact-K/subset-loss and boundary row-flip variants also failed or transferred poorly.

**Do not run next:** a full TopKGAT port. It would be a high-cost graph-CF objective retune and would need to beat the same top-half boundary failure mode that local SL@K/exact-K evidence already rejects.

### NT-SSM / neighbor-type-aware sampled softmax — arXiv:2605.24015, ICML 2026

Source: https://arxiv.org/abs/2605.24015

The paper analyzes graph-CF contrastive learning by unfolding prediction into multi-hop neighbor-pair weights, then proposes NT-SSM to update neighbor-pair types more selectively than standard sampled-softmax CL.

**Local disposition: research-only.** This is newer than the earlier SGL/xSimGCL sweep and more principled, but it is still an optimization/loss change inside graph CF. Local contrastive and hard-negative attempts were not just under-tuned: they repeatedly showed the same strong-or-orthogonal dilemma, where decorrelation came from weak/noisy models rather than useful complementary signal.

**Possible smoke only if forced:** implement a seed42-only LightGCN continuation control:

```bash
# validation-only sketch; do not materialize full-test candidates
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy --with 'torch==2.5.1' --python 3.11 \
  python scripts/nt_ssm_smoke.py \
    --split-dir artifacts/validation/val_random_uniform_seed42 \
    --base-config emb128_L4_reg1e-3 \
    --epochs 1 \
    --control old-bpr \
    --out-dir artifacts/nt_ssm_smoke/seed42
```

Kill gate: solo must stay within `0.0020` of old-BPR control and 50/50 z-blend vs emb128 4-seed must improve by `> +0.0007`. If not, close without panel expansion.

### RaDAR — arXiv:2603.16800, WWW 2026

Source: https://arxiv.org/abs/2603.16800

RaDAR combines graph generative / diffusion-guided augmentation, asymmetric contrastive learning, and relation-aware edge refinement for noisy sparse graph-CF settings.

**Local disposition: reject-same-family.** The method’s target failure mode is noisy/sparse graph augmentation. This dataset is small, relatively dense, and public-like validation is uniform top-half. SGL/DirectAU/xSimGCL/DNS and the UltraGCN-style constraint-loss smoke already indicate that graph-SSL/constraint orthogonality is weak-model noise here.

Do not run unless a separate diagnostic first proves that local LightGCN errors are caused by noisy edges rather than intrinsic boundary ambiguity. Current reports point the other way.

### StageCF / interests burn-down diffusion — arXiv:2605.05165

Source: https://arxiv.org/abs/2605.05165

StageCF proposes a diffusion process tailored to collaborative interactions, replacing generic Gaussian diffusion with a burn-down / burn-up process over user interests.

**Local disposition: research-only / reject for this sprint.** It is structurally fresher than another LightGCN loss, but it is still pure interaction-CF and likely expensive. Earlier scans already warned that diffusion recommender papers often underperform strong tuned baselines, and local MultiVAE/generative smoothing did not add complementary signal.

Only revisit if the goal changes from final-slot safety to long-horizon research. It is not a next submission axis.

### TAGCF / semantics into topology — arXiv:2602.21099

Source: https://arxiv.org/abs/2602.21099

TAGCF transforms semantic knowledge into topology by using LLM-inferred interaction intents as intermediate attribute nodes in a User-Attribute-Item graph, then applies adaptive relation-weighted graph convolution.

**Local disposition: TRY, but only as a train-review semantic topology/profile smoke.** This is the only fresh method family that can add information not already in the interaction graph. The local raw data has train-side review text at `data/raw/public/data/train.json`; sample records include `text`, `date`, `hours`, and `gameID`. That makes a bounded local semantic probe feasible without external scraping or Steam appid mapping.

Important distinction from blocked metadata:

- allowed local signal: `train.json` review text, `date`, `hours`, `early_access`, `found_funny` when present;
- blocked external signal: Steam appid metadata, genre/tag/category, external reviews, user profiles, owned-games, or reverse mapping.

Important distinction from prior weak text probes:

- do not submit another raw text score or generic TF-IDF residual;
- reduce TAGCF to a falsifiable topology/profile smoke: semantic clusters become attribute nodes or profile centroids, then evaluate whether those scores fix LightGCN errors on the uniform panel;
- require a fixed feature construction before seeing split metrics.

## Exact validation-only probe shape if work continues

### Probe A — local review semantic profile smoke

This is the safest actionable probe because it uses only already-provided train data.

1. Split discipline:
   - use `artifacts/validation/val_random_uniform_seed42` first;
   - build item/user profiles from split train interactions only;
   - never use held-out validation text for the held-out positive row when constructing that user’s profile.
2. Representation:
   - first CPU-cheap baseline: `char/word TF-IDF -> TruncatedSVD(128)` or existing text embedding artifacts if present;
   - optional stronger pass only if cheap baseline has any positive signal: Qwen3/BGE/ModernBERT embeddings from train review text only.
3. Scores:
   - item profile: robust mean of review vectors per `gameID` using split-train rows;
   - user profile: mean of that user’s split-train review/item vectors, optionally hours-weighted as a predeclared ablation;
   - candidate score: cosine(user_profile, candidate_item_profile);
   - blend checks: solo, 50/50 within-user z-blend with emb128 4-seed, and residualized score against popularity/base score.
4. No full-test materialization. Output only validation scores and a markdown/json report under `artifacts/semantic_topology_smoke/` and `reports/`.

Seed42 smoke gate:

```text
solo >= 0.7350
50/50 z-blend vs emb128 4-seed >= emb128_4seed + 0.0007
fixes > breaks against emb128_4seed
no lift concentrated only in high-popularity/head-item buckets
```

Panel escalation gate:

```text
3 uniform splits: seed42, seed7, seed123
fixed variant chosen before panel
mean delta vs emb128_4seed >= +0.0015
min split delta >= 0
positive splits = 3/3
fixes > breaks pooled
paired exact/McNemar p < 0.05 when available
```

If seed42 fails, close the local semantic topology/profile axis and do not run Qwen3/LLM rerankers.

### Probe B — TAGCF-lite semantic attribute graph

Run only if Probe A gives positive signal. Build small attribute nodes from local review clusters, not from external metadata:

- cluster item review vectors into `K_attr` semantic attributes using split-train data;
- add item-attribute and user-attribute edges weighted by split-train review/profile membership;
- score candidates with a simple attribute-overlap cosine or one relation-weighted propagation step;
- compare against Probe A to prove topology adds signal beyond profile cosine.

Kill gate: TAGCF-lite must beat Probe A by `+0.0007` on seed42 and must not regress emb128 blend. Otherwise, topology is extra complexity without signal.

## Explicit non-actions

- Do not create `submissions/*.csv`.
- Do not run `kaggle competitions submit`.
- Do not materialize full-test candidates.
- Do not reverse-map `gameID` to Steam appid.
- Do not query Steam user/profile/owned-games.
- Do not scrape external Steam reviews.
- Do not rerun closed graph-CF objective families without a materially new validation-only design.

## Background-task reconciliation

After this report was drafted, the completed background searches were retrieved and reconciled against the local closure ledger.

### Closed-axis audit

The closed-axis audit reinforced the hard filter rather than changing it. It identified `reports/failed_axes.json` plus the June 12-16 closure reports as the controlling evidence, with the following families excluded unless a genuinely new independent signal source first passes a strict validation gate:

- LightGCN + Stage2/meta-learner stacks;
- item-item linear/spectral filters;
- pseudo-label transduction;
- checkpoint/SWA-like averaging;
- boundary/row-flip/TAG-CF/boundary covariate expansion;
- UltraGCN-style constraint-loss at the current design;
- DIN/set-prediction/exact-K/DPP/candidate-conditioned attention;
- SASRec/sequential modeling;
- hyperbolic geometry;
- contrastive CF / hard-negative / DNS / XSimGCL / SGL / DirectAU;
- ensemble-track strong-member blends;
- hours/date/text/popularity residual cocktails and stacker-trap feature sets.

This supports the report’s main decision: do not surface a method as “new” if it only renames one of these local failure modes.

### Latest-paper search

The latest-paper search found several public-code generative or sequence candidates: Mamba4Rec, SIGMA, PDRec, DiffuRec, GiffCF, ADRec, PreferGrow, CDRec, DCRec, and FAVE. They are not immediate submission axes here for three reasons:

1. The strongest Mamba/sequence candidates are still chronological next-item recommenders, while local SASRec/sequence framing is already closed for this set-membership/top-half task.
2. The diffusion/generative candidates are interaction-only and potentially interesting for long-horizon research, but StageCF-style burn-down diffusion already captures the representative current-paper branch in this report and remains research-only because local MultiVAE/generative smoothing and UltraGCN-style smoke failed the strong+orthogonal test.
3. None of those candidates adds the missing independent information source. They mostly transform the interaction graph/history; the only remaining local non-adjacency source is train-review text.

Therefore the paper shortlist does not change the recommended first action: if any further work is done, run the train-review semantic profile/topology smoke before spending GPU time on another sequence/diffusion backbone.

### Implementation-candidate search

The implementation search independently converged on the same practical candidates already handled above: TopKGAT/Rankformer-style top-K/listwise training, GFormer-style graph transformers, StageCF diffusion CF, uCTRL/AdvInfoNCE contrastive debiasing, DICE/CausE causal debiasing, and TAGCF semantic topology.

Reconciliation:

- TopKGAT/Rankformer are rejected as same-family after local SL@K-lite/exact-K failures.
- GFormer is graph-transformer LightGCN-adjacent and not a new signal source.
- StageCF remains research-only, not a final-slot candidate.
- uCTRL/AdvInfoNCE/RaDAR-style methods are contrastive/debiasing retunes already covered by the closed contrastive/hard-negative family.
- DICE/CausE are causal-popularity/conformity debiasing variants and risk repeating the popularity residual trap.
- TAGCF remains the only useful implementation hint, but only when reduced to local train-review semantic attributes/profiles, not external metadata or appid mapping.

## Bottom line

If the goal is final-package safety, stop here and keep the two locked submissions. If the goal is one more no-submit research probe, the only defensible next step is **local train-review semantic profile/topology smoke**. It is not submission-worthy by itself; it is merely the last unclosed information axis that does not violate the no-submit / no-external-reverse-mapping constraints.

LATEST_METHOD_UNTRIED_AXIS_SCAN_DONE
