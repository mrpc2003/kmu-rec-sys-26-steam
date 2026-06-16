# Latest paper / Hugging Face model feasibility scan

- Date: 2026-06-02 KST
- Scope: KMU RecSys 26 Steam played prediction, validation-only exploration.
- Safety: no hidden/test label access, no Kaggle submission, no candidate/submission CSV materialization.
- Raw search artifacts:
  - `reports/20260602_latest_papers_arxiv_raw.json`
  - `reports/20260602_selected_papers_arxiv_metadata.json`
  - `reports/20260602_hf_model_probe_raw.json`

## Current task constraint recap

The task is not open-ended top-N recommendation. The Kaggle candidate set is a per-user balanced binary ranking task: select exactly half of each user's candidates as played. The strongest known local base is the emb128 L4 reg1e-3 4-seed LightGCN family. Existing closures include LightGCN capacity/seed expansion, SGL/DirectAU/xSimGCL/DNS, itemKNN/EASE/ALS/MultiVAE, SASRec, GBDT/FM fusion, TAG-CF, layer mixture, and SL@K-lite continuation.

Therefore, a new paper/model is useful only if it introduces either:

1. a materially different set/top-K/boundary objective that was not already approximated by SL@K-lite;
2. an inference operator that changes boundary ranking with multi-split sign stability;
3. a non-adjacency signal from allowed train-only data, especially review text, that remains predictive after controlling for LightGCN and popularity.

## Paper shortlist and triage

| priority | source | mechanism | distinct from closed axes? | fit to this competition | action |
|---:|---|---|---|---|---|
| 1 | RAGR: Review-Augmented Generative Recommendation, arXiv `2605.17267` | interleaves item semantic IDs and review semantic IDs, then aligns item generation with DPO | Partly. Review text is a non-adjacency signal, but full generative sequence objective overlaps the already-rejected SASRec/sequence framing. | We have review text in `train.json`, but no test-time candidate review text. The feasible reduction is review-derived item/user semantic centroids or semantic-ID clusters, not full RAGR. | Run a train-only Qwen3/ModernBERT review-embedding residual probe; do not build full GR first. |
| 2 | Qwen3 Embedding report, arXiv `2506.05176` via HF model cards; Qwen3-VL Embedding report `2601.04720` | strong public pretrained text/multimodal embeddings/rerankers | Yes only if it extracts review semantics that prior TF-IDF/MiniLM missed. | Public pretrained models are allowed if reproducible. Use train reviews only; no Steam scraping. | Most actionable HF probe: Qwen3-Embedding-0.6B review semantic profile + boundary/reranker variant. |
| 3 | DynamicPO, arXiv `2605.00327`; MASS-DPO, arXiv `2605.10784` | dynamic/multi-negative preference optimization to sharpen boundaries and avoid preference collapse | Partly. It is boundary-aware but overlaps DNS/hard-negative and SL@K-like objectives. | Needs a generative or pairwise preference training setup. Direct application to LightGCN is nontrivial and likely high-cost. | Defer unless review-embedding probe shows real orthogonal signal. |
| 4 | TopKGAT, arXiv `2601.18432`, WWW 2026, GitHub `StupidThree/TopKGAT` | architecture derived from differentiable Precision@K gradient dynamics | Conceptually new architecture, but the top-K objective axis was already attacked by SL@K-lite and rejected. | Full implementation cost is high; likely another graph-CF encoder with high correlation to LightGCN on this task. | Not first priority. Only consider a tiny advisory/code-read after text probe. |
| 5 | Talos, arXiv `2601.19276`, WWW 2026 | optimizes top-K accuracy under distribution shift | Related to top-K metric optimization. | Potentially interesting, but still objective-geometry axis after SL@K-lite rejection. | Read if time, but not immediate GPU bet. |
| 6 | Breaking the Top-K Barrier / SL@K, arXiv `2508.05673`, KDD 2025, GitHub `Tiny-Snow/IR-Benchmark` | SoftmaxLoss@K, quantile top-K truncation, reported +6.03% average | Already reduced to SL@K-lite continuation and rejected on this project. | Current implementation produced consistent negative deltas vs BPR continuation. | Closed unless a much more faithful full benchmark port is explicitly desired. |
| 7 | TAG-CF / test-time aggregation family | test-time aggregation/inference operator | Already probed with a 3-split panel. | Best fixed mean delta only +0.000767 and not sign-stable enough. | Closed for submission. |

## Hugging Face model feasibility

HF Hub metadata was queried with `huggingface_hub` and config download was smoke-tested for representative non-gated models. The server has 4× Tesla V100-PCIE-32GB GPUs. V100 lacks native BF16 acceleration, so models whose cards assume BF16 should be loaded with FP16 or avoided for first probes.

| priority | model | gated | license/tag | why it matters | feasibility verdict |
|---:|---|---:|---|---|---|
| 1 | `Qwen/Qwen3-Embedding-0.6B` | false | Apache-2.0 | latest high-quality text embedding, 0.6B, 32K context, 1024-dim, instruction-aware | **Use first.** Config download succeeded; likely fits V100 in FP16. |
| 2 | `nomic-ai/modernbert-embed-base` | false | Apache-2.0 | much lighter text embedding, supports 768/256 dim Matryoshka truncation | **Fallback / ablation.** Good if Qwen3 is too slow. |
| 3 | `BAAI/bge-m3` | false | MIT | very strong multilingual embedding, 1024 dim, huge public usage | **Baseline embedding.** Less new but robust and easy. |
| 4 | `Qwen/Qwen3-Reranker-0.6B` | false | Apache-2.0 | pairwise text-ranking model for query-document relevance | **Boundary-only candidate.** Expensive if run on all rows; feasible on rank boundary pairs after summaries are built. |
| 5 | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | false | Apache-2.0 | strong instruction embedding/rerank-like representation | Feasible, but lower priority than Qwen3. |
| 6 | `jinaai/jina-embeddings-v4` | false | model card did not expose a license tag in the HF API probe | powerful multimodal/text embedding, 3.8B, technical report `2506.18902` | Defer. Heavier; multimodal advantage is mostly irrelevant because game images/metadata are unavailable. |
| 7 | `Qwen/Qwen3-VL-Embedding-2B` | false | Apache-2.0 | state-of-the-art multimodal embedding, report `2601.04720` | Defer. No visual candidate metadata; V100 BF16 caveat. |
| 8 | HF recommender-specific weights such as `viberec/*SASRec*`, `deem-data/recbole-models` | mostly public | varies | pretrained SASRec/RecBole-style artifacts | Not directly transferable because user/item ID spaces are different; useful only as code/style reference. |

## Most actionable validation-only probe

### Probe A — Qwen3 review semantic residual

Goal: test whether modern HF embeddings extract a train-only review semantic signal that is not already LightGCN/popularity.

1. Use only `train.json` review text and known validation folds.
2. Encode review texts with `Qwen/Qwen3-Embedding-0.6B` in FP16. Fallback: `nomic-ai/modernbert-embed-base` with `truncate_dim=256`, then `BAAI/bge-m3`.
3. Build item semantic profile: mean/robust mean of reviews for each `gameID` in fold train.
4. Build user semantic profile: mean of the user's review embeddings, optionally hours-weighted.
5. Candidate score: cosine(user_profile, item_profile), plus variants residualized against log popularity and base LightGCN score.
6. Evaluate three calibrated uniform splits using per-user top-half. Gate:
   - fixed variant across all splits;
   - mean Δ ≥ 0.00355 for strict pass;
   - fixes > breaks and paired exact/McNemar support;
   - no candidate CSV and no Kaggle submit.

Kill condition: if Qwen3 semantic residual is near zero or negative after residualizing against base/popularity, close all HF text-embedding axes. This would be stronger evidence than prior TF-IDF/MiniLM because Qwen3 is substantially newer and stronger.

### Probe B — Qwen3 reranker boundary-only rescue

Run only if Probe A gives any 3-split positive signal.

1. For each validation user, take only rank K/K+1 or a small boundary band from the base LightGCN prediction.
2. Build compact text summaries:
   - user side: top representative snippets or centroid-nearest reviews from the user's history;
   - game side: representative snippets for the candidate game from fold-train reviews.
3. Use `Qwen/Qwen3-Reranker-0.6B` to rescore boundary pairs.
4. Swap only if reranker margin is strong and fixed threshold is predeclared.

Kill condition: if boundary reranker fixes are not > breaks on at least 2/3 splits, stop. Do not tune thresholds on one split.

## Explicit non-actions

- No Kaggle submission was made.
- No test candidate labels were accessed.
- No public leaderboard probing was performed.
- No candidate/submission CSV was created.
- No external Steam review scraping or hidden-label acquisition is allowed or needed.

## Recommendation

The only genuinely worth-running path from the latest paper/HF scan is **HF review semantics**, starting with `Qwen/Qwen3-Embedding-0.6B`. Top-K objective papers are newest and relevant academically, but this project already ran an SL@K-lite continuation and got a clean 3-split rejection. Recommender-specific HF pretrained weights are not directly usable because the user/item ID vocabulary is dataset-specific. Therefore, the next concrete step should be a no-submit Qwen3 review-embedding residual probe on the calibrated uniform split panel.
