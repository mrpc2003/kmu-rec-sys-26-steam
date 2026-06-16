# Local train-review semantic topology smoke — closure

- Date: 2026-06-16 UTC
- Safety: validation-only; no Kaggle submit; no `submissions/` write; no full-test candidate materialization
- Script: `scripts/semantic_residual_probe.py`
- Command family: `tfidf-svd-smoke`, split `val_random_uniform_seed42`, `tfidf_dim=128`
- Artifact dir: `artifacts/semantic_topology_smoke/tfidf-svd-smoke`
- JSON report: `reports/20260616T_local_review_semantic_topology_smoke_tfidf-svd-smoke.json`
- Detailed Markdown report: `reports/20260616T_local_review_semantic_topology_smoke_tfidf-svd-smoke.md`

## Result

The local train-review semantic profile/topology smoke **does not pass** the seed42 gate.

| metric | value |
|---|---:|
| base emb128 4-seed accuracy | `0.765053` |
| best semantic blend variant | `base_plus_ap0p050_sem_bin` |
| best blend accuracy | `0.765553` |
| best delta vs base | `+0.000500` |
| fixes / breaks | `115 / 105` |
| McNemar exact p | `0.5441` |
| strict pass | `false` |
| semantic binary solo accuracy | `0.552410` |
| semantic hours-weighted solo accuracy | `0.556611` |
| coverage | `19996 / 19996` validation rows |

## Gate verdict

**KILL_WEAK_SOLO_AND_TINY_BLEND.** Semantic-only ranking is far below the `solo >= 0.7350` smoke gate, and the best blend lift is only `+0.000500`, below the `+0.0007` seed42 escalation threshold and not statistically supported (`p=0.5441`).

Do not escalate this TF-IDF/SVD local-review topology/profile design to a 3-split panel, Qwen3/LLM reranking, or full-test materialization.

## What remains open

Only a materially stronger semantic encoder could reopen the semantic axis, and only if it first clears the same seed42 gate without external Steam metadata, appid reverse mapping, external review scraping, or hidden/test labels.

LOCAL_REVIEW_SEMANTIC_TOPOLOGY_SMOKE_DONE
