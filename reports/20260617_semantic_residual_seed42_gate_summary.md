# Qwen3 Semantic Encoder Seed42 Gate Smoke

Validation-only rerun for the requested one-shot strong text encoder check. No Kaggle submit, no candidate CSV, no hidden labels, no external scraping.

## Setup

- model: `Qwen/Qwen3-Embedding-0.6B`
- split: `val_random_uniform_seed42`
- base: emb128 LightGCN 4-seed, accuracy `0.765053`
- embedding_dim: `1024`
- embeddings_reused: `True` (existing Qwen3 item-embedding artifact; metrics recomputed in this run)
- covered rows: `19996` / `19996`

## Gate result

| best variant | accuracy | Δ vs base | fixes | breaks | McNemar p | pass +0.0007 seed42 gate |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_am0p020_sem_bin_resid` | 0.765253 | +0.000200 | 51 | 47 | 0.7620 | False |

## Verdict

`CLOSE_TEXT_ENCODER_AXIS_FOR_NOW` — best seed42 delta is below the predeclared +0.0007 smoke gate. The standalone semantic score is far weaker than graph co-occurrence; tiny positive blends are noise-scale.

The full earlier 3-split Qwen3/BGE/ModernBERT reports already show the same pattern, so this rerun does not justify more text-encoder budget unless a new text architecture changes the source signal, not just the embedding model.
