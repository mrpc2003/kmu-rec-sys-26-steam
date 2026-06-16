# UltraGCN-style constraint-loss smoke

Safety: validation-only; no Kaggle submit; no `submissions/` write; no full-test candidate.

## Result

- split: `val_random_uniform_seed42`
- artifact dir: `artifacts/ultragcn_constraint_smoke/d128_seed42/val_random_uniform_seed42`
- solo accuracy: **0.71644**
- delta vs emb128 single-seed 0.76205: **-0.04561**
- delta vs emb128 4-seed 0.76505: **-0.04861**
- corr_z vs emb128 4-seed: `0.7692`
- 50/50 z-blend accuracy: `0.75155`
- tier: **KILL_WEAK_SOLO** — solo 0.71644 <= 0.76000 kill gate

## Config

- emb_dim=128, epochs=80, batch_size=4096, lr=0.001, reg=0.0001
- bpr_weight=1.0, pointwise_weight=0.2, item_constraint_weight=0.05
- item_topk=10, seed=42, device=cuda:0

## Gate

Escalate only if the 50/50 z-blend beats emb128 4-seed by more than +0.0007 on this smoke, then rerun as a 3-split panel. Otherwise close the axis.

ULTRAGCN_CONSTRAINT_SMOKE_DONE
