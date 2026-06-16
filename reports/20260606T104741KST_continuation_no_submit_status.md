# 2026-06-06 Continuation status — no-submit exploration

## Safety

- Kaggle submit executed in this continuation: **false**
- Candidate submission CSV written in this continuation: **false**
- Hidden labels / external Steam scraping: **false**
- Existing autonomous runner process was observed alive; no manual submission command was issued here.

## Current public submission snapshot

Read-only Kaggle submissions check confirms current best visible public score remains:

- `candidate_rank_blend_emb128_emb192.csv` — public `0.77825` (2026-06-01)

Recent autonomous submissions after that did not beat it:

- rankblend/ALS residual family: public `0.77785`–`0.77805`
- boundary score/blend/frontier family: public `0.77575`–`0.77755`
- TAG-CF full-test seed2024 zblend: public `0.77615`

## Existing runner/process state

- `scripts/aggressive_quota_runner.py` is still running externally under PID 231/249.
- Runner state has `autonomous_submission_enabled=true`, `manual_gate_required=false`, and quarantine guards for the recent negative-transfer families.
- GPU state before new launch: GPU0/1/2 free; GPU3 had an orphaned ~4.3GB allocation from a non-live PID.

## New no-submit checks run now

### 1. Deep data-signature EDA

Command:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy \
  python scripts/deep_data_signature_eda.py
```

Output:

- `reports/20260601_deep_data_signature_eda.json`
- `reports/20260601_deep_data_signature_eda.md`

Key result: candidate marginal is best explained by random-positive + uniform-negative mixture, so uniform split remains the primary public surrogate. This supports continuing K-aware / boundary-only objectives rather than reverting to sqrt-pop/pop-biased objectives.

### 2. Temporal compatibility gate

First run without scipy failed because `recsys_played_utils.py` imports `scipy.sparse`; rerun with scipy succeeded.

Command:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy \
  python scripts/temporal_compatibility_gate.py --beta 10.0
```

Artifacts:

- `artifacts/temporal_compat/val_random_uniform_seed42/summary.json`
- `artifacts/temporal_compat/val_random_uniform_seed42/temporal_scores.csv`

Result versus base `0.76505`:

- `T_only`: `0.51760`, delta `-0.24745`
- `rank_sum`: `0.65863`, delta `-0.10642`
- `rank_sum_resid`: `0.67243`, delta `-0.09262`
- `boundary_swap`: `0.66803`, delta `-0.09702`

Verdict: **reject temporal reranking axis**.

### 3. Boundary covariate expansion

Command:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy \
  python scripts/boundary_covariate_expansion.py
```

Artifacts:

- `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json`
- `artifacts/boundary_covariate/val_random_uniform_seed42/boundary_pairs_covariates.csv`

Boundary result:

- `d_cooc_raw` AUC `0.6740` — strong but popularity-confounded
- `d_logpop` AUC `0.6775` — confirms the trap
- `d_cooc_resid` AUC `0.5341`, CI excludes 0.5 but below escalation bar
- `d_knn_resid` AUC `0.5037`, chance-level

Verdict: **intermediate/soft no-go**; no candidate promotion.

### 4. Exact-K subset-loss smoke test

Torch compatibility issue found and fixed before launching full run:

- plain `uv --with torch` resolved to `torch 2.12.0+cu130`, which detects CUDA but fails on V100/sm_70 with `no kernel image is available`.
- verified working command uses `torch==2.10.0+cu128` from PyTorch cu128 index.

Smoke command:

```bash
UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=0 uv run --python 3.13 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy \
  python scripts/lightgcn_exactk_subset_loss.py --device cuda:0 \
  --pretrain-epochs 2 --ft-epochs 1 --steps-per-epoch 1 \
  --batch-users 64 --batch-size 4096 \
  --out-dir artifacts/exactk_subset_smoke
```

Smoke result:

- ran successfully on GPU0
- wrote `artifacts/exactk_subset_smoke/val_random_uniform_seed42/summary.json`
- smoke tier: `SUBSET_NO_GAIN_NOISE` (expected because this was only a 2-epoch functionality check)

## Full run launched

Background process:

- Hermes process session: `proc_5f51172f310d`
- shell PID: `2607`
- Python child PID observed: `2938`
- output dir: `artifacts/exactk_subset_20260606T104621KST`
- GPU: `CUDA_VISIBLE_DEVICES=0`

Command:

```bash
UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=0 uv run --python 3.13 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy \
  python scripts/lightgcn_exactk_subset_loss.py --device cuda:0 \
  --pretrain-epochs 200 --ft-epochs 40 --steps-per-epoch 40 \
  --batch-users 2048 --batch-size 4096 \
  --out-dir artifacts/exactk_subset_20260606T104621KST
```

This is validation-only and writes `summary.json` plus validation score diagnostics, not a Kaggle submission file.

## Next action when the background process completes

1. Read `artifacts/exactk_subset_20260606T104621KST/val_random_uniform_seed42/summary.json`.
2. If tier is not positive, mark exact-K subset loss as rejected/closed.
3. If it shows a real isolated gain versus BPR fine-tune, extend to a 3-uniform-split panel before any materialization/submission decision.
4. Do not submit anything from this continuation without the runner's established guards and a fresh result review.
