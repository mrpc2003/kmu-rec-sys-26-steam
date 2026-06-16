# 2026-06-06 11:36 KST — repeat-until-candidate loop status

## User request

우현 requested: "새 후보를 찾을때까지 반복".

## Actions taken

1. Created a durable hourly Hermes cron job to continue no-submit candidate discovery in this repo.
   - job id: `4d627b59804f`
   - name: `KMURecSys26 Steam no-submit candidate discovery loop`
   - schedule: every 1 hour
   - deliver: origin Discord thread
   - safety: no Kaggle submit, no hidden labels, no external Steam scraping, no weakening of quarantine guards.

2. Verified previous exact-K subset-loss full result exists and is already closed:
   - `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`
   - tier: `SUBSET_NO_GAIN_NOISE`
   - subset vs BPR fine-tune isolated delta: `+0.00000`
   - exact-K is not a candidate unless a deliberate new hyperparameter reason is introduced.

3. Launched immediate no-submit hours-confidence backbone probes on free GPUs.

## Running jobs

| mode | GPU | session | log path | artifact path |
|---|---:|---|---|---|
| `user_quantile` | 0 | `proc_624b0d88f188` | `logs/hours_confidence_user_quantile_20260606T113608KST.log` | `artifacts/hours_confidence/user_quantile/val_random_uniform_seed42/summary.json` |
| `item_quantile` | 1 | `proc_2b22f443ce97` | `logs/hours_confidence_item_quantile_20260606T113609KST.log` | `artifacts/hours_confidence/item_quantile/val_random_uniform_seed42/summary.json` |
| `balanced` | 2 | `proc_6666d3427f1b` | `logs/hours_confidence_balanced_20260606T113610KST.log` | `artifacts/hours_confidence/balanced/val_random_uniform_seed42/summary.json` |

All three use V100-compatible PyTorch:

```bash
uv run --python 3.13 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy \
  python scripts/hours_confidence_lightgcn_gate.py ...
```

## Candidate escalation rule for these jobs

- If any mode returns `CONF_GAIN_CHECK_ENSEMBLE`, escalate to seed/panel expansion.
- If all return `CONF_PLATEAU_NO_GAIN` or `CONF_REGRESS`, the cron loop will continue to the next orthogonal no-submit probe.
- No Kaggle submission is performed by these jobs.

## Current standing

No new candidate found yet; discovery loop and three no-submit GPU probes are running.
