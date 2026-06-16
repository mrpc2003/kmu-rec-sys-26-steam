# 2026-06-06 12:50:11 KST — repeat-until-candidate tick status

## Safety

- Kaggle submit executed: false
- Candidate/submission CSV written: false
- Hidden labels / external Steam scraping: false
- Quarantine/runner guards weakened: false

## Completed this tick

1. DNS pool=1 four-seed panel completed and aggregated: `reports/20260606T125011KST_dns_pool1_panel_aggregate.json` / `reports/20260606T125011KST_dns_pool1_panel_aggregate.md`.
   - decision: `DNS_POOL1_CLOSED_NO_CANDIDATE`; raw mean Δref `+0.00090`, z mean Δref `+0.00100`, 50/50 blend Δref `+0.00090`.
2. Queue-exhausted orthogonal smoke run: `reports/20260606T125011KST_multi_interest_svd_smoke.json` / `reports/20260606T125011KST_multi_interest_svd_smoke.md`.
   - decision: `NO_CANDIDATE`; best 50/50 blend Δref `-0.07141`.

## Previously-read required artifacts

- `state/aggressive_quota_runner_state.json`: aggressive runner enabled, public best snapshot 0.77825; quarantine guards preserved.
- `logs/latest_exactk_subset_outdir.txt`: points to stale/missing timestamp outdir, but canonical `artifacts/exactk_subset/val_random_uniform_seed42/summary.json` exists and is `SUBSET_NO_GAIN_NOISE`.
- hours-confidence summaries: user/item/balanced all `CONF_PLATEAU_NO_GAIN`.
- temporal compatibility: all tested reranks regress hard vs base.
- boundary covariate: residualized covariates below escalation bar / soft no-go.

## Active continuation jobs observed

A concurrent/background no-submit DNS multi-split panel is now running for `val_random_uniform_seed123` (not a Kaggle submission; score artifacts only):

| split | seed | GPU | log | expected summary |
|---|---:|---:|---|---|
| `val_random_uniform_seed123` | 42 | 0 | `logs/dns_pool1_val_random_uniform_seed123_seed42_20260606T124548KST.log` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed42/val_random_uniform_seed123/summary.json` |
| `val_random_uniform_seed123` | 123 | 1 | `logs/dns_pool1_val_random_uniform_seed123_seed123_20260606T124549KST.log` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed123/val_random_uniform_seed123/summary.json` |
| `val_random_uniform_seed123` | 2024 | 2 | `logs/dns_pool1_val_random_uniform_seed123_seed2024_20260606T124550KST.log` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed2024/val_random_uniform_seed123/summary.json` |
| `val_random_uniform_seed123` | 7 | 3 | `logs/dns_pool1_val_random_uniform_seed123_seed7_20260606T124720KST.log` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed7/val_random_uniform_seed123/summary.json` |

At 12:51 KST these logs had reached roughly epoch 40/200; no summaries were complete yet.

## Active process snapshot

```text
    PID    PPID STAT     ELAPSED %CPU %MEM COMMAND
   7613       1 Ssl     01:19:46  0.0  0.0 uv run --with pandas --with numpy --with scipy --with wandb python /opt/data/kaggle/kmu-rec-sys-26-steam/scripts/aggressive_quota_runner.py --sleep-no-quota 300 --sleep-no-candidate 600 --sleep-after-submit 21600
   7630    7613 Sl      01:19:45  0.1  0.0 /opt/data/home/.cache/uv/builds-v0/.tmpjFP1x1/bin/python /opt/data/kaggle/kmu-rec-sys-26-steam/scripts/aggressive_quota_runner.py --sleep-no-quota 300 --sleep-no-candidate 600 --sleep-after-submit 21600
  13789    6284 Ss         04:29  0.0  0.0 /usr/bin/bash -lic set +m; set -euo pipefail SPLIT=val_random_uniform_seed123 SEED=42 POOL=1 GPU=0 TS=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST) OUT="artifacts/dns_pool1_multisplit/${SPLIT}/seed${SEED}" mkdir -p logs LOG="logs/dns_pool${POOL}_${SPLIT}_seed${SEED}_${TS}.log" printf '%s\n' "$LOG" > "logs/latest_dns_pool${POOL}_${SPLIT}_seed${SEED}.logpath" exec > >(tee -a "$LOG") 2>&1 printf '[%s] launching DNS pool=%s split=%s seed=%s gpu=%s out=%s\n' "$TS" "$POOL" "$SPLIT" "$SEED" "$GPU" "$OUT" UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=$GPU uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split "$SPLIT" --dns-pool "$POOL" --seed "$SEED" --device cuda:0 --epochs 200 --out-dir "$OUT" printf '[%s] done DNS pool=%s split=%s seed=%s\n' "$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST)" "$POOL" "$SPLIT" "$SEED"
  14109   13789 Sl         04:29  0.1  0.0 uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with torch==2.10.0+cu128 --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 42 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed42
  14121    6284 Ss         04:28  0.0  0.0 /usr/bin/bash -lic set +m; set -euo pipefail SPLIT=val_random_uniform_seed123 SEED=123 POOL=1 GPU=1 TS=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST) OUT="artifacts/dns_pool1_multisplit/${SPLIT}/seed${SEED}" mkdir -p logs LOG="logs/dns_pool${POOL}_${SPLIT}_seed${SEED}_${TS}.log" printf '%s\n' "$LOG" > "logs/latest_dns_pool${POOL}_${SPLIT}_seed${SEED}.logpath" exec > >(tee -a "$LOG") 2>&1 printf '[%s] launching DNS pool=%s split=%s seed=%s gpu=%s out=%s\n' "$TS" "$POOL" "$SPLIT" "$SEED" "$GPU" "$OUT" UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=$GPU uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split "$SPLIT" --dns-pool "$POOL" --seed "$SEED" --device cuda:0 --epochs 200 --out-dir "$OUT" printf '[%s] done DNS pool=%s split=%s seed=%s\n' "$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST)" "$POOL" "$SPLIT" "$SEED"
  14441   14121 Sl         04:28  0.1  0.0 uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with torch==2.10.0+cu128 --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 123 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed123
  14453    6284 Ss         04:27  0.0  0.0 /usr/bin/bash -lic set +m; set -euo pipefail SPLIT=val_random_uniform_seed123 SEED=2024 POOL=1 GPU=2 TS=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST) OUT="artifacts/dns_pool1_multisplit/${SPLIT}/seed${SEED}" mkdir -p logs LOG="logs/dns_pool${POOL}_${SPLIT}_seed${SEED}_${TS}.log" printf '%s\n' "$LOG" > "logs/latest_dns_pool${POOL}_${SPLIT}_seed${SEED}.logpath" exec > >(tee -a "$LOG") 2>&1 printf '[%s] launching DNS pool=%s split=%s seed=%s gpu=%s out=%s\n' "$TS" "$POOL" "$SPLIT" "$SEED" "$GPU" "$OUT" UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=$GPU uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split "$SPLIT" --dns-pool "$POOL" --seed "$SEED" --device cuda:0 --epochs 200 --out-dir "$OUT" printf '[%s] done DNS pool=%s split=%s seed=%s\n' "$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST)" "$POOL" "$SPLIT" "$SEED"
  14774   14453 Sl         04:27  0.1  0.0 uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with torch==2.10.0+cu128 --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 2024 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed2024
  14786   14109 Rl         04:26  102  0.6 /opt/data/home/.cache/uv/builds-v0/.tmp5ojEAe/bin/python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 42 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed42
  14842   14441 Rl         04:25  102  0.6 /opt/data/home/.cache/uv/builds-v0/.tmpnGBkmf/bin/python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 123 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed123
  14898   14774 Rl         04:24  102  0.6 /opt/data/home/.cache/uv/builds-v0/.tmpw5wPZe/bin/python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 2024 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed2024
  15178    6284 Ss         02:58  0.0  0.0 /usr/bin/bash -lic set +m; set -euo pipefail SPLIT=val_random_uniform_seed123 SEED=7 POOL=1 GPU=3 TS=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST) OUT="artifacts/dns_pool1_multisplit/${SPLIT}/seed${SEED}" mkdir -p logs LOG="logs/dns_pool${POOL}_${SPLIT}_seed${SEED}_${TS}.log" printf '%s\n' "$LOG" > "logs/latest_dns_pool${POOL}_${SPLIT}_seed${SEED}.logpath" exec > >(tee -a "$LOG") 2>&1 printf '[%s] launching DNS pool=%s split=%s seed=%s gpu=%s out=%s\n' "$TS" "$POOL" "$SPLIT" "$SEED" "$GPU" "$OUT" printf '[%s] note gpu3 has prior stale/not-found allocation but large memory headroom; validation-only job uses small memory\n' "$TS" UV_LINK_MODE=copy CUDA_VISIBLE_DEVICES=$GPU uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split "$SPLIT" --dns-pool "$POOL" --seed "$SEED" --device cuda:0 --epochs 200 --out-dir "$OUT" printf '[%s] done DNS pool=%s split=%s seed=%s\n' "$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST)" "$POOL" "$SPLIT" "$SEED"
  15499   15178 Sl         02:57  0.2  0.0 uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with torch==2.10.0+cu128 --with numpy --with pandas --with scipy python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 7 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed7
  15511   15499 Rl         02:54  103  0.6 /opt/data/home/.cache/uv/builds-v0/.tmpyQzFIY/bin/python scripts/lightgcn_dns.py --split val_random_uniform_seed123 --dns-pool 1 --seed 7 --device cuda:0 --epochs 200 --out-dir artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed7
```

## GPU snapshot

```text
0, Tesla V100-PCIE-32GB, 476, 32768, 8, 2
1, Tesla V100-PCIE-32GB, 476, 32768, 7, 2
2, Tesla V100-PCIE-32GB, 476, 32768, 8, 2
3, Tesla V100-PCIE-32GB, 4792, 32768, 7, 2
```

## Candidate status

No strict or weak validation candidate found in this tick. DNS was closed; multi-interest smoke did not gate. No Kaggle submission was run.
