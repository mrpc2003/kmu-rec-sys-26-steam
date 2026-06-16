# bivae_recvae_userrow_seed42_smoke

- validation_only: true
- no_kaggle_submit: true
- candidate_csv_written: false
- full_test_candidate_materialized: false
- split: `val_random_uniform_seed42`

## verdict

`KILL_WEAK_SOLO`

RecVAE-style user-row solo accuracy missed the 0.735 smoke floor.

## metrics

| metric | value | gate |
|---|---:|---:|
| solo accuracy | 0.706041 | >= 0.735000 |
| base emb128 4-seed accuracy | 0.765053 | ref 0.765053 |
| 50/50 z-blend accuracy | 0.745949 | >= 0.765753 |
| blend delta vs base | -0.019104 | > +0.000700 |
| corr_z vs base | 0.698117 | <= 0.950000 |
| rank corr vs base | 0.661491 | diagnostic |
| fixes / breaks | 1568 / 2748 | fixes > breaks |

## item-degree bucket audit

| item degree decile | changed rows | fixes | breaks | net |
|---:|---:|---:|---:|---:|
| 0 | 202 | 69 | 133 | -64 |
| 1 | 234 | 63 | 171 | -108 |
| 2 | 278 | 85 | 193 | -108 |
| 3 | 305 | 132 | 173 | -41 |
| 4 | 382 | 125 | 257 | -132 |
| 5 | 380 | 139 | 241 | -102 |
| 6 | 507 | 198 | 309 | -111 |
| 7 | 626 | 248 | 378 | -130 |
| 8 | 726 | 298 | 428 | -130 |
| 9 | 676 | 211 | 465 | -254 |

## outputs

- `artifacts/bivae_recvae_userrow_seed42_smoke/val_random_uniform_seed42/recvae_userrow_validation_scores.csv`
- `reports/20260616T_bivae_recvae_userrow_seed42_smoke.json`
- `reports/20260616T_bivae_recvae_userrow_seed42_smoke.md`

BIVAE_RECVAE_USERROW_SEED42_SMOKE_DONE
