# bm25_lmf_seed42_smoke

- validation_only: true
- no_kaggle_submit: true
- candidate_csv_written: false
- full_test_candidate_materialized: false
- split: `val_random_uniform_seed42`

## verdict

`KILL_WEAK_SOLO`

BM25+LMF solo accuracy missed the 0.735 floor, so this one-shot is underpowered despite competition precedent.

## metrics

| metric | value | gate |
|---|---:|---:|
| solo accuracy | 0.670134 | >= 0.735000 |
| base emb128 4-seed accuracy | 0.765053 | ref 0.765053 |
| 50/50 z-blend accuracy | 0.735047 | >= 0.765753 |
| blend delta vs base | -0.030006 | > +0.000700 |
| corr_z vs base | 0.587728 | <= 0.950000 |
| rank corr vs base | 0.543255 | diagnostic |
| fixes / breaks | 1691 / 3589 | fixes > breaks |
| changed-row failed-union overlap | 0.069886 | low |

## item-degree bucket audit

| item degree decile | changed rows | fixes | breaks | net |
|---:|---:|---:|---:|---:|
| 0 | 227 | 66 | 161 | -95 |
| 1 | 262 | 75 | 187 | -112 |
| 2 | 290 | 78 | 212 | -134 |
| 3 | 367 | 143 | 224 | -81 |
| 4 | 433 | 126 | 307 | -181 |
| 5 | 517 | 166 | 351 | -185 |
| 6 | 614 | 211 | 403 | -192 |
| 7 | 754 | 262 | 492 | -230 |
| 8 | 804 | 286 | 518 | -232 |
| 9 | 1012 | 278 | 734 | -456 |

## outputs

- `artifacts/bm25_lmf_seed42_smoke/val_random_uniform_seed42/bm25_lmf_validation_scores.csv`
- `reports/20260616T_bm25_lmf_seed42_smoke.json`
- `reports/20260616T_bm25_lmf_seed42_smoke.md`

BM25_LMF_SEED42_SMOKE_DONE
