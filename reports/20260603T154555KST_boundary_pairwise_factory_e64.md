# Boundary pairwise candidate factory

- safety: no Kaggle submit; no hidden labels; no Steam scraping; public test score files only for materialization.
- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | changed | manual-risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `boundary_pairwise_vote2_B4_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |
| 2 | `boundary_pairwise_vote2_B8_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |
| 3 | `boundary_pairwise_vote2_B16_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |
| 4 | `boundary_pairwise_vote2_B4_tau0_guard0_cap2` | +0.000800 | 2/3 | 498/450 | 0.1268 | 948 | True |
| 5 | `boundary_pairwise_vote2_B8_tau0_guard0_cap2` | +0.000800 | 2/3 | 498/450 | 0.1268 | 948 | True |
| 6 | `boundary_pairwise_vote2_B16_tau0_guard0_cap2` | +0.000800 | 2/3 | 498/450 | 0.1268 | 948 | True |
| 7 | `boundary_pairwise_vote192_B4_tau0_guard0.1_cap2` | +0.000467 | 3/3 | 197/169 | 0.1581 | 366 | True |
| 8 | `boundary_pairwise_vote192_B8_tau0_guard0.1_cap2` | +0.000467 | 3/3 | 197/169 | 0.1581 | 366 | True |
| 9 | `boundary_pairwise_vote192_B16_tau0_guard0.1_cap2` | +0.000467 | 3/3 | 197/169 | 0.1581 | 366 | True |
| 10 | `boundary_pairwise_vote192_B4_tau0_guard0.1_cap1` | +0.000433 | 3/3 | 194/168 | 0.1888 | 362 | True |

## Materialized

- file: `submissions/candidate_autorun_boundary_pairwise_probe_latest.csv`
- sha256: `7539c1160abe5ccb836381488dc36e94570b0229d193b8d5dc6926e4a98b4237`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
