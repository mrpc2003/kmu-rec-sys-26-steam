# Boundary pairwise candidate factory

- safety: no Kaggle submit; no hidden labels; no Steam scraping; public test score files only for materialization.
- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | changed | manual-risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `boundary_pairwise_vote192_B4_tau0_guard0.1_cap2` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 2 | `boundary_pairwise_vote192_B4_tau0_guard0.1_cap3` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 3 | `boundary_pairwise_vote192_B8_tau0_guard0.1_cap2` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 4 | `boundary_pairwise_vote192_B8_tau0_guard0.1_cap3` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 5 | `boundary_pairwise_vote192_B16_tau0_guard0.1_cap2` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 6 | `boundary_pairwise_vote192_B16_tau0_guard0.1_cap3` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 7 | `boundary_pairwise_vote192_B24_tau0_guard0.1_cap2` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 8 | `boundary_pairwise_vote192_B24_tau0_guard0.1_cap3` | +0.000433 | 2/3 | 196/170 | 0.1912 | 366 | True |
| 9 | `boundary_pairwise_vote192_B4_tau0_guard0.1_cap1` | +0.000433 | 2/3 | 194/168 | 0.1888 | 362 | True |
| 10 | `boundary_pairwise_vote192_B8_tau0_guard0.1_cap1` | +0.000433 | 2/3 | 194/168 | 0.1888 | 362 | True |

## Materialized

- file: `artifacts/aggressive_boundary_pairwise/manual_qa_candidate.csv`
- sha256: `964c17634089ac83602c0917d2efff23f36ffbc5359cf0296d14397467e87f4b`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
