# TAG-CF full-test candidate

- variant: `tagcf_fulltest_seed2024_sym_a0.1_raw_zblend_bw0.5`
- output: `submissions/candidate_tagcf_fulltest_seed2024_sym_a0p1_zblend.csv`
- safety: no hidden labels / no external scraping / no submit
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`

## Row diffs

- rankblend_public_best: `646`
- boundary_w_minus_0p75: `780`
- boundary_w2: `742`
- frontier_z_w1920_w64_minus_0p25: `510`
