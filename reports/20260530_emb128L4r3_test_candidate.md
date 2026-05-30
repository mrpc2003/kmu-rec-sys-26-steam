# emb128_L4_reg1e-3 4-seed ensemble — test candidate

- file: `/opt/data/kaggle/kmu-rec-sys-26-steam/artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv`
- sha256: `7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195`
- rows=19998 label_1=9999 label_0=9999 bad_users=0
- ids_contiguous=True labels_binary=True
- rowdiff vs submitted emb64 ensemble (public 0.77125): 924 (4.62%)

## Gate

uniform (public surrogate): emb128 ensemble **0.76505** vs emb64 ensemble 0.76145 (+0.0036, > single-seed noise 0.0007). Projected public ~0.776 via transfer ratio 1.26 (extrapolated from emb64; direction solid, magnitude indicative).
