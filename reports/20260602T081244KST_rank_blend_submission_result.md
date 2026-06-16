# Rank-blend emb128⊕emb192 forced submission result
- file: `/opt/data/kaggle/kmu-rec-sys-26-steam/submissions/candidate_rank_blend_emb128_emb192.csv`
- sha256: `835b8b90ce30116a3df7a7575e6ccdaec268af9c1acb01ca0c15c733b3152b2e`
- status: **SubmissionStatus.COMPLETE**
- public: **0.77825**
- previous public best `candidate_emb128_emb64_zblend.csv`: 0.77815
- Δ vs previous public best: **+0.00010**
- Δ vs emb128 0.77745: **+0.00080**

## Validation evidence before submit
- 3-split uniform mean Δ vs emb128: +0.00083
- split Δ: [0.0017, 0.0003, 0.0005]
- McNemar p by split: [0.0727, 0.7785, 0.5972]; Fisher p=0.3421
- strict gate: False

## Diff vs previous public-best zblend
- flips: 596 (2.98%)
- promoted old=0→new=1: 298
- demoted old=1→new=0: 298

### User candidate-count bucket
| bucket | rows | flips | flip_frac |
|---|---:|---:|---:|
| 11+ | 2550 | 68 | 2.67% |
| 2 | 4282 | 122 | 2.85% |
| 3-4 | 5228 | 128 | 2.45% |
| 5-6 | 3990 | 136 | 3.41% |
| 7-10 | 3948 | 142 | 3.60% |

### Item popularity quintile
| quintile | rows | flips | flip_frac | mean_item_pop |
|---|---:|---:|---:|---:|
| Q1 lowest | 4000 | 96 | 2.40% | 18.78 |
| Q2 | 3999 | 170 | 4.25% | 33.39 |
| Q3 | 4000 | 176 | 4.40% | 64.21 |
| Q4 | 3999 | 115 | 2.88% | 138.46 |
| Q5 highest | 4000 | 39 | 0.97% | 392.41 |

## Interpretation
Forced/manual-risk quota-burn candidate. Public improved slightly, but validation McNemar gate remained non-significant; treat as public-LB information/possible hedge, not proof of private superiority.
