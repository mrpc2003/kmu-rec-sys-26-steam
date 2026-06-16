# 2026-06-06 12:50:11 KST — multi-interest SVD64 prototype smoke

## Safety

- Kaggle submit executed: false
- Candidate/submission CSV written: false
- Hidden labels / external Steam scraping: false

## Why this probe

The fixed queue was exhausted after hours-confidence/exact-K/temporal/boundary/DNS. This bounded CPU smoke tests a different multi-interest routing hypothesis: a user may have multiple recency-separated interest prototypes rather than one LightGCN user vector.

## Results

| score | solo acc | solo Δref | corr_z vs ref | 50/50 zblend | blend Δref | decision |
|---|---:|---:|---:|---:|---:|---|
| `mi_svd64_mean_interest` | 0.55881 | -0.20624 | 0.1671 | 0.69364 | -0.07141 | `REJECT_WEAK_SOLO_BELOW_FLOOR` |
| `mi_svd64_recency2_max_interest` | 0.54631 | -0.21874 | 0.1239 | 0.68904 | -0.07602 | `REJECT_WEAK_SOLO_BELOW_FLOOR` |
| `mi_svd64_recency3_max_interest` | 0.54291 | -0.22214 | 0.1060 | 0.69004 | -0.07502 | `REJECT_WEAK_SOLO_BELOW_FLOOR` |

## Verdict

`NO_CANDIDATE` — best blend delta `-0.07141`; no 3-split escalation and no submission.
