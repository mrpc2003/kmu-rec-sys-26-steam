# Jackknife uncertainty boundary probe

Safety: validation-only; no Kaggle submit; no candidate/submission CSV; no hidden labels; no external scraping.

- verdict: **WEAK_SIGNAL**
- strict pass count: `0`
- top row-level score artifacts: `3` files

| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `vote_consensus__high_capacity_gap__B1__w0.1` | +0.000367 | -0.001200 | 2/3 | 252 | 230 | 0.3388 |
| 2 | `vote_consensus__all_boundary__B1__w0.1` | +0.000333 | -0.001100 | 2/3 | 293 | 273 | 0.4245 |
| 3 | `vote_consensus__low_vote_abs__B1__w0.1` | +0.000333 | -0.001100 | 2/3 | 293 | 273 | 0.4245 |
| 4 | `vote_consensus__high_capacity_gap__B1__w0.05` | +0.000300 | -0.001300 | 2/3 | 245 | 227 | 0.434 |
| 5 | `vote_consensus__all_boundary__B1__w0.05` | +0.000267 | -0.001200 | 2/3 | 283 | 267 | 0.5225 |
| 6 | `vote_consensus__low_vote_abs__B1__w0.05` | +0.000267 | -0.001200 | 2/3 | 283 | 267 | 0.5225 |
| 7 | `ucb_all__high_capacity_gap__B1__w0.05` | +0.000200 | -0.001100 | 1/3 | 340 | 328 | 0.6704 |
| 8 | `ucb_all__high_capacity_gap__B1__w0.1` | +0.000200 | -0.001100 | 1/3 | 340 | 328 | 0.6704 |
| 9 | `ucb_all__all_boundary__B1__w0.05` | +0.000167 | -0.001300 | 2/3 | 402 | 392 | 0.7494 |
| 10 | `ucb_all__all_boundary__B1__w0.1` | +0.000167 | -0.001300 | 2/3 | 402 | 392 | 0.7494 |
| 11 | `ucb_all__low_vote_abs__B1__w0.05` | +0.000167 | -0.001300 | 2/3 | 402 | 392 | 0.7494 |
| 12 | `ucb_all__low_vote_abs__B1__w0.1` | +0.000167 | -0.001300 | 2/3 | 402 | 392 | 0.7494 |
| 13 | `vote_consensus__high_seed_std__B1__w0.1` | +0.000033 | -0.001900 | 2/3 | 234 | 232 | 0.9631 |
| 14 | `vote_consensus__high_seed_std__B1__w0.05` | -0.000033 | -0.002000 | 2/3 | 227 | 229 | 0.9627 |
| 15 | `lcb_all__high_capacity_gap__B1__w0.05` | -0.000100 | -0.001600 | 1/3 | 320 | 326 | 0.8441 |
| 16 | `lcb_all__high_capacity_gap__B1__w0.1` | -0.000100 | -0.001600 | 1/3 | 320 | 326 | 0.8441 |
| 17 | `capacity_agree_128__high_capacity_gap__B1__w0.05` | -0.000133 | -0.000800 | 1/3 | 345 | 353 | 0.7911 |
| 18 | `capacity_agree_128__high_capacity_gap__B1__w0.1` | -0.000133 | -0.000800 | 1/3 | 345 | 353 | 0.7911 |
| 19 | `ucb_all__high_seed_std__B1__w0.05` | -0.000300 | -0.002100 | 2/3 | 323 | 341 | 0.5095 |
| 20 | `ucb_all__high_seed_std__B1__w0.1` | -0.000300 | -0.002100 | 2/3 | 323 | 341 | 0.5095 |
| 21 | `std_demotion__high_capacity_gap__B1__w0.05` | -0.000400 | -0.000800 | 1/3 | 298 | 322 | 0.3557 |
| 22 | `std_demotion__high_capacity_gap__B1__w0.1` | -0.000400 | -0.000800 | 1/3 | 298 | 322 | 0.3557 |
| 23 | `lcb_all__all_boundary__B1__w0.05` | -0.000467 | -0.001600 | 1/3 | 355 | 383 | 0.3203 |
| 24 | `lcb_all__all_boundary__B1__w0.1` | -0.000467 | -0.001600 | 1/3 | 355 | 383 | 0.3203 |
| 25 | `lcb_all__low_vote_abs__B1__w0.05` | -0.000467 | -0.001600 | 1/3 | 355 | 383 | 0.3203 |
| 26 | `lcb_all__low_vote_abs__B1__w0.1` | -0.000467 | -0.001600 | 1/3 | 355 | 383 | 0.3203 |
| 27 | `std_demotion__high_seed_std__B1__w0.05` | -0.000533 | -0.001600 | 1/3 | 278 | 310 | 0.2011 |
| 28 | `std_demotion__high_seed_std__B1__w0.1` | -0.000533 | -0.001600 | 1/3 | 278 | 310 | 0.2011 |
| 29 | `std_demotion__all_boundary__B1__w0.05` | -0.000600 | -0.001000 | 0/3 | 337 | 373 | 0.189 |
| 30 | `std_demotion__all_boundary__B1__w0.1` | -0.000600 | -0.001000 | 0/3 | 337 | 373 | 0.189 |