# OpenCode axis-finding iteration 01 — 20260607T144515KST

## Verdict

**NO_SAFE_AXIS.** I did not launch a validation probe. Direct inspection found no fresh independent bounded axis that avoids the closed/stalled/weak/quarantined families and has a credible path to Hermes strict gate versus current public best `candidate_rank_blend_emb128_emb192.csv` (`0.77825`).

## Safety flags

| flag | value |
|---|---:|
| validation_only | `true` |
| candidate_csv_written | `false` |
| full_test_candidate_or_submission_csv_created | `false` |
| kaggle_submit_executed | `false` |
| hidden_labels_used | `false` |
| private_answers_used | `false` |
| external_steam_scraping_used | `false` |
| credentials_or_tokens_printed | `false` |
| quarantine_or_guard_logic_weakened | `false` |
| git_stage_commit_push_executed | `false` |
| recursive_cron_scheduled | `false` |

## What I did

- Obeyed the critical no-delegation rule despite `[search-mode]`; no sub-agents were launched.
- Searched scripts, reports, and state directly with `rg`, checked runnable Python surfaces with AST-grep, and inspected recent metric reports plus quarantine/guard state.
- Did not run `kaggle competitions submit`, did not write under `submissions/`, did not create a candidate/full-test/uploadable CSV, and did not stage/commit/push.

## Axis decision

No probe launched.

The strongest remaining metric-bearing row is a **diagnostic** same-family ALS residual/boundary row from `reports/20260607T130533KST_current_best_als_independent_confirmation.json`: mean delta `+0.0011335600453423744`, min delta `+0.0004000800160031126`, positive splits `3/3`, fixes/breaks `462/394`, pooled exact p `0.021965674090633346`. It is not a candidate because it is below the strict `+0.0015` mean-delta gate and is not the pre-registered row. The pre-registered row failed with mean `+0.0008001600320064103`, min `-0.0003000600120024455`, positive splits `2/3`, fixes/breaks `504/456`, p `0.12924401684163647`.

## Closed/rejected surfaces reconfirmed

- OTTO/source-separated co-visitation: independent mean +0.0006668, min -0.0006001, 2/3, p=0.1700; forced public 0.77815 below 0.77825.
- ALS/rankblend residual: pre-registered fresh-panel row failed mean/min/3-of-3/p gates; diagnostic row below +0.0015 and same-family only.
- UserKNN: previous smoke below strict effect-size gate; fine-grid incomplete/stalled with invalid-divide warnings.
- Jackknife uncertainty: weak/split-negative smoke and expanded run incomplete.
- Boundary/frontier/capacity/TAG-CF: public-negative or quarantined/near-duplicate risk.
- Semantic/text/LM, multi-interest SVD, SASRec/DIN, SGL/XSimGCL/DirectAU, MultiVAE/AlphaRec/TurboCF: prior reports show weak, redundant, closed, public-negative, or too large for a bounded credible tick.

## Ranked next-axis hypotheses

| rank | axis | status | why not launched now |
|---:|---|---|---|
| 1 | Genuinely new validation-label-free base model family not already implemented locally | not_launched | No local script/report surface was both fresh and credible within <=3600s. Starting a new family from scratch in this iteration would likely be underpowered or exceed the bounded validation-only scope. |
| 2 | Finite-value/debug-only UserKNN cleanup | not_launched | This would repair a stalled family rather than introduce materially new independent information; broad UserKNN fine-grid is explicitly stalled with warning-dominated incomplete logs. |
| 3 | New non-boundary disagreement diagnostic independent of rankblend/capacity/frontier/jackknife | not_launched | Direct rg plus AST inventory found available disagreement/margin probes collapse into closed boundary, frontier/capacity, rankblend residual, jackknife, layer-mix, or public-negative families. |

## Outputs

- JSON report: `reports/20260607T144515KST_axis_loop_iter01_opencode.json`
- Markdown report: `reports/20260607T144515KST_axis_loop_iter01_opencode.md`
- Validation artifacts: none created in this iteration.
