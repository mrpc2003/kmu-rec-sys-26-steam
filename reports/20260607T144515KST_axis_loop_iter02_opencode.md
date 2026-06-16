# OpenCode axis-finding iteration 02 — 20260607T144515KST

## Verdict

**NO_SAFE_AXIS.** I did not launch a new probe. The remaining plausible local surfaces either repeat closed/stalled families, conflict with quarantine/public-negative transfer, or would be a weak one-split smoke that cannot satisfy Hermes' strict gate (`mean_delta >= +0.0015`, `min_delta >= 0`, `3/3` positive, fixes > breaks, pooled p < 0.05, no quarantine conflict).

## Safety flags

- validation_only: true
- candidate_csv_written: false
- full_test_candidate_or_submission_csv_created: false
- kaggle_submit_executed: false
- hidden_labels_used: false
- private_answers_used: false
- external_steam_scraping_used: false
- credentials_or_tokens_printed: false
- quarantine_or_guard_logic_weakened: false
- git_stage_commit_push_executed: false
- recursive_cron_scheduled: false

## Probe decision

No validation probe was launched. I checked the available script/report surface directly, without sub-agents:

- `scripts/lightgcn_hyperbolic.py` is a real local entrypoint, but geometry is already closed as redundant in `reports/din_set_encoder_closure_report.md` and prior advisory text explicitly mentions hyperbolic among closed priors.
- DIN/target-conditioned set prediction is closed in `reports/din_set_encoder_closure_report.md`: d64/d128 are below the emb128 4-seed reference and equal-blend is negative vs the true reference.
- TurboCF/MultiVAE/item-item linear/EASE/ALS-like shallow variants are already classified as redundant or public-negative.
- UserKNN and jackknife remain stalled/incomplete or weak, with warning-dominated logs and no strict metric report.
- Boundary/frontier/rankblend/ALS/BPR/TAG-CF/OTTO residual retunes conflict with quarantine/public-negative or independent strict-fail evidence.

Launching any of those now would be a repeat of a rejected family, not a materially new bounded design.

## Best apparent metric row

The strongest metric-bearing row remains the already rejected ALS diagnostic:

- variant: `diagnostic_atlas_als_f32_popa4_w0.20_band1_from_20260607T130533KST_current_best_als_independent_confirmation`
- mean Δ vs rankblend: `+0.0011335600453423744`
- min Δ: `+0.0004000800160031126`
- positive splits: `3/3`
- fixes/breaks: `462/394`
- pooled exact p: `0.021965674090633346`
- quarantine conflict: `true`

It is not actionable because it fails the strict mean-delta gate and is same-family/quarantine conflicted. The pre-registered ALS row failed more strongly: mean `+0.0008001600320064103`, min `-0.0003000600120024455`, positive `2/3`, p `0.12924401684163647`.

## Evidence checked

- Read `reports/20260607T144515KST_axis_loop_iter01_opencode.json` and Hermes rejection context.
- Read `reports/20260607T130533KST_current_best_als_independent_confirmation.json` for ALS strict/diagnostic rows.
- Read `state/aggressive_quota_runner_state.json` for quarantined/public-negative families.
- Read `reports/din_set_encoder_closure_report.md` for set-prediction, geometry, sequence, text/LM, latent, and item-item closure evidence.
- Ran broad `rg` searches across `scripts/`, `reports/`, `artifacts/`, and `state/` for probe/family/safety/quarantine terms.
- Ran `ast-grep` over `scripts/*.py` to inventory runnable Python entrypoints.
- Inspected active processes; only the expected OpenCode-Hermes controller/monitor and this OpenCode run were observed, not a Kaggle submit process.

## Required output files

- JSON: `reports/20260607T144515KST_axis_loop_iter02_opencode.json`
- Markdown: `reports/20260607T144515KST_axis_loop_iter02_opencode.md`
- Artifact directory: `artifacts/opencode_hermes_axis_loop_20260607T144515KST/`
