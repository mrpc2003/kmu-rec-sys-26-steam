# OpenCode axis-finding iteration 02 — 20260607T074546KST

## Verdict

`NO_SAFE_AXIS` — I did not launch a new probe. The only fresh-looking validation surfaces found in this iteration were already completed and rejected/weak, while the remaining candidates repeat closed, stalled, quarantined, public-negative, or underpowered families.

## Safety flags

- validation_only: true
- candidate_csv_written: false
- full_test_candidate_or_submission_csv_created: false
- kaggle_submit_executed: false
- hidden_labels_used / private_answers_used: false
- external_steam_scraping_used: false
- credentials_or_tokens_printed: false
- quarantine_or_guard_logic_weakened: false
- git_stage_commit_push_executed: false
- recursive_cron_scheduled: false

## Evidence checked

- Recent Hermes/OpenCode loop state: `reports/20260607T074528KST_opencode_hermes_axis_rejection_loop_summary.json`, `reports/20260607T074546KST_axis_loop_iter01_opencode.json`, `reports/20260607T063604KST_opencode_improvement_axis_loop.json`.
- Safety/quarantine state: `state/aggressive_quota_runner_state.json`, `state/autonomous_submission_policy.json`.
- Completed/rejected probe surfaces: layer-mixture summaries under `artifacts/layermix_probe/emb128_L4_r3_seed{42,7,123}/summary.json`, `reports/20260601_slk_lite_panel_aggregate.json`, `reports/20260606T125011KST_multi_interest_svd_smoke.json`.
- Direct searches: `rg` over reports/scripts/state for strict gates, quarantine/public-negative families, probe outputs, layermix/multi-interest/UserKNN/jackknife evidence; `ast-grep` for Python `subprocess.run` call sites; directory listing of the requested artifact root.

## Closed/rejected axis notes

| Axis | Decision reason |
|---|---|
| UserKNN gated residual fine-grid | Stalled/incomplete; expected reports absent; repeated invalid-divide warnings; prior smoke below mean and p gates. |
| Jackknife uncertainty boundary | Smoke weak: mean Δ `+0.0003667`, min Δ `-0.0012002`, `2/3` positive, p `0.3388`; expanded run incomplete. |
| DNS/capacity/frontier/emb192/rankblend/boundary/TAG-CF residuals | Closed or quarantined by public-negative/non-improving transfer in runner state. |
| LightGCN++ layer mixture | Already completed: seed42 best nonbase `+0.001700` but p `0.1065`; seed7 `+0.000400`; seed123 `+0.000300`; no 3/3 significant strict pass. |
| SL@K-lite | Completed reject: epoch-1 mean Δ `-0.009035`, epoch-2 mean Δ `-0.021671`, direction gate `0/3`. |
| Multi-interest SVD smoke | Completed reject: best blend50 Δ `-0.071414` versus emb128 reference. |
| Semantic/text/README/LM, temporal, hours-confidence, exact-K, last-slot | Closed by recent reports or prompt rejection list as weak, redundant, split-negative, or public-noise risk. |

## New probe

None launched. Starting a probe now would either duplicate a closed/stalled family or create an underpowered one-split diagnostic that cannot credibly pass Hermes' strict gate.

## Output files

- JSON: `reports/20260607T074546KST_axis_loop_iter02_opencode.json`
- Markdown: `reports/20260607T074546KST_axis_loop_iter02_opencode.md`
