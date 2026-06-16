# KMURecSys26 Steam — OpenCode axis-finding iteration 02

You are OpenCode acting as a constrained validation-only worker inside `/opt/data/kaggle/kmu-rec-sys-26-steam`.

CRITICAL EXECUTION RULES:
- Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- Your job: find a fresh independent improvement axis OR launch exactly one bounded validation-only probe.
- Do NOT run `kaggle competitions submit`.
- Do NOT create full-test candidate/submission/uploadable CSVs. Do NOT write under `submissions/`.
- Do NOT use hidden labels/private answers. Do NOT scrape external Steam data.
- Do NOT print credentials/tokens/secrets.
- Do NOT weaken quarantine/guard logic.
- Do NOT git add/stage/commit/push.
- Do NOT schedule cron jobs.
- Validation-only scripts, split artifacts, logs, and reports are allowed.
- If launching a probe, it must be bounded with explicit timeout <= 3600 seconds and report/log paths. Prefer <= 20 minutes.

Objective:
- Find a real axis beyond current public-best behavior.
- Current public best reference: `candidate_rank_blend_emb128_emb192.csv`, public `0.77825`.
- Main internal uniform reference: emb128 4-seed, around `0.76505`.
- Weak one-split or tiny `+0.000x` blips are NOT candidates; they require stronger multi-split evidence.

Strict gate Hermes will apply:
- validation_only=true
- candidate/full-test/submission CSV written=false
- kaggle_submit_executed=false
- mean_delta >= +0.0015
- min_delta >= 0
- 3/3 positive validation splits
- fixes > breaks
- pooled exact/McNemar p < 0.05 when available
- no quarantine/near-duplicate/public-negative-family conflict

Closed/stalled families to avoid unless you introduce a materially new bounded design:
- OTTO/source-separated co-visitation family already forced-tested after user approval: independent strict failed (`mean_delta +0.0006668`, `min_delta -0.0006001`, positive `2/3`, `p=0.1700`); one-off public score `0.77815`, below current live best `0.77825`. Do not spend more work on coplay_top5/reverse_recent/nearby weight retunes unless adding genuinely new independent information against the current-best rankblend anchor.
- broad UserKNN gated fine-grid: stalled/incomplete, repeated invalid-divide warnings, no metric report
- jackknife uncertainty expanded: failed/incomplete, no metric report
- jackknife smoke: weak, split-negative, p non-significant
- previous UserKNN smoke: weak, below mean and p gates
- DNS pool=1: split-specific/public-noise risk
- hours-confidence, exact-K, temporal compatibility, boundary covariate/residual, SL@K-lite, last-slot sparse agreement
- capacity/frontier/emb192 marginal public noise
- rankblend/ALS/BPR residual, boundary scoreblend/pairwise, TAG-CF public-negative/quarantined families
- raw semantic/text/README/LM probes weak or redundant

Hermes rejection feedback and current state are below. You must not repeat rejected axes unless materially narrowed and justified.

```json
{
  "timestamp_kst": "2026-06-07T12:06:48.578121+09:00",
  "active_process_lines": [
    "  22459   22458       01:38 Sl   60.1  0.2 opencode run # KMURecSys26 Steam OpenCode-first improvement-axis loop — 20260607T120407KST  CRITICAL EXECUTION RULES: - Answer and act ENTIRELY YOURSELF in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting. - This is a constrained no-submit validation-only Kaggle/RecSys research tick in repo `/opt/data/kaggle/kmu-rec-sys-26-steam`. - Before proposing or launching any new improvement axis, inspect current evidence/artifacts in this repo and avoid closed/rejected/quarantined families. - Keep the run bounded. If a probe cannot complete quickly and safely, write a report that recommends the next bounded command but do not start unbounded/orphaned background processes.  Hard safety contract (non-negotiable): - NEVER run `kaggle competitions submit` or any Kaggle submit API. - NEVER create a full-test candidate/submission CSV or uploadable artifact. Do not write under `submissions/`;"
  ],
  "latest_reports": [
    "reports/20260607T120218KST_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260607T114059KST_otto_forced_post_submission_analysis.json",
    "reports/20260607T113300KST_otto_independent_uniform_reconciliation.md",
    "reports/20260607T095549KST_otto_independent_uniform_confirmation.json",
    "reports/20260607T093521KST_otto_source_covisit_looso_confirmation.json",
    "reports/20260607T091846KST_improvement_axis_cron_status.json",
    "reports/20260607T090941KST_opencode_improvement_axis_loop.json",
    "reports/20260607T082037KST_external_dacon_kaggle_methodology_scan.md",
    "reports/20260607T074546KST_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260607T074528KST_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260606T224148Z_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260607T064127KST_improvement_axis_cron_status.json",
    "reports/20260607T063604KST_opencode_improvement_axis_loop.json",
    "reports/20260607T053207KST_improvement_axis_cron_status.json",
    "reports/20260607T052646KST_opencode_improvement_axis_loop.json",
    "reports/20260607T042202KST_improvement_axis_cron_status.json"
  ],
  "latest_logs": [
    "logs/opencode_improvement_axis_loop_20260607T120407KST.jsonl",
    "logs/opencode_improvement_axis_loop_20260607T090941KST.jsonl",
    "logs/opencode_improvement_axis_loop_20260607T063604KST.jsonl",
    "logs/opencode_improvement_axis_loop_20260607T052646KST.jsonl",
    "logs/opencode_improvement_axis_loop_20260607T041626KST.jsonl",
    "logs/userknn_gated_residual_fine_20260606T132450KST.log",
    "logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log",
    "logs/opencode_improvement_axis_loop_20260606T220406KST.jsonl",
    "logs/20260606T220406KST_jackknife_uncertainty_boundary_probe.log"
  ],
  "state_files": [
    "state/aggressive_quota_runner_state.json",
    "state/autonomous_submission_policy.json"
  ],
  "known_incomplete": {
    "userknn_gated_residual_fine": {
      "expected_json": "reports/20260606T132450KST_userknn_gated_residual_fine.json",
      "expected_md": "reports/20260606T132450KST_userknn_gated_residual_fine.md",
      "expected_exist": false,
      "log_tail": "valid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n"
    },
    "jackknife_uncertainty_boundary_expanded": {
      "expected_json": "reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json",
      "expected_md": "reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.md",
      "expected_exist": false,
      "log_tail": "[20260606T224201KST] launching expanded validation-only jackknife uncertainty boundary probe\n[split] val_random_uniform_seed42\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n[split] val_random_uniform_seed42 base=0.765853; scanned=560\n[split] val_random_uniform_seed7\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n[split] val_random_uniform_seed7 base=0.760852; scanned=1120\n[split] val_random_uniform_seed123\n/opt/data/kaggle/kmu-rec-sys-26-steam/scripts/userknn_residual_probe.py:114: RuntimeWarning: invalid value encountered in divide\n  return np.where(std > 1e-12, (val - mean) / std, 0.0)\n"
    }
  },
  "latest_failed_axes_tail": [
    {
      "id": "stacker_logreg_lightgcn_stage2_20260530",
      "submitted_file": "artifacts/stacker_20260530/test_candidate/candidate_stacker_logreg_emb64_L3_reg1e-04.csv",
      "sha256": "ebd69b42548f4c48651905d54bd1e985d0bbbe57a7aa3db846b6ca05fdc637a0",
      "public_score": 0.75355,
      "anchor_at_submit": 0.76245,
      "delta_vs_anchor": -0.0089,
      "predicted_pooled_oof_gain": 0.00907,
      "actual_transfer_ratio": -0.981,
      "method": "logreg meta-learner over LightGCN + Stage2(mean_z, itemknn_bm25_top3, ease1000, als_popa2) + within-user z/rank + log_pop + cand_count; pooled-val train, per-user top-half decode",
      "honest_validation": "user-level GroupKFold OOF +0.00907, leakage gap 3e-5 (passed leakage check but DID NOT transfer)",
      "failure_hypothesis": "meta-learner exploited validation negative-sampler artifacts (log_pop weight -0.42, within-user Stage2 z) that correlate with constructed sqrtpop/popbin/recent negatives but NOT with the true hidden-test negative distribution; single robust LightGCN generalized better",
      "correlation_policy": "do NOT resubmit any LightGCN+Stage2 meta-learner that leans on popularity/within-user-Stage2 features without an out-of-distribution negative-sampler validation; honest GroupKFold alone is insufficient gating here"
    },
    {
      "id": "otto_coplay_top5_reverse_recent_forced_20260607T114059KST",
      "submitted_file": "submissions/candidate_otto_coplay_top5_reverse_recent_w0090_w0040_forced_20260607T114059KST.csv",
      "sha256": "d70af0f1e325c5c59985ed0df6dbc4232950edf6971f0ad26c72f7da8205985e",
      "public_score": 0.77815,
      "current_live_best_at_submit": 0.77825,
      "delta_vs_current_live_best": -0.0001,
      "emb128_reference_public": 0.77745,
      "delta_vs_emb128_reference": 0.0007,
      "independent_validation_mean_delta_vs_emb128": 0.0006668000266719654,
      "independent_validation_min_delta_vs_emb128": -0.000600120024004891,
      "independent_validation_positive_splits": "2/3",
      "independent_validation_p": 0.1700198674835568,
      "actual_transfer_ratio_vs_emb128": 1.04979,
      "method": "emb128_L4_reg1e-3 4-seed full-test score, within-user z, plus OTTO source co-visitation residual 0.090*z_coplay_top5_mean + 0.040*z_reverse_recent, per-user top-half decode",
      "failure_hypothesis": "The OTTO residual is a real but weak positive relative to the emb128 backbone; it does not provide enough independent signal beyond the stronger rank/capacity/ALS blend current best. Fresh split seed2718 was negative, matching the public outcome of not beating current best.",
      "correlation_policy": "Do not escalate same-panel tuned OTTO co-visitation/reverse_recent weights unless a fresh independent panel passes strict gate versus the current live-best anchor, not only versus emb128; diagnostic p-values are insufficient when min split delta is negative."
    }
  ],
  "recent_rejections": [
    {
      "iteration": 1,
      "sentinel": "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS",
      "hermes_verdict": "REJECTED_CONTINUE",
      "rejection_reasons": [
        "best_metrics_failed:min_delta,fixes_gt_breaks,quarantine"
      ],
      "safety_issues": [
        "active_kaggle_submit_process_detected"
      ],
      "metrics_checked": [
        {
          "variant": "blocked boundary/frontier internal row from prior reports; not actionable",
          "mean_delta": 0.0017003400680135987,
          "min_delta": null,
          "positive_splits": 3,
          "fixes": null,
          "breaks": null,
          "p": 0.0005242948680109718,
          "quarantine_conflict": true,
          "strict_pass": false,
          "failed": [
            "min_delta",
            "fixes_gt_breaks",
            "quarantine"
          ]
        }
      ]
    }
  ]
}
```

Required outputs for this iteration:
- Markdown report: `reports/20260607T120245KST_axis_loop_iter02_opencode.md`
- JSON report: `reports/20260607T120245KST_axis_loop_iter02_opencode.json`
- Optional validation artifacts only under `artifacts/opencode_hermes_axis_loop_20260607T120245KST`

Required JSON shape (you may add fields):
```json
{
  "safety_flags": {
    "validation_only": true,
    "candidate_csv_written": false,
    "full_test_candidate_or_submission_csv_created": false,
    "kaggle_submit_executed": false,
    "hidden_labels_used": false,
    "private_answers_used": false,
    "external_steam_scraping_used": false,
    "credentials_or_tokens_printed": false,
    "quarantine_or_guard_logic_weakened": false,
    "git_stage_commit_push_executed": false,
    "recursive_cron_scheduled": false
  },
  "axis_decision": "fresh axis tested / probe launched / no safe axis / blocked",
  "new_probe": {
    "launched": false,
    "status": "not_launched | running | completed | failed",
    "command": null,
    "pid_file": null,
    "log": null,
    "report_json": null,
    "report_md": null,
    "artifact_dir": null
  },
  "best_or_top_metrics": {
    "variant": null,
    "mean_delta_vs_base": null,
    "min_delta_vs_base": null,
    "positive_splits": null,
    "fixes": null,
    "breaks": null,
    "pooled_p_exact": null,
    "quarantine_conflict": false
  },
  "ranked_next_axis_hypotheses": [],
  "verdict": "STRICT_PASS | WEAK_SIGNAL | REJECT | NEXT_PROBE_RUNNING | NO_SAFE_AXIS | BLOCKED"
}
```

Final response requirements:
- Summarize what you did and why.
- Include report paths.
- Include safety flags.
- End with exactly one sentinel line:
  - OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND
  - OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING
  - OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS
  - OPENCODE_AXIS_LOOP_DONE_BLOCKED
