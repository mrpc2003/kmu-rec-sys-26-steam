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
  "timestamp_kst": "2026-06-07T07:48:41.499419+09:00",
  "active_process_lines": [
    "   6081     170       02:55 Ss    0.0  0.0 /usr/bin/bash -lic set +m; set -euo pipefail cd /opt/data/kaggle/kmu-rec-sys-26-steam TS=$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST) LOG=\"logs/${TS}_opencode_hermes_axis_rejection_loop_driver.log\" mkdir -p logs reports artifacts printf '[%s] starting OpenCode↔Hermes rejection loop\\n' \"$TS\" | tee -a \"$LOG\" python3 scripts/opencode_hermes_axis_rejection_loop.py \\   --max-iters 6 \\   --opencode-timeout-sec 900 \\   --probe-wait-sec 3600 \\   --sleep-between-iters-sec 20 \\   2>&1 | tee -a \"$LOG\" RC=${PIPESTATUS[0]} printf '[%s] OpenCode↔Hermes rejection loop exit rc=%s log=%s\\n' \"$(TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST)\" \"$RC\" \"$LOG\" | tee -a \"$LOG\" exit \"$RC\"",
    "   6404    6081       02:55 S     0.0  0.0 tee -a logs/20260607T074546KST_opencode_hermes_axis_rejection_loop_driver.log"
  ],
  "latest_reports": [
    "reports/20260607T074528KST_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260606T224148Z_opencode_hermes_axis_rejection_loop_summary.json",
    "reports/20260607T064127KST_improvement_axis_cron_status.json",
    "reports/20260607T063604KST_opencode_improvement_axis_loop.json",
    "reports/20260607T053207KST_improvement_axis_cron_status.json",
    "reports/20260607T052646KST_opencode_improvement_axis_loop.json",
    "reports/20260607T042202KST_improvement_axis_cron_status.json",
    "reports/20260607T041626KST_opencode_improvement_axis_loop.json",
    "reports/20260607T031100KST_improvement_axis_cron_status.json",
    "reports/20260607T020557KST_improvement_axis_cron_status.json",
    "reports/20260607T005922KST_improvement_axis_cron_status.json",
    "reports/20260606T235336KST_improvement_axis_cron_status.json",
    "reports/20260606T224445KST_repeat_until_candidate_status.json",
    "reports/20260606T220406KST_opencode_improvement_axis_loop.json",
    "reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json",
    "reports/20260606T125011KST_dns_pool1_panel_aggregate.json"
  ],
  "latest_logs": [
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
  "recent_rejections": [
    {
      "iteration": 1,
      "sentinel": "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS",
      "hermes_verdict": "REJECTED_CONTINUE",
      "rejection_reasons": [
        "best_metrics_failed:mean_delta,min_delta,positive_splits,fixes_gt_breaks"
      ],
      "safety_issues": [],
      "metrics_checked": [
        {
          "variant": null,
          "mean_delta": null,
          "min_delta": null,
          "positive_splits": null,
          "fixes": null,
          "breaks": null,
          "p": null,
          "quarantine_conflict": false,
          "strict_pass": false,
          "failed": [
            "mean_delta",
            "min_delta",
            "positive_splits",
            "fixes_gt_breaks"
          ]
        }
      ]
    }
  ]
}
```

Required outputs for this iteration:
- Markdown report: `reports/20260607T074546KST_axis_loop_iter02_opencode.md`
- JSON report: `reports/20260607T074546KST_axis_loop_iter02_opencode.json`
- Optional validation artifacts only under `artifacts/opencode_hermes_axis_loop_20260607T074546KST`

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
