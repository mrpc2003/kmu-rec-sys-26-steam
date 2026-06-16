# OpenCode task: KMURecSys26 Steam improvement-axis discovery loop

CRITICAL EXECUTION RULES — follow exactly:

1. Answer and act entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents, do NOT wait for parallel agents, and do NOT stop with “waiting for other agents”.
2. This is a Kaggle competition no-submit research loop. Absolutely forbidden:
   - `kaggle competitions submit`
   - creating submission CSVs or full-test candidate CSVs under `submissions/` or any path intended for upload
   - reading/using hidden labels or any private leaderboard answer source
   - external Steam scraping or collecting extra Steam reviews
   - printing secrets/tokens/credentials
   - weakening existing quarantine/guard logic
3. Allowed outputs only:
   - validation-only scripts under `scripts/` if needed
   - validation-only artifacts under `artifacts/opencode_axis_loop_20260606T220406KST/` or a clearly named validation artifact directory
   - logs under `logs/`
   - reports under `reports/`
4. Do not commit, push, or stage anything. If you change code, run `python -m py_compile` for touched scripts and report the diff paths.
5. Use only train/validation artifacts already in this repo and official public competition data under `data/raw/public/data`.
6. If you launch a long-running job, make it bounded, validation-only, log to `logs/`, write a report path, and state the command. Prefer using free GPUs only for V100-compatible torch: `uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' ...`.
7. End your final answer with exactly one of these sentinel lines:
   - `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
   - `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
   - `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`

Project/workdir:

`/opt/data/kaggle/kmu-rec-sys-26-steam`

User goal:

Use OpenCode to run an aggressive improvement-axis discovery loop. The goal is to find a real improvement axis beyond current public best behavior, not to repeat already closed axes. If a strict improvement axis is not immediately found, you must launch the next bounded validation-only probe that has a credible chance of finding one.

Current strongest public snapshot / target:

- Current known public best: `candidate_rank_blend_emb128_emb192.csv`, public `0.77825`.
- Main internal reference: emb128 4-seed ensemble uniform reference `0.76505`.
- Public LB has tracked the uniform-negative validation split better than sqrtpop/popbin.
- Noise band / weak blip caution: approximately `±0.0007`; one-split +0.000x is not enough.
- Candidate escalation requires multi-split support, fixes > breaks / paired significance when available, and no near-duplicate/quarantine conflict.

Current active process to avoid duplicating:

- A UserKNN gated residual fine-grid is currently running as `proc_d98ef5d36b4a`:
  - script: `scripts/userknn_gated_residual_probe.py`
  - report md: `reports/20260606T132450KST_userknn_gated_residual_fine.md`
  - report json: `reports/20260606T132450KST_userknn_gated_residual_fine.json`
  - artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST`
- Do not launch a duplicate of that exact grid unless you first verify it exited/failed.

Important recently closed/no-go axes:

- DNS pool=1 was fully rerun with score artifacts and rejected as split-specific noise:
  - `artifacts/dns_pool1_multisplit/three_uniform_panel/three_uniform_panel_summary.json`
  - best 3-split mean vs ref around `-0.00270`; positive splits `1/3`.
- Hours-confidence edge weighting closed:
  - user_quantile `0.76195`, item_quantile `0.76265`, balanced `0.76225`; all `CONF_PLATEAU_NO_GAIN`.
- Exact-K subset objective closed:
  - full prior summary `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`: subset vs BPR no gain; vs pretrained negative.
- Temporal compatibility axis rejected:
  - best temporal combiner around `0.67243`, massive regression vs `0.76505`.
- Boundary covariate expansion soft no-go:
  - residualized cooc/KNN AUC weak; raw signals pop-trap.
- SL@K-lite objective rejected:
  - `reports/20260601_slk_lite_panel_aggregate.md`, all 3 splits negative.
- Last-slot sparse agreement rejected:
  - `reports/20260604T115818KST_last_slot_sparse_agreement_probe.md`.
- Prior semantic/readme/LM text probes were weak or redundant; do not rerun raw semantic axes unless the design is genuinely different and multi-split validation-only.
- Capacity frontier / emb192 marginal gains were already public-tested as noise; emb128 sweet spot remains.

Useful weak-but-not-strict prior signals worth triage, not blind submission:

- `reports/20260602_aggressive_user_gated_zblend.md`: full user zblend had mean Δ about `+0.0015`, 3/3 positive, p≈0.003, but was marked manual-risk/no strict pass and related public variants later did not safely beat current best. Treat as a clue, not direct submit.
- `reports/20260603T180707KST_userknn_gated_residual_probe.md`: best UserKNN-gated variants had mean Δ around `+0.0008~+0.0009`, some 3/3 positive, strict pass count 0. A fine-grid is currently running.

Your required loop behavior in this OpenCode run:

1. Inspect the repo state, relevant reports, runner state/quarantine, and active process/log situation.
2. Decide whether the active UserKNN fine-grid has completed or is still running; if complete, analyze its report and decide if it is a strict candidate axis, weak clue, or reject.
3. Choose the next most credible improvement axis that is NOT a near-duplicate of the closed axes above. Prefer axes that are:
   - validation-only
   - low correlation / residual / uncertainty / boundary-calibration based
   - multi-split compatible
   - cheap enough to run bounded now
   - able to produce row-level score artifacts for aggregation, not only a printed score
4. If the chosen axis can run now, implement or launch the bounded validation-only probe. If it is too long, write a robust launch script/report and start at least the first bounded job.
5. Produce a report at:

`reports/20260606T220406KST_opencode_improvement_axis_loop.md`

and a compact JSON at:

`reports/20260606T220406KST_opencode_improvement_axis_loop.json`

Report schema/content:

- safety flags: validation_only, candidate_csv_written, kaggle_submit_executed, hidden_labels_used, external_scraping_used
- active/completed process summary
- closed-axis summary since this run
- ranked next-axis hypotheses with why each is independent
- exact command(s) run or launched
- artifacts/reports produced
- verdict: `IMPROVEMENT_AXIS_FOUND`, `NEXT_PROBE_RUNNING`, or `NO_SAFE_AXIS`
- next Hermes action: how to monitor or aggregate the result

Be aggressive but honest. The user wants an improvement axis, but false positives are worse than no candidate. A one-split weak blip must trigger expansion, not submission.
