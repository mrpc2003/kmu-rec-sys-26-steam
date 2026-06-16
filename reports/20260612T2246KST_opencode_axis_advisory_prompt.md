# OpenCode no-submit improvement-axis advisory — KMURecSys26 Steam

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response. Do NOT delegate to sub-agents. Do NOT wait for any parallel agents.
- This is an ADVISORY-ONLY no-submit task.
- Do NOT run `kaggle competitions submit`, any Kaggle API submit call, or any command that uploads/probes leaderboard outcomes.
- Do NOT run `kaggle competitions submissions` unless you only need read-only history; prefer not to call Kaggle at all.
- Do NOT create full-test `ID,Label` candidate/submission CSV files under `submissions/` or anywhere else.
- Do NOT use hidden labels, private answers, external Steam scraping, or external Steam metadata collection. Metadata is blocked unless the user/professor explicitly approves later.
- Do NOT print credentials, tokens, API keys, OAuth values, W&B keys, or secret env values.
- Do NOT modify repository files. Return advisory text in stdout only.
- You may inspect local repo reports/scripts if needed, but do not write, stage, commit, push, submit, or create candidate artifacts.
- End your final text with the exact line: `OPENCODE_AXIS_ADVISORY_DONE`.

Context:
- Workdir: `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- Competition: binary played prediction for `(userID, gameID)`, Accuracy, per-user test candidates decode with top-half positives.
- Current public best: `submissions/candidate_rank_blend_emb128_emb192.csv`, public `0.77825`.
- Stable backbone: emb128 LightGCN 4-seed, public `0.77745`, uniform surrogate around `0.76505`.
- Observed leaderboard top is `0.78795`, so our gap is about `+0.00970`; do not assume global Bayes ceiling, but do not submit weak axes.
- Public LB tracks uniform-negative validation better than sqrtpop/popbin; hard samplers are stress checks.
- Strict late-stage gate default: validation-only, no candidate CSV, no submit, mean delta >= `+0.0015`, min split delta >= 0, all splits positive, paired evidence when available, no quarantine/near-duplicate conflict.

Key recent reports to inspect if useful:
- `reports/20260612T2210KST_reopened_axis_results_and_next_plan.md`
- `reports/20260612T1901KST_progress_and_blockers_summary.md`
- `reports/20260612T213950KST_uniform_wide_ease_itemknn_aggregate.md`
- `reports/20260612T214616KST_gfcf_uniform_panel_probe.md`
- `reports/failed_axes.json`

Closed/weak axes you should avoid recommending again unless you propose a genuinely different mechanism:
- LightGCN capacity retune only: emb128 sweet spot; emb192 public did not beat emb128, rank-blend public best but weak statistically.
- EASE/ItemKNN/BM25/TFIDF/EASE-HTR: 3-split wide audit best mean only `0.742849`.
- GF-CF / graph spectral item-item filters: 3-split best blend mean `0.763786`, below emb128.
- Turbo-CF / PPR-like item-item filters: redundant/weak in previous report.
- ALS/WMF: pre-registered row weak, strict gate fail.
- stacker/logreg/GBDT meta-learning: negative public transfer due validation artifact/popularity down-weighting.
- hyperbolic geometry: lower solo and harmful blend.
- SASRec/sequential: objective mismatch for set-membership top-half task.
- simple hours/date/text/popularity residuals: weak or pop-trap in previous audits.
- seed expansion or simple ensemble retune alone: possible stability hedge, not a `+0.00970` gap explanation.

Your task:
1. Reconcile the current state from the above reports.
2. Identify whether there is any safe, internal-data-only next improvement axis worth running now.
3. If yes, propose exactly one bounded validation-only probe design that Hermes can implement/run next, with:
   - mechanism and why it is not already closed,
   - exact existing scripts to reuse or a minimal new script outline,
   - splits, metrics, strict gate, expected runtime,
   - safety flags.
4. If no safe internal-data-only axis remains, say so directly and classify remaining progress as metadata-blocked or final-packaging.
5. Do not recommend external metadata collection unless you label it blocked pending professor/user approval.

Required final structure:

## Verdict
One of: `NEXT_PROBE_DESIGN`, `NO_SAFE_INTERNAL_AXIS`, `BLOCKED_BY_METADATA_APPROVAL`, `WEAK_SIGNAL_ONLY`, `FAILED_BLOCKER`.

## Evidence Read
Bullet list of files/signals you used.

## Recommended Next Action
If `NEXT_PROBE_DESIGN`, give one concrete validation-only probe. If not, say what should happen next.

## JSON Summary
Put a fenced JSON object with keys:
- `verdict`
- `recommended_probe_name`
- `validation_only`
- `candidate_csv_written`
- `kaggle_submit_executed`
- `external_metadata_required`
- `blocked_reason`
- `strict_gate`
- `why_not_closed_axis`
- `commands_or_script_outline`

Final line must be exactly:
OPENCODE_AXIS_ADVISORY_DONE
