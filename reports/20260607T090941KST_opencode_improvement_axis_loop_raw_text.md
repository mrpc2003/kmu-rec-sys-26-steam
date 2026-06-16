I detect execution intent: validate whether a fresh no-submit improvement axis is credible, and produce the required loop reports. I’ll inspect the prior evidence/state first, then either run one bounded validation-only smoke or document why it is unsafe/duplicative.

Initial evidence confirms the named stalled/missing probes stayed incomplete or weak, and the external scan’s only plausible fresh clue is source-separated co-visitation rather than plain ItemKNN. I’m checking the implementation surface now to decide if a bounded smoke can be safely run.

The prior paper-guided work already covered time-decay ItemKNN/BM25, but not a validation-only residual blend that keeps OTTO-like source scores separate across co-play, ordered transitions, last-K, and hours weighting. I’m going to implement that as a bounded 3-uniform-split smoke against the emb128 4-seed reference.

I’m adding a standalone smoke script that writes only validation score diagnostics and the mandated reports; it never reads `data/pairs.csv`, never writes under `submissions/`, and uses only `artifacts/validation/*` candidates.

The smoke script is in place. I’m running syntax verification and then the bounded 600-second validation probe with output logged.

Pyright can’t resolve the project’s runtime packages in this environment and also flagged a few broad `object` annotations; I’m tightening the local annotations and limiting the diagnostic suppression to missing runtime imports.

The bounded smoke completed within the timeout and produced the reports; I’m verifying parseability, diagnostics, and safety outputs now.

The probe result is a weak but nonzero signal: 3/3 positive and p<0.05, but mean Δ is +0.00123, below the +0.0015 escalation gate. I’m cleaning the final diagnostic issue and checking the produced artifacts only.

Produced the required loop reports and ran one bounded validation-only smoke:

- `reports/20260607T090941KST_opencode_improvement_axis_loop.json`
- `reports/20260607T090941KST_opencode_improvement_axis_loop.md`
- Probe script: `scripts/otto_source_covisit_smoke.py`
- Validation artifacts: `artifacts/opencode_axis_loop_20260607T090941KST/`

Result: `WEAK_SIGNAL`, not strict pass. Top variant `base_plus_score_coplay_top5_mean_w0.2` had mean Δ `+0.00123358`, min split Δ `+0.00050010`, 3/3 positive splits, fixes/breaks `699/625`, pooled exact p `0.04479`. It misses the required mean Δ `+0.0015`, so no candidate/submission escalation.

Verified JSON parseability, report existence, validation score artifact headers are not `ID,Label`, `scripts/otto_source_covisit_smoke.py` has no LSP diagnostics, and no Kaggle submit/full-test candidate creation was performed.

OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL