I detect advisory review intent: you want a read-only implementation checklist for the similarity guard, not a patch. I’ll inspect the three target files directly and synthesize the minimal changes without submitting, logging secrets, or modifying files.
Read-only review completed. The current failure mode is clear: `run_one_submission()` immediately loops after a submit, while selection only blocks by variant name/submitted filename, not by prediction identity, row similarity, family, or post-result calibration.

## Implementation checklist

1. **Exact duplicate guards needed**
   - Block if candidate `sha256` matches any prior `submission_results[].sha256`.
   - Block if candidate row labels are identical to any previously submitted CSV or known current best CSV.
   - Block if `variant` already in `submitted_variants`.
   - Treat identical SHA under different variant names as a hard error and record it in state.

2. **Near-duplicate row-diff threshold**
   - For 19,998-row binary top-half CSVs, hard-block candidates with `<200` differing rows versus any submitted/best candidate.
   - Quarantine/manual-review candidates with `200-500` differing rows.
   - Require `>=500` row diffs unless the post-analysis artifact explicitly approves a smaller, theory-backed change.

3. **Family/quarantine rule**
   - Define family from parsed variant fields, e.g. `(mode, axis root, popa bucket, htr/non-htr)`.
   - Compute transfer ratio: `public_delta_vs_best / validation_mean_delta`.
   - If first family submission has ratio `<= 0`, quarantine that whole family.
   - If two submissions in a family fail to beat best, quarantine all nearby weights/modes in that family.
   - For the observed state, `score_als_htr_f32_it30_alpha20_popa4` residual weights should be quarantined immediately.

4. **Post-analysis artifacts before next submission**
   - Require a new artifact before unlocking the next submit, for example:
     - `reports/{ts}_autorun_post_submission_calibration.json`
     - `reports/{ts}_autorun_post_submission_calibration.md`
   - Include public delta, validation delta, transfer ratio, candidate-vs-prior row-diff matrix, SHA duplication check, family quarantine decision, and explicit `next_submission_allowed: true/false`.
   - Add state flag like `pending_post_analysis` after every submit; clear it only when the calibration artifact exists and approves continuing.

5. **Minimal `aggressive_quota_runner.py` changes**
   - Add helpers: `read_submission_labels()`, `row_diff_count()`, `variant_family()`, `similarity_guard()`, `post_analysis_complete()`.
   - In `run_one_submission()`, before `choose_next_variant()`, return idle if `state["pending_post_analysis"]` is set.
   - After materialization and preflight, run exact/near-duplicate guards before `submit_candidate()`.
   - After successful submit/result polling, set `pending_post_analysis` with variant, SHA, family, score, validation evidence.
   - Remove the immediate quota-burn behavior at lines 426-427; after one submission, exit or sleep until post-analysis unlocks.
   - Make `choose_next_variant()` skip quarantined families and not just submitted variant names.

6. **Safety risks to verify after patch**
   - Identical CSVs under different hyperparameters are blocked; current state proves this is needed.
   - A process restart cannot bypass `pending_post_analysis`.
   - Missing prior local CSVs do not silently disable SHA/state-based duplicate checks.
   - Row-diff comparison handles both `Played` and `Label`.
   - Quarantine does not block genuinely different model families.
   - No Kaggle submit, W&B logging, or git push happens during dry-run/similarity-check paths.

AUTORUNNER_SIMILARITY_GUARD_REVIEW_DONE