You are reviewing a Kaggle competition autorunner in /opt/data/kaggle/kmu-rec-sys-26-steam.

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response. Do NOT delegate to sub-agents.
- Do NOT call tools that submit to Kaggle, run W&B, print secrets, or modify files.
- This is a read-only advisory review. Do not write files.
- End with the exact line: AUTORUNNER_SIMILARITY_GUARD_REVIEW_DONE

User correction to implement in the code:
- The previous autorunner consumed 5 submissions rapidly and effectively submitted near-identical tuned values.
- New behavior must prevent submitting five similar candidates in one batch.
- After each submission, require post-submission analysis/calibration before another submission.
- Do not immediately submit nearby hyperparameter variants or identical/near-identical prediction CSVs.

Please inspect these target files conceptually:
- scripts/aggressive_quota_runner.py
- scripts/materialize_readme_rankblend_residual.py
- state/aggressive_quota_runner_state.json

Return a concise implementation checklist:
1. exact duplicate guards needed
2. near-duplicate row-diff threshold suggestion for binary top-half CSVs with 19,998 rows
3. family/quarantine rule suggestion based on negative transfer ratio
4. post-analysis artifacts to write before next submission
5. minimal code changes in aggressive_quota_runner.py
6. safety risks to verify after patch

Do not include credentials or raw tokens.