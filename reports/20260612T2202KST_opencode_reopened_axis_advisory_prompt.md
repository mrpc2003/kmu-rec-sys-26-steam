# OpenCode advisory prompt — KMURecSys26 reopened axis

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- This is an ADVISORY-ONLY no-submit task.
- Do NOT run `kaggle competitions submit`, `kaggle competitions submissions`, or any network/API call that submits or probes leaderboard outcomes.
- Do NOT create full-test `ID,Label` candidate/submission CSV files.
- Do NOT use hidden labels, private answers, or external Steam scraping/metadata collection.
- Do NOT print credentials, tokens, API keys, OAuth data, W&B keys, or environment secret values.
- Do NOT modify repository files. Return your advisory text in stdout only.
- You may inspect local reports/scripts if needed, but keep the final answer concise and action-oriented.
- End your final text with the exact line: OPENCODE_REOPENED_AXIS_ADVISORY_DONE

Context:
- Competition: Kaggle `kmu-rec-sys-26-steam`, binary played prediction for `userID,gameID` pairs, Accuracy, per-user hidden labels are exactly 50/50 positives/negatives, public LB uses half the test.
- Current user's public best: `submissions/candidate_rank_blend_emb128_emb192.csv`, public 0.77825.
- Current leaderboard top verified by Kaggle CLI: 0.78795, so gap vs our best is +0.00970 (~194 public rows / 19,998). Treat prior “near Bayes ceiling” conclusion as too strong; current saturation may be within our LightGCN/co-occurrence family, not global ceiling.
- Stable backbone: emb128 LightGCN 4-seed public 0.77745; uniform surrogate seed42 ensemble ref around 0.76505.
- Rules checked from Kaggle pages/local docs: no Steam review scraping or external answer acquisition; final two submissions can be selected; eCampus must reproduce selected submissions exactly. External Steam metadata should be considered unsafe until professor confirmation.
- Current no-submit work launched after reopening:
  - New 20 uniform validation panel generated: `artifacts/validation_uniform_panel20_20260612T214626KST`, 20 splits, no overlap/missing safety issues.
  - Wide EASE/ItemKNN audit completed: best `score_itemknn_bm25_sum` mean 0.742849 across seed123/42/7, far below 0.76505 LightGCN reference.
  - GF-CF spectral full 3-split run is still running. Smoke seed42: best solo 0.75245; best 50/50 blend with emb128 around 0.765053, essentially tied with reference.
- Known closed/weak axes from reports: EASE/ItemKNN/ALS/WMF/Turbo-CF/MultiVAE redundant or weak; DNS pool1 marginal then rejected on multi-split; hours-confidence edge weighting max +0.0006 noise band; exact-K subset loss no gain; temporal compatibility regressed; boundary covariates and residual atlas weak/pop-trap; OTTO/source co-visitation forced public 0.77815 < 0.77825 and strict gate failed; semantic/text/LM weak; capacity frontier emb128 sweet spot; emb192 public was slightly worse; SGL/DirectAU/xSimGCL weaker.

Task:
Given the 0.78795 external target and the just-completed EASE/ItemKNN negative result, identify whether any materially new, legal, bounded, validation-only experiment remains worth launching next. Prioritize axes that could plausibly explain a +0.0097 public gap and are not just retuning a quarantined family.

Output schema:
1. Verdict: one of `LAUNCH_NEXT_PROBE`, `WAIT_FOR_GFCF_THEN_DECIDE`, or `NO_SAFE_AXIS_AFTER_GFCF`.
2. Ranked next probes (max 3): for each, provide hypothesis, why it is independent from closed axes, exact local evidence/files to inspect, rough command/script idea, gate to pass, and risks.
3. What NOT to repeat.
4. Recommended immediate next action for Hermes while GF-CF is running.

Remember: no candidate CSV, no Kaggle submit, no external scraping, stdout only, end with sentinel.
