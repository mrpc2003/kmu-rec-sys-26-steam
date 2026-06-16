# OpenCode no-submit follow-up advisory — KMURecSys26 Steam

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- This is ADVISORY-ONLY. Do NOT modify repository files. Do NOT create scripts, artifacts, reports, or candidate files.
- Do NOT run `kaggle competitions submit` or any command/API that submits to Kaggle.
- Do NOT create or materialize full-test `ID,Label` candidate/submission CSV files.
- Do NOT use hidden labels/private answers, reverse-engineering, or external Steam scraping/metadata collection.
- Do NOT print credentials, tokens, API keys, OAuth data, W&B keys, or environment secret values.
- If you inspect files, restrict yourself to existing local reports/scripts/docs in this repo.
- End your final text with the exact line: OPENCODE_FOLLOWUP_AXIS_ADVISORY_DONE

Repository: /opt/data/kaggle/kmu-rec-sys-26-steam
Competition: Kaggle `kmu-rec-sys-26-steam`, binary Steam played prediction for userID-gameID pairs, Accuracy. Test/pairs structure is per-user 50/50 positive/negative; public LB uses half of test.

Current live leaderboard context:
- User current public best: `submissions/candidate_rank_blend_emb128_emb192.csv`, public 0.77825.
- Stable baseline: emb128 LightGCN 4-seed, public 0.77745, uniform-surrogate seed42 0.76505.
- Leaderboard #1 observed: 0.78795, so the gap is real; however many low-cost CF/spectral axes have now been rejected.

Already closed / weak axes to avoid repeating:
- EASE/ItemKNN/BM25/TFIDF wide audit on 3 uniform splits: best `score_itemknn_bm25_sum` mean 0.742849, far below emb128.
- GF-CF spectral panel on 3 uniform splits: best solo mean 0.757285; best blend50 mean 0.763786, below emb128 0.76505.
- Turbo-CF prior: best solo 0.74155, blend 0.75825, redundant.
- ALS-WMF/current-best reconciliation: did not safely exceed current best.
- TAG-CF full-test family: public negative transfer; quarantined.
- Hours-confidence gates: noise/negative; best about +0.00060 single mode, not enough.
- DNS pool=1: split-specific noise; final prior closure says reject.
- UserKNN global/gated residual: weak/unstable in prior context.
- Boundary/scoreblend/frontier/OTTO forced families: public or validation failures/quarantine.
- Seed expansion/SWA/pseudo-label are fallback/diagnostic only unless they produce clear multi-split evidence.

Latest new artifacts:
- `reports/20260612T2210KST_reopened_axis_results_and_next_plan.md`
- `reports/20260612T214616KST_gfcf_uniform_panel_probe.md/json`
- `reports/20260612T213950KST_uniform_wide_ease_itemknn_aggregate.md/json`
- `artifacts/validation_uniform_panel20_20260612T214626KST/` with 20 clean uniform splits.

Task:
1. Inspect the above evidence and the repo structure if useful.
2. Propose exactly ONE next validation-only improvement axis that is not a near-duplicate of closed axes and has the highest expected value under the 0.78795 gap.
3. Give a concrete minimal validation plan using existing scripts if possible, or a small new script design if necessary. Do not write it.
4. Classify the proposed axis as one of: STRICT_CANDIDATE_READY, WEAK_EXPAND, DIAGNOSTIC_ONLY, or NO_SAFE_AXIS.
5. Include strict safety flags:
   - validation_only
   - candidate_csv_written
   - kaggle_submit_executed
   - hidden_labels_used
   - external_scraping_used
6. If you judge there is no safe local axis left, say so directly and explain whether professor-approved Steam metadata is the only remaining high-value path.

Output format:
- Short verdict
- One next axis
- Why it is independent from closed axes
- Minimal commands or implementation sketch
- Gate criteria
- Risks / stop conditions
- Safety flags
- Final sentinel line
