# OpenCode no-submit advisory prompt — after boundary v1 forced probe

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response.
- Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- Do NOT run commands. Do NOT write files. Do NOT create candidate/submission CSVs.
- Do NOT call Kaggle, do NOT submit, do NOT scrape external hidden-label sources, do NOT request credentials.
- This is an advisory-only review.
- End with exactly this sentinel line: `BOUNDARY_AFTERCARE_ADVISORY_DONE`

Context:
- Competition: KMURecSys26 Steam binary played prediction, Accuracy.
- Current best kept candidate: `candidate_rank_blend_emb128_emb192.csv` / final slot1 public 0.77825.
- Stable backup: emb128 LightGCN 4-seed public 0.77745.
- Latest forced manual-risk boundary v1 probe:
  - file `submissions/candidate_boundary_v1_ridge_fast_panel20_forced.csv`
  - public 0.77705, delta vs best -0.00120.
  - row diff vs current best 176, estimated public changed rows 88, public implied precision ~0.432.
  - scored panel20 ridge-fast gate failed: mean flip precision 0.545, mean net rows +6.3, positive split ratio 0.70, worst split -14, top2/top1 pass false.
- Closed axes already include: LightGCN capacity/seed/SWA/checkpoint, GF-CF/EASE/ItemKNN, logreg stacker/meta-learner, ALS residual/boundary retunes, OTTO co-visitation weak signal, pseudolabel/transduction, metadata without gameID→appid mapping, boundary row-flip families.
- Official constraints: no external hidden labels, no Steam scraping/deanonymization without approved mapping, final candidates must be reproducible.

Task:
Give a compact verdict for the next step. Choose exactly one of:
1. `NO_SAFE_INTERNAL_AXIS` — stop speculative probes; move to final packaging.
2. `SAFE_PROBE` — one validation-only internal probe is still worth running.
3. `NEEDS_HUMAN_RULE_DECISION` — only an external/rules-dependent path remains.

If `SAFE_PROBE`, specify only one concrete validation-only probe with:
- hypothesis
- why it is independent from the closed families
- exact artifacts to read/write under reports/ or validation-only artifacts/
- gate before any candidate materialization
- risk of public overfit

If `NO_SAFE_INTERNAL_AXIS`, explain why in 5 bullets and list final-packaging items to verify.

Do not be optimistic for appearance. Penalize same-family retunes hard.
