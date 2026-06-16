# OpenCode no-submit adviser — beat first place, not final packaging

CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response.
- Do NOT delegate to sub-agents.
- Do NOT wait for parallel agents.
- Do NOT call Kaggle submit.
- Do NOT create any full-test candidate CSV or file under submissions/.
- Do NOT collect external Steam metadata or scrape Steam.
- Do NOT print credentials, tokens, auth files, or environment secrets.
- Do NOT modify repo files except this run may be captured by Hermes from stdout.
- Return a concrete ranked experiment plan that can be run validation-only.
- End with exactly: OPENCODE_FIRST_PLACE_AXIS_DONE

Context:
- Workdir: /opt/data/kaggle/kmu-rec-sys-26-steam
- Competition: kmu-rec-sys-26-steam, binary played prediction, Accuracy, per-user exact top-half structure.
- Current best public: final slot1/rank-blend emb128+emb192 public 0.77825.
- Stable LightGCN emb128 L4 reg1e-3 4-seed public 0.77745, uniform surrogate 0.76505.
- Leaderboard #1: 0.78795, gap +0.00970 (~194 rows / 19,998), so final packaging is not enough.
- Already weak/closed: EASE/ItemKNN/BM25/TFIDF/EASE-HTR, Turbo-CF, GF-CF/spectral, ALS/WMF residual, boundary/residual stackers, SGL/DirectAU/xSimGCL/DNS/exact-K variants, 8-seed simple expansion, emb192/cross-capacity blend, logreg stacker/public negative transfer.
- User clarified: goal is to improve and beat first, not final selection.
- Remaining plausible internal axes from project memory: pseudo-label transduction/self-training over test/validation candidate structure, SWA/checkpoint averaging, and a genuinely different backbone that remains validation-only first. External metadata is blocked until explicit approval.

Task:
Give a ranked no-submit plan for the next 6-12 hours of experiments to maximize chance of beating 0.78795. Include:
1) Which experiment to launch first and why.
2) Validation-only implementation outline using existing scripts/data.
3) Strict gates for escalation to full-test materialization (but do not materialize now).
4) What to reject immediately.
5) Any risks of public overfit or rule issues.
