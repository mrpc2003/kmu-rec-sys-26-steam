CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- Read-only advisory only. Do NOT write, modify, delete, move, or create any files. Do NOT run Kaggle submit. Do NOT create candidate/submission CSVs.
- Do NOT print secrets, tokens, credentials, environment variable values, or config contents.
- End with the exact sentinel line: SEMANTIC_RESIDUAL_ADVISORY_DONE

Repository: /opt/data/kaggle/kmu-rec-sys-26-steam
Competition: KMU RecSys 26 Steam binary played prediction. Validation-only. Public labels hidden. Per-user top-half decoder. Strong baseline is emb128 L4 reg1e-3 LightGCN 4-seed ensemble; calibrated gate is uniform validation splits seed42/7/123, MDE about +0.00355. Prior closed axes include LightGCN variants, capacity, TAG-CF/layer mix, SL@K-lite, SASRec, GBDT/FM, boundary feature factory, README BPR/ALS/pop/hour residuals.

Task: Review this planned next probe and identify implementation pitfalls before Hermes implements it:
- Use only in-bundle train review text, no external scraping.
- For each fold split, build item text/profiles from fold train_interactions.csv only, not held-out validation positives.
- Use a non-gated HF embedding model such as Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, or nomic-ai/modernbert-embed-base; fallback to a cheap vectorizer only as a smoke/failure diagnostic.
- Construct item embeddings/profile, user positive centroid from fold train history, score validation candidate rows by cosine(user,item), normalize within user, residualize/blend against existing emb128 LightGCN ensemble scores, evaluate top-half accuracy and paired flips across seed42/7/123.
- No candidate CSV, no submission file. Allowed outputs later are reports/* and artifacts/semantic_residual_probe/* validation-only score/metric artifacts.

Please output:
1. Top 10 correctness/leakage/performance pitfalls.
2. Recommended minimal script architecture.
3. Exact validation gate and what would count as fail/pass.
4. Any reasons Qwen3-Embedding-0.6B may fail on V100 and fallback order.
