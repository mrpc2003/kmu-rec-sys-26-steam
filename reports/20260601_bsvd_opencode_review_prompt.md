CRITICAL EXECUTION RULES: Answer ENTIRELY YOURSELF in this single response. Do NOT delegate to sub-agents. Do NOT wait for any parallel agents. Do NOT write or modify files. End with the exact line: BSVD_REVIEW_DONE

We are in repo /opt/data/kaggle/kmu-rec-sys-26-steam for a Kaggle implicit-feedback Steam played task. Last submission slot remains. You are a read-only coding/research reviewer.

Safety contract: validation-only; no hidden-label acquisition; no public-LB probing; no Kaggle submission; no test/candidate/submission CSV materialization; do not run commands; do not modify files. Only inspect the proposal and give pitfalls / acceptance criteria.

Candidate probe: BSVD Boundary Seed-Vote Decoder. Existing canonical LightGCN emb128/L4/reg1e-3 4-seed validation scores exist for uniform splits. Idea: within each user, compute baseline raw-score mean top-half prediction. Compute seed-wise top-half votes (4 seeds) and mean seed ranks. Only rows in a fixed baseline boundary band w=min(20,max(3,ceil(0.10*h_user))) can be reordered; top-safe and bottom-safe rows are anchored. Boundary score = seed_tophalf_vote + 1e-3 * normalized inverse mean_seed_rank + 1e-6 * within-user z(base_score). Evaluate one fixed variant score_bsvd_boundary_vote_w10cap20 on splits seed42/7/123 against baseline raw-mean ensemble with paired exact binomial/McNemar. Gate: mean delta >= +0.00355, pooled fixes>breaks, p<0.05, direction >=2/3. Also report boundary oracle upper bound diagnostic.

Known available score sources:
- val_random_uniform_seed42: seed42 score from artifacts/layermix_probe/emb128_L4_r3_seed42/layermix_validation_scores.csv column score_layermix_uniform; seed7/123/2024 from artifacts/lightgcn_emb128L4r3_ens/seed{7,123,2024}/val_random_uniform_seed42/lightgcn_scores.csv column score_lightgcn. Base mean verifies to row_acc 0.7650530106021204.
- val_random_uniform_seed7 and val_random_uniform_seed123: seed42/7/123/2024 from artifacts/split_panel_emb128/{split}/seed{42,7,123,2024}/lightgcn_scores.csv. Base means verify to 0.7609521904380876 and 0.7599519903980796.

Task: Give a concise reviewer verdict: (1) is this distinct enough from closed axes to run one validation-only probe, (2) implementation pitfalls that could fake a gain/loss, (3) any changes needed before running, (4) expected kill conditions. Korean preferred. End exactly with BSVD_REVIEW_DONE
