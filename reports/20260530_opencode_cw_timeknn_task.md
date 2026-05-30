TASK: Fix and extend the KMU RecSys Steam played-prediction next-step validation script. This is VALIDATION-ONLY research. You must NOT submit to Kaggle, NOT scrape external Steam data, NOT reverse-engineer hidden labels.

Repo: /opt/data/kaggle/kmu-rec-sys-26-steam
Primary file: scripts/paper_guided_next_steps.py
Reference file (existing Stage2 ItemKNN): scripts/score_popularity_itemknn_ease.py
Shared utils: scripts/recsys_played_utils.py (do not break its public API).

CONTEXT / RULES (must respect):
- Task: binary implicit-feedback "played" prediction for (userID, gameID) pairs; metric Accuracy.
- Inference shape is per-user candidate ranking, then top-half -> Label=1. Every score must remain compatible with evaluate_tophalf / predict_tophalf in recsys_played_utils.py and must NOT violate per-user top-half positive counts.
- Validation splits live under artifacts/validation/: val_random_sqrtpop_seed42, val_recent_sqrtpop_seed42, val_random_popbin_seed42, val_random_communitypop_seed42, val_recent_communitypop_seed42. Each has train_interactions.csv and candidates.csv.
- All training features must be computed from fold-train only (no leakage from heldout positives).
- Run env: use `env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 uv run --with numpy --with pandas --with scipy --with scikit-learn python ...`.

BUG TO FIX (confirmed): add_weighted_implicit_logit() (around lines 622-647) produces score_cw_weighted_implicit_logit that scores BELOW 0.5 (worse than random: ~0.40-0.53) on every split. Root cause: train/inference feature distribution mismatch. The training pairs (train_scored) only get add_basic_stats + add_community_scores, but feature_columns() also includes score_time_itemknn_*, score_time_ease_*, score_graph_svd_k64, score_review_pseudocat_*. Those columns are missing for training pairs and get filled with 0.0 (lines 628-630), while candidate inference rows have REAL values. So the LogisticRegression learns weights against all-zero columns and then sees real values at inference -> degenerate, often sign-flipped scores.

REQUIRED FIX for CW-lite:
1. Make the training-pair feature matrix come from the SAME feature pipeline as the candidate rows. Concretely: build the CW training pairs (positives = fold-train observed interactions, negatives = community-reliable sampled negatives via the existing sample_community_negatives), then run them through the SAME functions used for candidates: add_basic_stats, add_community_scores, add_time_decay_graph_scores, add_svd_scores, add_review_pseudocat_scores -- all fit on fold-train only. Do NOT zero-fill features that the model will see as real at inference.
2. If recomputing every heavy feature for all training pairs is too slow, instead restrict feature_cols used by the logit to ONLY the features that are actually computed identically for both train pairs and candidates (e.g. drop the heavy ones), and document that choice. Either way, train and inference feature semantics must match.
3. Keep per-user normalization consistent: if candidate features are within-user normalized before being used, the training-pair features must be normalized the same way. Check whether decision_function output should also be ranked within user (it is converted to top-half later, so raw decision_function is fine, but the FEATURES must be on the same scale as inference).
4. Sanity gate inside the script: after computing score_cw_weighted_implicit_logit, assert (or log a clear WARNING) if its row_accuracy on val_random_sqrtpop_seed42 is below 0.55, so future regressions are visible.

REQUIRED ADDITION for time-decay ItemKNN (TFPS-style), promote to a clean reusable form:
5. The probe already has score_time_itemknn_hl{90,365,730}_{sum,top3} via add_time_decay_graph_scores. Make the time-decay edge weighting explicit and correct: the user-item matrix entries should be weighted by recency_weights(half_life) BEFORE BM25/cosine item-item similarity, matching the Stage2 ItemKNN BM25 approach in score_popularity_itemknn_ease.py (bm25_weight + compute_item_similarity + per-user top-3 mean). Add a BM25-weighted time-decay variant (e.g. score_time_itemknn_bm25_hl{hl}_top3) so it is directly comparable to the existing Stage2 score_itemknn_bm25_top3. Keep the existing plain variants too.

CONSTRAINTS:
- Do not change the no-submit posture or remove safety checks.
- Do not edit files outside scripts/ and reports/ unless strictly required; if you touch recsys_played_utils.py keep backward compatibility.
- After editing, run `python -m py_compile` on changed scripts and do ONE fast smoke: run the scorer on a SINGLE split (val_random_sqrtpop_seed42) only, with reduced cost if needed, to confirm score_cw_weighted_implicit_logit now scores >= 0.55 row accuracy and the new time-decay BM25 variant produces finite values. Do NOT run the full 5-split sweep (Hermes will do that).
- Print the smoke result (CW-lite row accuracy on val_random_sqrtpop_seed42, and a couple of time-decay scores) to stdout.
- Clean up any scratch files you create.

End your final message with a line: VERDICT: IMPL_COMPLETE  (or VERDICT: IMPL_BLOCKED with the reason).
