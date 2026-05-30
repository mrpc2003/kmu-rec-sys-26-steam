# AI-Q deep researcher query — KMU Steam RecSys next-step validation

We are working on Kaggle `kmu-rec-sys-26-steam`.

Verified local facts:
- Task: binary implicit-feedback prediction. For each `userID, gameID`, predict whether user played game.
- Metric: Accuracy.
- Test set is exactly 50% played and 50% non-played. Public LB uses only half of the test.
- Train: `data/train.json`, ~175,000 Steam reviews with `userID`, `gameID`, `text`, `date`, `hours`, `hours_transformed=log2(1+hours)`.
- Test pairs are local `data/pairs.csv`; all test users/items are seen in train.
- Current inference shape: rank candidate games inside each user candidate set and label top half as played.
- Existing project methods: popularity, BPR, ItemKNN BM25, EASE, ALS, z-score blends. First approved public score: 0.74594.
- Previous local probes: cheap SVD/time/text axes do not beat Stage2 alone; review TF-IDF is weak standalone but may help as semantic cluster/blend feature under pop-bin stress.
- Rules: no external Steam scraping, no hidden-label reverse engineering, no Kaggle submission without explicit user approval. Public pretrained models are allowed only if reproducible.

Need a practical research/implementation memo, not a generic survey. Focus on these next actions:
1. ICPNS/community-aware negative sampling or validation split for implicit-feedback recommendation.
2. PURL/PU-learning or Correct-and-Weight style weighted implicit scorer.
3. Time-decay positive weighting / TFPS-like temporal filtration for EASE/ItemKNN/ALS.
4. Train-only review embeddings or pseudo-categories as a low-risk auxiliary feature.

For each action, provide:
- relevant papers or credible sources with URLs;
- exact adaptation to this competition;
- lightweight first experiment and stress gates (`sqrt-pop`, `recent`, `pop-bin`, per-user top-half);
- failure risks and reject criteria;
- reproducibility notes for eCampus.

Do not recommend scraping Steam metadata or submitting to Kaggle. Keep the final output concise and implementation-oriented.