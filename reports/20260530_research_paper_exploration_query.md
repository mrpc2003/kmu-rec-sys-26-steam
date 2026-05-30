# AI-Q query — KMU RecSys 26 Steam paper-guided exploration

We are working on a Kaggle-style recommender-system competition: `kmu-rec-sys-26-steam`.

Verified local competition facts:
- Task: binary implicit-feedback prediction: for each `userID, gameID` pair, predict whether the user played the game.
- Metric: Accuracy.
- Test set is exactly 50% played and 50% non-played; public leaderboard is half of test only.
- Train: `data/train.json`, about 175,000 Steam reviews with fields `userID`, `gameID`, `text`, `date`, `hours`, `hours_transformed=log2(1+hours)`.
- Test pairs are in local `data/pairs.csv` (competition/baseline text may call it `pairs_Played.csv`). Test users/items are seen in train.
- Current local approach: candidate ranking within each user’s candidate set and top-half conversion to `Label=1`; validation uses user-preserving candidate groups and negative samplers including sqrt-popularity/bin-matched negatives.
- Existing methods in project: popularity baseline, BPR baseline, ItemKNN BM25, EASE/EASE variants, ALS CF, z-score blend. First approved submission public score: 0.74594.
- Operational constraints: no external Steam review scraping, no reverse engineering hidden labels, no private test leakage. Publicly available pretrained models are allowed only if reproducible. No Kaggle submission without explicit user approval.

Research request:
Find recent, high-quality papers from roughly 2023–2026 that address tasks similar to this: implicit-feedback recommendation, binary user-item played/purchased/clicked prediction, game/content recommendation from reviews, graph collaborative filtering, sequential recommendation, contrastive/denoising recommendation, LLM/text-enhanced recommendation, negative sampling or debiasing for implicit feedback.

For each selected paper/family, provide:
1. Paper title, year, venue/arXiv/URL when available.
2. Core method idea in 2–4 sentences.
3. What data signal it uses: user-item ID graph, sequence/time, text/reviews, hours/intensity, popularity/negative sampling, LLM embeddings, etc.
4. How it could be adapted to this Steam played prediction competition without violating rules.
5. Concrete exploratory experiments to run first, ranked by expected ROI and implementation cost.
6. Failure risks specific to this competition: public-LB half split, balanced top-half labels, validation surrogate mismatch, small train size, text noise, compute cost, reproducibility for eCampus.
7. Recommended gating metrics before any submission.

Output should be practical and competition-focused, not a generic survey.