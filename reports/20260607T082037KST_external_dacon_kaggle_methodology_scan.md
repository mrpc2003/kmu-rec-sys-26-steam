# External DACON/Kaggle Methodology Scan

- Timestamp: 20260607T082037KST
- Project: KMURecSys26 Steam binary played prediction
- Purpose: 조사 요청 — DACON/Kaggle 추천 대회에서 실제 사용된 공개 방법론을 모아 현재 Steam 대회에 적용 가능한 축을 분류한다.
- Safety: no Kaggle submit, no candidate CSV generation, no hidden/private-label use, no Steam external scraping, no git stage/commit/push.

## Sources inspected

### Kaggle — OTTO Multi-Objective Recommender System

- Kaggle 1st place writeup/search excerpt: `https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution`
  - Average candidates around 1200.
  - Multiple co-visitation matrix variants with different action-type/time-period weights.
  - Candidate generation and reranker features both use those variants.
  - Session embedding adjusted by target action type.
  - Some models trained only on non-visited targets to reduce overlap with revisit-based candidates/features.
- TheoViel 3rd-place GitHub: `https://github.com/TheoViel/kaggle_otto_rs`
  - Classical candidate extraction + reranker pipeline.
  - About 80 candidates/session.
  - 744 engineered features.
  - RAPIDS/cuDF feature computation on 32GB V100.
  - XGBoost rank/classifier models; blending for carts/orders.
- Nicolaivicol OTTO GitHub: `https://github.com/nicolaivicol/otto-recommender`
  - Two-stage pipeline: retrieve candidates → rank → top-20 per objective.
  - Candidate sources include self/revisited items, click→click, click→cart/buy, cart→cart, cart→buy, buy→buy, Word2Vec item similarity, and popularity candidates.
  - Retrieval recall is explicitly measured by candidate pool size before ranking.
- General OTTO candidate rerank article/search result: `https://www.leoniemonigatti.com/blog/recommender-system-ml.html`
  - Common competition recipe: co-visitation matrix candidate generation followed by GBDT reranker.

### Kaggle — H&M Personalized Fashion Recommendations

- Wp-Zhang silver solution GitHub: `https://github.com/Wp-Zhang/H-M-Fashion-RecSys`
  - 2 recall strategies.
  - For each recall strategy: LGB ranker, LGB classifier, DNN.
  - Ensemble of different recall strategies improved LB from about 0.0286 to 0.0292.
  - Hardware-limited setting: average 50 candidates/user, 4 weeks of training data.
- Kaggle solution-summary/search excerpts:
  - LGB model used to select about 130 product candidates/user.
  - Features: user/item static attributes, user-item interaction counts, similarities, popularity, recall-source features, time/last purchase/average interval, age-product and higher-order combinations.
  - Recall methods: recent popular items, age-personalized popular items, and combinations.
- H&M 4th-place/search excerpt:
  - Two Tower MMoE was a focus area.
  - LightGBM improved CV by roughly 0.0005–0.0008 in the cited solution summary.

### DACON — 웹 기사 추천 AI 경진대회

- Competition/data page: `https://dacon.io/competitions/official/236290/data`
  - Tags: 정형, 언어, 추천시스템, 웹 로그.
  - Metric: Recall.
- GNOEYHEAT GitHub: `https://github.com/GNOEYHEAT/RecSys_article`
  - Private 4th, private score 0.34141.
  - CF_user, CF_item, CF_user+item blend.
  - Reported results show CF_user stronger than CF_item, and user+item blend with alpha 0.3/0.5 tested.
- DACON interview GNOEYHEAT: `https://dacon.io/competitions/official/236290/talkboard/412169`
  - Key trick: consider recommending articles the user already viewed.
  - Practical point: revisit/self-history candidates mattered for Recall.
- DACON interview 윤대혁: `https://dacon.io/competitions/official/236290/talkboard/412172`
  - Baseline/simple fundamentals mattered more than flashy methods.
  - Cosine similarity-only approach was expanded by studying/implementing additional theory.
- DACON interview 삼성역_4번출구: `https://dacon.io/en/forum/412171`
  - Matrix Factorization was the main studied/applied component.
  - They intentionally avoided submitting for 15 days to reduce leaderboard overfitting.
  - Process: research → hypothesis → experiment → eliminate hypotheses.
- DACON code share private 5th: `https://dacon.io/competitions/official/236290/codeshare/11033`
  - TF-IDF + improved baseline.
  - Related post metadata on the page reports:
    - Private 2nd: SentenceTransformer + MF + Ensemble.
    - Private 1st: BM25-weighted cosine similarity + LMF.
- DACON code share private 7th: `https://dacon.io/competitions/official/236290/codeshare/11036`
  - Hybrid GNN + content-aware recommender.
  - TF-IDF → TruncatedSVD → content similarity.
  - User–article interaction graph, GraphSAGE, node features.
  - Final score combines GNN similarity, interaction matrix, and content similarity.
- DACON code share private 12th: `https://dacon.io/competitions/official/236290/codeshare/11198`
  - TF-IDF + Doc2Vec for article recommendation.

## Cross-competition patterns

1. Two-stage retrieval/reranking dominates practical Kaggle RecSys.
   - First build a diverse candidate pool from multiple retrieval sources.
   - Then rank with GBDT/XGBoost/LightGBM/DNN using many interaction/time/popularity/source features.

2. Candidate-source diversity is often more important than model novelty.
   - OTTO/H&M both emphasize multiple recall/candidate strategies.
   - H&M silver solution explicitly reports that ensembling different recall strategies improved score.

3. Co-visitation/transition matrices are a strong competition heuristic.
   - OTTO used multiple co-visitation matrices by action type and time/aggregation window.
   - Even when a final ranker exists, co-visitation appears both as candidate source and reranker feature.

4. Revisit/self-history candidates can matter.
   - OTTO includes `src_self`/revisited candidates.
   - DACON web article winner interview explicitly says considering articles already viewed by users was important.

5. MF/LMF/CF are still competitive when the task is mostly interaction-only.
   - DACON top methods mention MF, LMF, user/item CF, BM25/cosine + LMF.
   - This matches our Steam setting better than content-heavy GNN/Transformer approaches.

6. Text/content features are usually auxiliary, not a complete replacement for CF.
   - DACON text methods: TF-IDF, Doc2Vec, SentenceTransformer, BM25.
   - High ranks came from combining text similarity with MF/LMF/ensemble rather than using text alone.

7. GNN/Two-Tower/Transformer methods appear, but mainly when side information, sequence domains, or cold-start are substantial.
   - H&M: Two Tower MMoE and DNN rankers with rich customer/product metadata.
   - DACON: GraphSAGE with article text/content features.
   - For Steam fixed-pair binary played prediction, these are plausible only if they add orthogonal signal beyond LightGCN/MF.

8. Validation discipline is a recurring winner behavior.
   - DACON winners mention avoiding leaderboard overfitting and using hypothesis elimination.
   - Kaggle top solutions often rely on reliable CV before tuning/blending.

## Fit to current Steam competition

Current project facts already established:

- Strong saturated backbone: LightGCN emb128 4-seed ensemble.
- Public LB tracks the uniform-negative validation split better than hard popularity samplers.
- Many families have been tested/closed: LightGCN variants, EASE/ALS/itemCF/MultiVAE/SASRec, GBDT/FM collaboration, text TF-IDF standalone, hyperbolic/capacity/seed/blend axes.
- Strict rule: no candidate CSV or Kaggle submit unless validation-only evidence passes gate.

### High-priority transferable axis: OTTO-style co-visitation/transition features

Why it is still worth checking:

- We have LightGCN, EASE, itemCF, ALS, but OTTO-style co-visitation is not just generic itemCF. It uses multiple windows/weights/type-specific transitions and exposes source-specific scores to a ranker/blend.
- Steam has timestamps (`date`) and strength (`hours`, `hours_transformed`), which can stand in for OTTO action type/time weights.
- The candidate set is fixed user-game pairs, so full candidate generation is unnecessary; we can score each given pair using transition/co-vis features from the user history.

Possible variants:

- Item co-play count from users who played both games.
- Time-decayed item→item co-visitation using review date order.
- Last-K history weighted co-visitation score for candidate game.
- Hours-weighted co-play/transition matrix.
- User-history max/mean/sum over item-to-candidate similarities.
- Source-specific features kept separate, then only validation-only residual blend/ranker tested.

Risk:

- Could collapse to existing itemCF/LightGCN signal.
- Must validate as paired residual delta on uniform split first.

### Medium-priority axis: DACON BM25/cosine + LMF-style weighted interaction model

Why:

- DACON private 1st metadata reports BM25-weighted cosine + LMF.
- Steam review text TF-IDF standalone was weak, but BM25/text similarity can be used as item-item smoothing or confidence weighting for latent factor scoring.

Possible variants:

- Logistic Matrix Factorization with confidence from hours/log-hours.
- BM25-weighted user profile over game review text, then combine with LMF score.
- Item text similarity only as residual/source feature, not as primary score.

Risk:

- Prior GBDT/FM/text probes suggest low orthogonality.
- Needs cheap validation-only smoke first.

### Medium-priority axis: robust User+Item CF blend, but not the stalled broad UserKNN run

Why:

- DACON GNOEYHEAT result shows user-CF stronger than item-CF and a user+item alpha blend was competitive.
- Our previous UserKNN fine-grid stalled with invalid divide warnings and no report, so as-is rerun is not acceptable.

Safer version:

- Patch NaN/std-zero handling.
- Bound to small grid and one split first.
- Compare UserKNN, ItemKNN, and User+Item residual blend against LightGCN on uniform validation.

Risk:

- Existing smoke was weak and p≈0.054/mean delta below strict threshold.
- Treat only as a cleaned diagnostic, not a candidate until strict gate passes.

### Low-to-medium priority: semantic text embedding source

Why:

- DACON private 2nd metadata reports SentenceTransformer + MF + Ensemble.
- Text methods in article recommendation are stronger because article text is directly semantic; Steam reviews are noisier and partly preference commentary.

Possible variants:

- Aggregate game text from provided train reviews only.
- Build SentenceTransformer/BM25 item vectors.
- Use item-to-item semantic similarity to user history as one source feature.
- Validate only as residual feature on top of LightGCN.

Risk:

- Standalone TF-IDF was weak in current project.
- External pretrained model is allowed only if publicly available and fully reproducible; no external Steam scraping.

### Lower priority: Two-Tower/MMoE/GraphSAGE/Transformer reranker

Why lower:

- H&M/DACON GNN/TwoTower successes relied on richer side info or content/news/fashion metadata.
- Our current problem is fixed-pair binary played prediction with strong graph signal saturation.
- SASRec/objective-mismatch was already rejected; GBDT/FM collaboration was highly correlated with LightGCN.

Possible use only if:

- It consumes a newly orthogonal source, e.g. co-visitation transition features or semantic clusters.
- It is not another heavy model that simply reconstructs LightGCN.

## Suggested next action order

1. Run a validation-only OTTO-style co-visitation/transition matrix smoke.
   - No submission/candidate CSV.
   - Features: all-pair co-play, time-decayed co-play, last-K history, hours-weighted similarity.
   - Evaluate source scores individually and as residual blend against LightGCN.
   - Gate: mean delta >= 0.0015, min delta >= 0, positive splits 3/3, fixes > breaks, p < 0.05.

2. If and only if co-vis has non-redundant signal, train a tiny ranker/residual blender using only existing fixed test-pair-style validation rows.
   - Do not create full-test predictions.
   - Do not use hidden labels.

3. Run DACON-LMF smoke only after co-vis smoke or in parallel on idle CPU.
   - LMF/logistic MF with hours confidence.
   - Optional BM25/text score as auxiliary residual.

4. Revisit UserKNN only as a cleaned bounded diagnostic.
   - Fix invalid divide/zero-std NaN path first.
   - Avoid broad fine-grid until the smoke gives a strict-compatible signal.

## Bottom line

The external methods do not suggest “another huge neural model” as the best next bet. The strongest transferable idea is:

> OTTO-style multi-source co-visitation/transition scoring + validation-only residual gate.

DACON reinforces that MF/LMF/user-item CF and simple similarity baselines can still win, but in our Steam project those overlap heavily with already-tested CF/LightGCN families. Therefore the only clearly under-tested external pattern is not plain itemCF, but a richer, time/hours/source-separated co-visitation feature family evaluated under the existing strict uniform-validation gate.
