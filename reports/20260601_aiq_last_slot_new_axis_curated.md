# AI-Q 신규축 탐색 결과 큐레이션 — 2026-06-01

**상태:** AI-Q `deep_researcher` job 성공 완료
**Raw artifact:** `reports/20260601_aiq_last_slot_new_axis_deep.json`
**Query:** `reports/20260601_aiq_last_slot_new_axis_query.md`
**No-submit:** Kaggle 제출 없음, candidate CSV 생성 없음

## 핵심 결론

AI-Q는 현재 마지막 슬롯 상황에서 **TAG-CF 계열 inference-time aggregation을 1순위로 계속 검증**하라고 판단했다. seed42 결과는 `+0.0032006`, McNemar `p=0.0090`로 방향성은 있으나, 단일 split MDE `+0.00355`에는 못 미치므로 **그 자체로 제출 후보는 아니다.** 지금 돌고 있는 seed7/seed123까지 포함한 3-split paired panel에서 살아남을 때만 마지막 제출 후보로 승격 가능하다.

AI-Q가 제시한 차순위는 다음 순서다.

1. **P0 — TAG-CF-style test-time neighbor aggregation**
   이미 실행 중인 축. full-test 후보는 3-split paired McNemar 통과 후에만 고려.
2. **P1 — LightGCN++-style layer-mixture rescoring**
   새 모델 학습이 아니라 `h0..hK` layer embedding 가중을 바꾸는 cheap rescoring 축. 기존 capacity sweep과 다르며, TAG-CF panel 이후 가장 실행비가 낮다.
3. **P1/P2 — boundary/listwise top-half reranking**
   SL@K / differentiable top-K 계열. 이론적으로 task와 맞지만 validation overfit 위험이 커서 tiny head/inner-fold 방식만 허용.
4. **P2 — non-sequential multi-interest centroid scoring**
   SASRec/DIN처럼 순서 모델이 아니라 유저 history item embedding을 K=2~4 centroid로 나누는 cheap routing. 기대값은 중간 이하.
5. **P3 — PU/denoising interaction-weight sensitivity**
   대부분 popularity/exposure/hard-negative trap과 겹치므로 마지막 단계에서는 거의 reject. model-disagreement 기반 positive-edge downweight만 짧은 probe로 가능.

## TAG-CF 제출 승격 조건

AI-Q와 기존 게이트를 합치면 TAG-CF는 아래 조건을 만족해야 한다.

- seed42/seed7/seed123 중 최소 2/3, 가능하면 3/3에서 `delta_vs_base > 0`
- pooled 또는 Fisher combined paired McNemar 유의
- 평균 delta가 단순 양수가 아니라 MDE 근방 또는 이상
- fixes > breaks가 최소 2/3 split에서 유지
- user degree, item degree, base-score margin subgroup에서 큰 regression 없음
- alpha/normalization variant grid는 seed42 이후 추가 확장 금지
- flip이 per-user cutoff 부근에서 발생하는지 확인
- item popularity / user history length / candidate marginal로 gain이 설명되지 않는지 확인

## 바로 실행 가치가 있는 다음 probe

TAG-CF panel이 실패하거나 경계선이면, AI-Q 기준으로 다음 cheap probe는 **LightGCN++ layer-mixture rescoring**이다.

실험 형태:

- full retrain 금지 또는 후순위
- 가능하면 LightGCN forward에서 layer별 embedding `h0,h1,h2,h3,h4`를 export
- fixed convex grid만 사용:
  - uniform baseline
  - shallow-heavy
  - mid-heavy
  - deep-light
  - small residual blend
- seed42에서 grid를 고정하고 seed7/seed123으로 transfer 확인
- split마다 best weight가 wildly 다르면 noise로 kill

## 보류/거절

- 새 graph encoder류: 대부분 같은 binary graph에서 LightGCN을 재발명할 가능성이 높아 마지막 슬롯엔 부적합
- LLM/text/semantic denoising: 이미 text 축이 약했고 비용 대비 기대값 낮음
- popularity/exposure/global quota/PU debiasing: public surrogate trap 가능성이 매우 큼
- IJCAI 2025 test-time graph transformation: 흥미롭지만 pseudo-label graph rewrite와 contrastive adaptation 위험이 커서 마지막 phase에는 과함

## 주요 출처

1. TAG-CF / message passing CF: https://proceedings.neurips.cc/paper_files/paper/2024/hash/10a3b1c30b8cceb507b9e8ddcc9a1a6a-Abstract-Conference.html
2. TAG-CF code: https://github.com/snap-research/Test-time-Aggregation-for-CF
3. LightGCN++ code: https://github.com/geon0325/LightGCNpp
4. RecSys 2024 accepted contributions: https://recsys.acm.org/recsys24/accepted-contributions
5. Simple effective loss for implicit feedback: https://arxiv.org/pdf/2601.04291
6. SL@K / Top-K ranking metric optimization: https://tiny-snow.github.io/assets/pdf/publications/KDD2025-ARXIV-2508.05673v1.pdf
7. Differentiable fast top-K selection: https://arxiv.org/html/2510.11472v1
8. difftopk: https://github.com/Felix-Petersen/difftopk
9. SoftSort: http://proceedings.mlr.press/v119/prillo20a/prillo20a.pdf
10. Multi-interest recommendation survey: https://arxiv.org/html/2506.15284v1
11. ComiRec: https://arxiv.org/pdf/2005.09347
12. PU implicit feedback MNAR: https://pmc.ncbi.nlm.nih.gov/articles/PMC12839574
13. Progressive Proximal Transport: https://haoxuanli-pku.github.io/papers/ICML%2025%20-%20Unbiased%20Recommender%20Learning%20from%20Implicit%20Feedback%20via%20Progressive%20Proximal%20Transport.pdf
14. Denoising implicit feedback slides: http://staff.ustc.edu.cn/~hexn/slides/wsdm21-ADT-slides.pdf
15. Denoising implicit feedback semantic scholar: https://www.semanticscholar.org/paper/Denoising-Implicit-Feedback-for-Recommendation-Wang-Feng/374f36c9081ab5dc686ab833c42a7297235cd13f
16. Test-Time Adaptation with Data-Centric Graph Transformation: https://www.ijcai.org/proceedings/2025/0510.pdf
