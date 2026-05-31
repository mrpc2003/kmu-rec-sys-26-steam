# 2026 SOTA 탐색 라운드 종결 — 4개 패밀리 음성 결과 (KMURecSys26 Steam, 2026-05-31 KST)

사용자 지시("최신 논문 방법론 지속 탐색")에 따라 best(emb128 4-seed, public **0.77745**, uniform
0.76505)를 넘을 **구조적으로 다른 강한 백본**을 2024~2026 문헌에서 탐색·구현·검증했다. 결론을
먼저: **4개 패밀리 모두 uniform 공개-대리 게이트를 통과하지 못했다.** CF 구조 천장이 SOTA 수준에서도
재확증됐다.

## 검증한 4개 패밀리 (parameter-free uniform 게이트: emb128 4-seed 0.76505를 noise 0.0007 이상 초과해야 채택)

| 패밀리 (출처) | 메커니즘 | 핵심 결과 (uniform) | 판정 |
|---|---|---|---|
| **XSimGCL** (Yu, TKDE'23) | edge-drop 제거 + embedding noise CL, 단일 pass | λ 0.02→0.741 / 0.05→0.706 / 0.1→0.667 / 0.2→0.624 단조 악화 | ❌ TIED_OR_WEAK |
| **Turbo-CF** (Park, SIGIR'24) | training-free 폴리노미얼 item-item LPF | solo 0.742(EASE>) but corr_z 0.78~0.85, blend −0.0068 | ❌ REDUNDANT |
| **AlphaRec-core** (ICLR'25 Oral) | frozen-LM 아이템표현 행동공간 집계 | solo 0.645, corr_z 0.486, blend −0.0348 | ❌ REJECT_WEAK |
| **DNS** (hard negative, KDD'21 계열) | BPR negative를 hard-mining해 백본 강화 | pool 8→0.677 / 16→0.652 / 32→0.626 단조 붕괴 | ❌ FAIL (popularity-skew) |

## 패밀리별 정직한 진단

1. **XSimGCL — contrastive-CF 패밀리 종결 (증거 확보).** SGL(edge-drop) 실패가 edge-drop 때문인지
   InfoNCE 때문인지 모호했는데, XSimGCL은 edge-drop을 제거해 변수를 분리했다. 결과적으로 **λ=0.02(거의
   순수 BPR)조차 백본 단일 seed 0.762 미만**이고 BPR loss는 0.119까지 잘 내려갔다 → 실패 원인은
   **InfoNCE-uniformity가 공유 임베딩을 끌어당겨 유저당 median-4 후보의 미세 ranking을 파괴**하는 것.
   SGL/SimGCL/XSimGCL/DirectAU 전체가 이 데이터에서 종결.
2. **Turbo-CF — graph-filtering CF는 EASE 패밀리.** "학습 불필요 새 패러다임"으로 보이지만 기계적으로는
   item-item 선형 스코어러. solo 0.742로 EASE(0.727)보다 강했지만 corr_z 0.78~0.85로 emb128과 신호 공유,
   blend 손해. classic-CF redundancy 재확인.
3. **AlphaRec-core — LM 표현 CF 축 종결.** 행동공간 집계(MiniLM의 raw-cosine과 다른 방식)로도 solo
   0.645로 floor 미만. MiniLM(0.639)과 거의 동일 → 인코딩 방식 무관하게, 익명 gameID + 노이즈 많은 유저
   리뷰라는 구조에선 텍스트 의미 신호가 본질적으로 약함.
4. **DNS — popularity-skew로 uniform 전이 실패.** hard negative pool이 클수록 단조 붕괴. 모델을 인기
   아이템 변별로 밀어붙이는데 public LB가 추종하는 uniform 분포엔 전이 안 됨(스크립트에 사전 경고한 리스크
   현실화). 대조군 pool1(0.76565)은 백본 재현성 확인용이며, ensemble +0.0006은 **단일 seed 변동(noise
   0.0007 이내)**으로 actionable하지 않음 — 이를 escalation으로 쫓는 것은 stacker-trap.

## 구조적 결론: 왜 SOTA가 안 먹히는가

2025-2026 추천 SOTA(XSimGCL/diffusion/LLM-rerank/AlphaRec)는 대부분 **대규모-희소 그래프 + text/sequence
신호** 전제. 이 대회는 정반대 — **소규모-상대조밀(6710u×2437i, 165k), 유저당 median-4 1:1 균형
reranking, test에 텍스트/시퀀스 없음, uniform 분포 추종**. SOTA의 강점(롱테일/콜드스타트/시맨틱/
hard-discrimination)이 이 구조에 전이될 이유가 약하다. 이것이 모든 직교축·SOTA가 게이트를 통과 못 한 근본 원인.

## 증거 기반으로 배제된 추가 후보 (실행 없이)

- **Diffusion RecSys** (arXiv 2505.09364 "Illusion of Progress"): 잘 튜닝된 베이스라인이 일관되게 초과 → 함정.
- **GBDT/tabular stacker**: logreg stacker가 이미 동일 방식 public 회귀(0.76245→0.75355).
- **LLM listwise reranker** (RankZephyr/RankK/DeAR): test pair에 텍스트 없음 → 적용 불가.

## 최종 상태

검증 가능한 직교축·백본 강화·SOTA 패밀리를 SOTA 수준에서 한 번 더 전수 탐색했고, 전부 음성. final-2는
변동 없음: **#1 emb128 4-seed (public 0.77745), #2 emb64 4-seed (0.77125)**. 새 제출 후보 없음.
지속 탐색 방침: 새 논문이 이 대회 구조(소규모 균형 reranking, uniform test, 텍스트 없는 test)에 구조적으로
부합할 때만 후보로 채택. 모니터링 계속.
