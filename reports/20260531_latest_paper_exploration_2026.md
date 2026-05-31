# 2026 최신 논문 탐색 — 새 방향 후보 평가 (KMURecSys26 Steam, 2026-05-31 KST)

best(emb128 4-seed, public 0.77745)를 넘을 **구조적으로 다른 강한 백본**을 2024~2026 문헌에서 탐색.
AI-Q 백엔드(8101/8100)는 다운 상태라 arXiv/web_search/Semantic Scholar 직접 조사로 수행.

## 후보별 판정

| 후보 (출처) | 핵심 | 이 대회 적합성 | 판정 |
|---|---|---|---|
| **XSimGCL** (Yu et al., TKDE'23) | edge-drop 제거, embedding 레이어별 uniform noise + cross-layer InfoNCE, 단일 forward pass | SGL과 달리 edge-drop 변수 분리 가능. 단 InfoNCE-uniformity 메커니즘은 공유 → 낮은 prior | 🔬 **실험 중** (GPU 4장) |
| Diffusion RecSys (survey 2501.10548; "Illusion of Progress" 2505.09364) | DDPM 기반 생성형 CF | **음성 증거 강함**: 18개 잘 튜닝된 베이스라인이 일관되게 diffusion 초과. 내 LightGCN 앙상블이 그 강베이스라인 | ❌ 함정, 추격 안 함 |
| GBDT/tabular stacker | 엔지니어드 feature에 트리 부스팅 | logreg stacker가 이미 동일 방식 public 회귀(0.76245→0.75355). 트리는 popularity 과적합 더 심함 | ❌ 알려진 stacker-trap |
| LLM listwise reranker (RankZephyr/RankK/DeAR, 2025) | LLM 기반 listwise 재정렬 | test pair는 익명 ID만, 텍스트 없음 | ❌ 적용 불가 |
| RP3β / 랜덤워크 item-item | popularity 페널티 3-step 랜덤워크 | classic CF family 전체가 이미 redundant(corr 0.73~0.83). item-item 계열 동일 버킷 추정 | ⏸ 저가치 (보류) |

## 핵심 통찰: 이 대회의 구조적 특수성

탐색된 2025-2026 SOTA(XSimGCL/diffusion/LLM-rerank)는 대부분 **대규모 희소** 그래프 +
text/sequence 신호 전제. 이 대회는 정반대:
- **소규모·상대조밀** (6710u × 2437i, 165k nnz)
- **유저당 median 4 후보의 1:1 균형 reranking** (test 50/50)
- **test에 텍스트/시퀀스/날짜 없음** (userID, gameID만)
- public test는 **uniform 분포 추종** (popularity 디바이싱이 오히려 해로움)

→ 대규모-희소-텍스트 전제의 SOTA가 이 소규모-조밀-익명 reranking 문제에 직접 전이될
구조적 이유가 약하다. 이것이 모든 직교축이 uniform 게이트를 통과 못 한 근본 배경.

## XSimGCL 정직한 기대치

- prior LOW: SGL(0.51~0.64)·DirectAU(0.55~0.60) 실패의 진짜 원인이 edge-drop이 아니라
  **InfoNCE가 공유 임베딩을 uniformity로 끌어 median-4 미세 ranking 파괴**라면 XSimGCL도 동일 운명.
- 그래도 실행 가치: edge-drop 변수를 깨끗이 분리해 **contrastive-CF 축을 추측이 아닌 증거로 종료**.
  단일 forward pass라 SGL보다 저렴(~35분/seed).
- 게이트: emb128 4-seed 0.76505를 noise(0.0007) 이상 초과해야 채택. solo가 floor 0.684 미만이면 즉시 reject.

## 지속 탐색 방침

새 방향은 "강+직교" 단일 모델축이 아니라, 이 대회 구조(소규모 균형 reranking, uniform test)에
**구조적으로 맞는** 방법이어야 함. 다음 라운드 후보: (1) XSimGCL 결과 확정 후 contrastive-CF 종결,
(2) 필요시 RP3β를 redundancy 확인용 1회 저비용 probe, (3) 최신 논문 모니터링 지속.
