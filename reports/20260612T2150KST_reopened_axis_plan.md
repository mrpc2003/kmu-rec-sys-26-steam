# KMURecSys26 Steam — 0.78795 LB 확인 후 재개방 계획

- 작성 시각: 2026-06-12 21:50 KST
- 범위: leaderboard 재확인, rules 확인, 기존 “종결” 판단 수정, 재개방 실험 시작 상태

## 1. 전제 수정

Kaggle CLI로 현재 leaderboard를 직접 확인했다.

```text
1위: 김준형(학부생-인공지능전공) 0.78795
2위: cho hyeon chan 0.78095
3위: SEO_DOZ 0.78085
4위: mrpc2003 0.77825
```

따라서 이전 문서의 “Bayes ceiling 근접” 표현은 너무 강했다. 0.77825와 0.78795의 차이는 0.00970, 19,998행 기준 약 194행이다. 이 정도 차이는 public noise나 micro-delta가 아니라, 아직 우리가 쓰지 못한 독립 신호가 존재한다는 쪽으로 보는 게 맞다.

## 2. rules 확인

Kaggle API의 competition pages에서 rules 내용을 확인했다.

- Steam 리뷰 직접 수집 또는 리버스엔지니어링으로 정답을 외부에서 획득하는 행위는 금지.
- 친구와 아이디어 논의는 가능하지만 코드/정답 공유 금지.
- 최종 2개 제출 선택 가능.
- 선택한 제출은 동일 결과를 재현하는 코드와 조건을 eCampus로 제출해야 함.
- 온라인에 공개되어 누구나 사용할 수 있는 pretrained model은 사용 가능.

해석상, **외부 Steam 메타데이터 수집은 아직 허용으로 단정하면 안 된다.** rules 문구가 “Steam 리뷰 직접 수집”과 “정답 외부 획득”을 직접 금지하고 있으므로, gameID가 appid라고 해도 장르/태그/출시일 수집은 교수자 확인 전에는 쓰지 않는 편이 안전하다.

## 3. 바로 시작한 작업

### 3.1 validation 해상도 보강

새 uniform panel 20개를 생성했다.

- 경로: `artifacts/validation_uniform_panel20_20260612T214626KST`
- split 수: 20
- 각 split rows: 19,996
- safety: fold-train overlap 0, missing user/item 0, per-user 1:1 유지

이 패널은 작은 delta를 더 안정적으로 볼 때 쓰기 위한 기반이다. 아직 LightGCN/새 모델 score가 모두 준비된 것은 아니므로, 당장은 low-cost CF/spectral 계열부터 여기에 얹는 방향이 맞다.

### 3.2 wide EASE/ItemKNN uniform audit

기존 “EASE/ALS/ItemKNN은 약하다” 판단이 sqrtpop 시절 평가에 끌려갔을 가능성을 확인하기 위해, uniform 3-split에서 EASE λ와 ItemKNN weighting을 넓혀 다시 돌리고 있다.

- background process: `proc_ef9668ab6010`
- run root: `artifacts/scores/20260612T213950KST_uniform_wide_ease_itemknn`
- methods: popularity, itemknn, itemknn_bm25, itemknn_tfidf, ease, ease_htr
- EASE λ: 10, 30, 100, 300, 1000, 3000, 10000
- safety: validation-only, no Kaggle submit, no full-test candidate

중간 결과상 seed42/seed7에서는 best가 여전히 ItemKNN BM25 sum 계열이며 0.742~0.743대다. LightGCN 0.765권과는 아직 거리가 있다. seed123까지 끝나야 최종 판정 가능하다.

### 3.3 GF-CF spectral probe

이전 Turbo-CF는 polynomial/linear graph filtering 쪽에 가까웠고, decomposition-based GF-CF의 ideal low-pass filter를 깊게 보지는 않았다. 그래서 새 validation-only script를 추가했다.

- script: `scripts/gfcf_uniform_panel_probe.py`
- smoke report: `reports/gfcf_uniform_panel_probe_smoke_20260612.md`
- full 3-split background process: `proc_70b2b7d5b932`
- run root: `artifacts/20260612T214503KST_gfcf_uniform_panel_probe`
- safety: validation-only, no Kaggle submit, no full-test candidate

smoke 기준 seed42에서는 GF-CF variant solo가 최대 0.75245, emb128 reference와의 50/50 blend가 최대 0.765053으로, 거의 기준선과 동률이었다. 3-split 결과가 필요하지만 “대형 +0.01 축”처럼 보이진 않는다.

## 4. 다음 판단 기준

1. EASE/ItemKNN wide audit과 GF-CF 3-split이 끝나면, 기존 LightGCN 기준과 비교한다.
2. 둘 다 0.765권을 명확히 넘지 못하면, training-free CF/spectral 계열은 “leaderboard gap의 주 원인 아님”으로 낮춘다.
3. 그 다음은 loss upgrade 쪽이다.
   - SimGCL/XSimGCL 정식 sweep이 정말 충분했는지 재감사.
   - InfoNCE/sampled-softmax, τ/noise/epoch sweep을 public surrogate 기준으로 다시 볼지 결정.
4. hours는 residual feature가 아니라 edge confidence/loss weighting으로 재검토한다.
5. 외부 Steam metadata는 rules상 회색지대라, 확인 전까지 실험에 넣지 않는다.

## 5. 현재 임시 결론

기존 “닫는 게 정답” 판단은 현재 LB 0.78795를 반영하면 철회해야 한다. 다만 바로 제출을 태울 후보가 생긴 것은 아니다. 지금은 **빠진 축을 찾는 단계로 되돌리되, 기존 quarantine family를 그대로 반복하지 않고 validation-only 저비용 축부터 닫는 상태**다.
