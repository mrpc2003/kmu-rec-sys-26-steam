# KMURecSys26 Steam — 재개방 축 실행 결과와 다음 계획

- 작성 시각: 2026-06-12 22:10 KST
- 작업 경로: `/opt/data/kaggle/kmu-rec-sys-26-steam`
- 범위: 0.78795 리더보드 확인 뒤 재개방한 no-submit 축 정리
- 안전 상태: Kaggle 제출 없음, full-test 후보 CSV 생성 없음, hidden label 사용 없음, 외부 Steam 수집 없음

## 1. 왜 다시 열었는가

이전 정리는 “우리 모델군 안에서 신호가 거의 포화됐다”는 쪽이었다. 그런데 Kaggle leaderboard를 다시 확인하니 1위가 `0.78795`였다. 우리 current best `0.77825`와 차이가 `+0.00970`이다. 19,998행 기준으로는 약 194행 규모라 단순 public noise라고 보기 어렵다.

그래서 판단을 이렇게 고쳤다.

- “전체 문제의 Bayes ceiling에 거의 닿았다”는 표현은 과했다.
- 현재 포화는 LightGCN/co-occurrence 계열 내부 포화일 가능성이 크다.
- 다만 새 제출을 바로 만들지는 않는다. 먼저 no-submit 검증으로 닫힌 축을 다시 확인한다.

## 2. 이번에 끝난 실행

### 2.1 EASE/ItemKNN wide audit

- 산출물: `reports/20260612T213950KST_uniform_wide_ease_itemknn_aggregate.md`
- run root: `artifacts/scores/20260612T213950KST_uniform_wide_ease_itemknn`
- splits: `val_random_uniform_seed42`, `val_random_uniform_seed7`, `val_random_uniform_seed123`
- methods: popularity, ItemKNN, BM25 ItemKNN, TF-IDF ItemKNN, EASE, EASE-HTR
- EASE lambda: 10, 30, 100, 300, 1000, 3000, 10000

결과는 명확히 음성이다.

- best: `score_itemknn_bm25_sum`
- mean acc: `0.742849`
- min acc: `0.742649`
- max acc: `0.743049`

emb128 4-seed uniform 기준선이 약 `0.76505`라서, best ItemKNN BM25도 약 `-0.0222` 아래다. EASE best도 `score_ease_lambda3000` mean `0.734980`에 그쳤다. 이 정도면 EASE/ItemKNN 재튜닝으로 0.78795 gap을 설명하기 어렵다.

### 2.2 GF-CF spectral panel

- 산출물: `reports/20260612T214616KST_gfcf_uniform_panel_probe.md`
- run root: `artifacts/20260612T214616KST_gfcf_uniform_panel_probe`
- splits: `val_random_uniform_seed42`, `val_random_uniform_seed7`, `val_random_uniform_seed123`
- variants: linear P, ideal low-pass, `P + gamma * idl(k)`
- k: 16, 32, 64, 128, 256, 512
- gamma: 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0

smoke에서는 seed42 한 split에서 blend가 emb128 기준선과 거의 동률까지 왔다. 3-split으로 늘리니 그 신호가 유지되지 않았다.

- best solo mean: `gfcf_P_plus_g0.3_idl_k16`, mean acc `0.757285`, min acc `0.755851`
- best blend50 mean: `gfcf_P_plus_g0.5_idl_k16`, mean blend50 acc `0.763786`, min blend50 acc `0.761552`
- mean corr_z vs emb128 ref: 대략 `0.89`

해석은 다음과 같다.

- GF-CF는 ItemKNN/EASE보다 강하지만, emb128 4-seed 기준선 `0.76505`를 넘지 못했다.
- 50/50 blend도 평균 `0.763786`이라 기준선보다 낮다.
- corr_z가 높아서, 남는 신호도 LightGCN과 많이 겹친다.
- seed42 smoke의 동률은 단일 split 착시였고, 3-split에서는 candidate 축이 아니다.

### 2.3 OpenCode advisory 시도

OpenCode adviser도 붙이려 했지만, 현재 OpenCode 호출이 provider/server 쪽 `Unexpected server error`로 실패했다.

- prompt: `reports/20260612T2202KST_opencode_reopened_axis_advisory_prompt.md`
- log: `logs/20260612T2202KST_opencode_reopened_axis_advisory.jsonl`
- 실패 메시지: `Unexpected server error. Check server logs for details.`
- plain smoke와 Vertex Flash 지정 smoke도 같은 유형으로 실패

이건 실험 blocker는 아니다. 이번 판단은 repo의 실제 산출물과 Hermes 직접 실행 결과로 냈다.

## 3. CF/spectral 재감사 판정

이번 재개방으로 확인한 것은 아래와 같다.

- EASE: 재튜닝해도 weak, LightGCN과 격차 큼.
- ItemKNN: BM25가 제일 낫지만 mean `0.742849`로 부족.
- Turbo-CF: 기존 report 기준 solo `0.74155`, blend가 emb128보다 `-0.00680`.
- GF-CF: 이번 3-split 기준 best blend mean `0.763786`, emb128 기준선 미달.
- ALS/WMF: 기존 독립 확인에서 weak-positive였지만 strict gate fail. pre-registered row mean `+0.000800`, min `-0.000300`, 2/3 positive, p `0.129`.

따라서 training-free CF/spectral/item-item 계열은 0.78795 gap의 주된 설명이 아니다. “안 해본 GF-CF decomposition이 남아 있다”는 의심은 이번 3-split 결과로 닫았다.

## 4. 지금 막힌 지점

막힌 지점은 실행 문제가 아니다. 실험은 돌아가고 결과도 나왔다. 막힌 곳은 신호 쪽이다.

- 제공 데이터만 쓰는 graph/co-occurrence 계열은 LightGCN이 대부분 흡수한다.
- item-item closed-form 계열은 강해져도 LightGCN과 겹치고, 독립 blend가 기준선을 못 넘는다.
- hours/date/text/popularity residual은 이전 실험에서 약하거나 pop-trap으로 바뀌었다.
- 0.78795와의 차이는 아직 설명되지 않았다.

남은 가능성은 셋뿐이다.

1. 1위가 우리가 아직 구현하지 않은 전혀 다른 합법 모델/후처리 축을 썼다.
2. 외부 metadata나 과제 해석 차이를 이용했을 수 있다. 다만 이건 교수자 확인 전에는 쓰지 않는 게 맞다.
3. public half에서 운이 좋게 튄 후보일 수 있다. 그래도 +0.00970은 커서 이것만으로 치부하면 안 된다.

## 5. 다음 실행 계획

### 5.1 바로 할 일

1. 이번 두 probe를 failed-axis ledger에 반영한다.
2. CF/spectral 계열은 더 넓히지 않는다. PPR/heat-kernel/BSPM류도 같은 item-item graph filter에 가까워서 지금 결과를 넘길 확률이 낮다.
3. full-test 후보 CSV는 만들지 않는다.
4. Kaggle submit은 하지 않는다.

### 5.2 다음으로 볼 수 있는 축

우선순위는 아래 순서다.

1. **규칙 확인이 필요한 metadata 축**
   - Steam appid/genre/tag/release-date 같은 외부 metadata가 허용되는지 교수자 확인이 필요하다.
   - 허용되면 0.78795 gap을 설명할 수 있는 거의 유일한 “새 정보” 축이다.
   - 허용 전에는 수집·학습·후보화하지 않는다.

2. **이미 닫힌 loss 축 재반복이 아닌 새 backbone 설계**
   - exact-K, SL@K-lite, DNS, SGL/DirectAU/xSimGCL은 닫혔다.
   - 새로 보려면 “기존 LightGCN 점수와 낮은 상관을 유지하면서 solo가 0.76권에 접근하는” 구조여야 한다.
   - 단순 loss retune, hard-negative retune, seed retune은 0.00970 gap 후보가 아니다.

3. **final-2 fallback 정리**
   - 현재까지 안전한 final fallback은 `candidate_rank_blend_emb128_emb192.csv` public `0.77825`와 emb128 4-seed public `0.77745`다.
   - 새 축이 strict gate를 통과하지 못하면 final packaging/eCampus 재현성 정리로 돌아간다.

## 6. 현재 상태 결론

이번 실행으로 닫힌 것은 “남은 low-cost CF/spectral 축”이다. EASE/ItemKNN wide audit도, GF-CF decomposition panel도 current backbone을 넘지 못했다. 0.78795 gap은 여전히 열려 있지만, 그 답은 item-item graph filter 쪽에는 없다는 쪽으로 좁혀졌다.

다음 분기점은 metadata 허용 여부와, 이미 닫힌 계열을 반복하지 않는 새 backbone 가설이 실제로 있는지다. 제출 후보를 만들거나 Kaggle에 올릴 단계는 아니다.
