# KMURecSys26 Steam — OpenCode 재가동 후 후속 판단

- 작성 시각: 2026-06-12 22:35 KST
- 작업 경로: `/opt/data/kaggle/kmu-rec-sys-26-steam`
- 범위: OpenCode 수리 뒤 no-submit 개선축 재검토
- 안전 상태: Kaggle 제출 없음, full-test 후보 CSV 생성 없음, hidden label 사용 없음, 외부 Steam 수집 없음

## 1. 이번에 한 일

OpenCode 플러그인 문제를 고친 뒤, 같은 repo에서 후속 advisory를 다시 실행했다.

- prompt: `reports/20260612T2230KST_opencode_reopened_axis_followup_prompt.md`
- raw answer: `reports/20260612T2230KST_opencode_reopened_axis_followup_raw_text.md`
- jsonl log: `logs/20260612T2230KST_opencode_reopened_axis_followup.jsonl`
- sentinel: `OPENCODE_FOLLOWUP_AXIS_ADVISORY_DONE` 확인
- OpenCode route: `Hephaestus - Deep Agent` / `openai/gpt-5.5`

OpenCode 로그에서 플러그인은 local wrapper로 정상 로드됐다.

```text
service=plugin path=file:///opt/data/home/.config/opencode/plugins/oh-my-openagent-local.js loading plugin
service=llm providerID=openai modelID=gpt-5.5 ... agent=Hephaestus - Deep Agent
```

## 2. 후속 advisory 판정

OpenCode의 판정은 `NO_SAFE_AXIS`였다. 의미는 “제공 데이터만 쓰는 로컬 실험에서 바로 strict candidate로 키울 만한 새 축이 없다”는 쪽이다.

추천된 단 하나의 다음 축은 **Steam item metadata side-information**이었다.

- 장르, 태그, 카테고리, 출시일 같은 item metadata를 `gameID`에 붙인다.
- 유저별 train positive item metadata 평균으로 user content profile을 만든다.
- validation candidate pair를 user profile과 item vector의 cosine similarity로 점수화한다.
- 기존 emb128/rankblend validation score와 z/rank blend를 비교한다.
- 단, 교수자 승인 전에는 수집도 학습도 하지 않는다.

이 축이 중요한 이유는 기존 실패축과 성격이 다르기 때문이다. 지금까지 닫힌 축은 대부분 interaction graph, item-item graph filter, LightGCN score residual, seed/loss/blend retune 쪽이었다. metadata는 외부 side information이라, 0.78795와의 차이를 설명할 수 있는 거의 유일한 새 정보 축이다.

## 3. failed-axis ledger 반영

이번에 끝난 두 no-submit 재감사 결과를 `reports/failed_axes.json`에 추가했다.

추가한 항목:

1. `uniform_wide_ease_itemknn_audit_20260612T213950KST`
   - best: `score_itemknn_bm25_sum`
   - mean accuracy: `0.742849`
   - 판정: LightGCN backbone보다 너무 낮아 item-item CF 재튜닝 축 닫음

2. `gfcf_spectral_panel_20260612T214616KST`
   - best solo: `gfcf_P_plus_g0.3_idl_k16`, mean `0.757285`
   - best blend50: `gfcf_P_plus_g0.5_idl_k16`, mean `0.763786`, min `0.761552`
   - 판정: seed42 smoke의 near-tie가 3-split에서 유지되지 않음. GF-CF/BSPM/PPR/heat-kernel류 item-item spectral filter 축 닫음

JSON parse 검증은 통과했다.

## 4. 안전 검증

후속 OpenCode 실행 뒤 확인한 내용이다.

- 새 `submissions/*.csv`: `0개`
- live Kaggle submit-like process: `0개`
- sentinel: 확인
- safety flags:
  - `validation_only: true`
  - `candidate_csv_written: false`
  - `kaggle_submit_executed: false`
  - `hidden_labels_used: false`
  - `external_scraping_used: false`

## 5. 지금 막힌 지점

이제 막힌 곳은 실행 문제가 아니다. OpenCode도 다시 정상이고, no-submit audit도 돌아간다.

문제는 남은 로컬 신호가 거의 다 interaction 기반이라는 데 있다. LightGCN/rankblend가 이미 그 부분을 강하게 먹고 있고, 뒤늦게 붙인 EASE, ItemKNN, GF-CF, ALS, OTTO, TAG-CF, boundary, hours, DNS, UserKNN 계열이 current best를 안정적으로 넘지 못했다.

그래서 다음 분기점은 하나다.

> Steam metadata를 써도 되는지 확인해야 한다.

승인이 나면 metadata-only validation probe부터 20개 uniform panel에서 바로 돌릴 수 있다. 승인이 안 나면 현재 상태에서는 final-2 packaging과 eCampus 재현성 정리로 돌아가는 쪽이 안전하다.

## 6. 교수자에게 확인할 문장 초안

아래처럼 물어보면 된다.

```text
Kaggle Steam played prediction 과제에서 제공된 train/test의 userID, gameID, text, date, hours 외에,
각 gameID가 실제 Steam appid에 대응한다고 가정하고 공개 Steam 상점/SteamSpy/공개 메타데이터에서
게임 장르, 태그, 카테고리, 출시일 같은 item metadata를 추가로 수집해 feature로 사용하는 것이 허용되나요?
리뷰 본문이나 숨은 정답을 수집하려는 목적은 아니고, gameID별 공개 item side information만 쓰려는 목적입니다.
```

승인 여부가 애매하면 사용하지 않는 쪽이 맞다.
