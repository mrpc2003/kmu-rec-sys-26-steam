# KMURecSys26 Steam — metadata 승인 후 1차 feasibility

- 작성 시각: 2026-06-13 08:28 KST
- 사용자 응답: `승인`
- 해석 범위: 공개 Steam item metadata 사용 승인으로 보고, **item metadata만** 붙일 수 있는지 먼저 확인
- 금지선 유지: Kaggle submit 없음, `submissions/` 쓰기 없음, full-test candidate CSV 없음, hidden label 없음, Steam 리뷰 수집 없음, Steam user/profile/owned-games 조회 없음

## 1. 현재 gameID 상태

로컬 `train.json`과 `pairs.csv`의 `gameID`는 Steam appid가 아니다.

```text
sample_gameIDs = [
  'g35322304', 'g49368897', 'g73495588', 'g68047320',
  'g51234623', 'g25723374', 'g58025004', 'g69033010'
]
```

`g` 뒤 숫자를 Steam Store API의 appid 후보로 넣어 봤지만, 전부 `success=false`였다.

```text
35322304 -> false
49368897 -> false
73495588 -> false
68047320 -> false
51234623 -> false
```

데이터 규모도 같은 결론이다.

```text
train rows: 175000
train users: 6710
train games: 2437
numeric_game_ids: 0
pairs rows: 19998
pairs users: 4737
pairs games: 2429
```

## 2. 판정

item metadata probe는 바로 시작할 수 없다. 필요한 join key가 없다.

가능한 안전 경로는 `gameID -> Steam appid` 매핑 파일이 이미 공개/제공된 경우뿐이다. 현재 repo의 `data/` 안에는 그런 파일이 없다.

아래 방식은 별도 승인이 없으면 진행하지 않는 편이 맞다.

- 제공된 `user_id`로 Steam profile/owned-games를 조회해 gameID를 역매핑
- 리뷰 문장 일부를 웹/Steam 리뷰와 대조해 게임명을 역추적
- 외부 Steam 리뷰/사용자 활동 데이터를 새로 긁어와 매칭

이 셋은 item metadata 추가가 아니라 익명화 해제/외부 사용자 데이터/외부 리뷰 매칭 쪽으로 넘어갈 수 있다. 대회 규칙 리스크가 커서 멈췄다.

## 3. 안전 상태

```text
validation_only: true
no_kaggle_submit: true
candidate_csv_written: false
full_test_scored: false
submission_csv_written: false
hidden_labels_used: false
steam_reviews_collected: false
steam_user_profile_or_owned_games_queried: false
external_item_metadata_probe_attempted: appdetails basic lookup for 5 numeric candidates only
```

## 4. 다음 선택

1. 교수자/대회 측에 `gameID -> Steam appid` 매핑 파일이 있는지 묻기
2. 익명화된 gameID의 역매핑 시도를 허용하는지 별도로 묻기
3. 둘 다 불가하면 외부 metadata 축은 닫고 final packaging으로 복귀

현재 추천은 1번이다. 매핑 파일 없이 억지로 역매핑을 시작하면, 성능 개선보다 규칙 리스크가 먼저 커진다.
