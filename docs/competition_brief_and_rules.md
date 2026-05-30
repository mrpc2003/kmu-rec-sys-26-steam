# KMU RecSys 26 Steam — 대회 설명 및 운영 규칙

작성 기준: 사용자가 공유한 Kaggle 대회 설명과 현재 저장소 운영 원칙을 프로젝트 규칙으로 정리한다. 이 문서는 **모델링/검증/제출 의사결정의 기준 문서**로 사용한다.

## 1. 태스크 정의

- 주제: 게임 플랫폼 **Steam 리뷰 데이터**를 이용한 추천 시스템 설계.
- 목표: 주어진 `userID, gameID` pair에 대해 해당 사용자가 해당 게임을 **플레이했는지 여부**를 예측한다.
- 출력 라벨:
  - `1`: 해당 user가 해당 game을 플레이했다고 예측.
  - `0`: 플레이하지 않았다고 예측.
- 평가 지표: 전체 pair 중 예측이 맞은 비율인 **Accuracy**.
- 테스트셋 구성: 플레이한 경우와 플레이하지 않은 경우가 **정확히 절반씩** 포함된다.
- Leaderboard:
  - 제출 직후 공개되는 Public Leaderboard 점수는 테스트셋 중 **절반**에 대한 accuracy.
  - 전체 테스트셋 accuracy는 대회 종료 후 공개된다.

## 2. 제공 데이터와 파일

### `data/train.json`

- 규모: 약 **175,000건**의 Steam 리뷰 인스턴스.
- 형식: JSON.
- 각 인스턴스 주요 필드:
  - `userID`: user 식별값.
  - `gameID`: game 식별값.
  - `text`: user가 game에 남긴 리뷰 본문.
  - `date`: 리뷰 입력 날짜.
  - `hours`: 이 user가 해당 game을 플레이한 총 시간.
  - `hours_transformed`: `log2(1 + hours)`.
- 모든 정보를 반드시 사용할 필요는 없지만, feature engineering 후보로는 user/game interaction, 리뷰 텍스트, 날짜, 플레이 시간 정보를 모두 검토할 수 있다.

### `data/pairs.csv` / `pairs_Played.csv`

- 각 행에 `userID, gameID` pair가 들어 있다.
- 해당 user가 해당 game을 플레이했을지를 예측한다.
- 대회 설명과 baseline 문서에서 `pairs_Played.csv`라는 이름이 언급될 수 있으나, 현재 로컬 Kaggle 파일명은 `data/pairs.csv`이다.
- 스크립트는 가능하면 `pairs.csv`를 기본으로 사용하고, 원본 baseline 재현처럼 `pairs_Played.csv`가 필요한 경우에만 로컬에서 복사/심볼릭 링크로 처리한다. 이 복사본은 Git에 올리지 않는다.

### Baseline 코드

- `baseline.py`: user와 무관하게 가장 많이 플레이된 게임을 추천하는 popularity baseline.
- `baseline_bpr.py`: popularity baseline과 BPR(Bayesian Personalized Ranking)을 함께 사용하는 baseline.
- 원본 `README.md`: baseline 실행 설명과 점수 향상 팁 포함.

## 3. 제출 형식 및 검증 규칙

- 제출 CSV는 `ID,Label` 형식을 따른다.
- `Label`은 반드시 `{0, 1}` 중 하나여야 한다.
- 현재 테스트셋의 hidden label 분포가 1:1이므로, 제출 후보는 최소한 다음을 확인한다.
  - 전체 `Label=1` 수와 `Label=0` 수가 테스트셋 구조와 일치하는지.
  - user별 후보군이 짝수일 때 top-half ranking 변환이 깨지지 않았는지.
  - `ID`가 `0..N-1` 연속인지.
  - 후보 CSV SHA256, 생성 명령, 사용 코드/commit, seed, 환경 조건을 기록했는지.
- Kaggle 제출은 사용자 명시 승인 후 **한 파일씩** 수행한다.
- 제출 전에는 preflight 리포트를 남기고, 제출 후에는 Public 점수와 Kaggle status를 별도 리포트로 남긴다.

## 4. 외부 데이터/윤리 규칙

- Steam 리뷰를 직접 수집하거나, 리버스엔지니어링 등으로 정답을 외부에서 획득하지 않는다.
- Kaggle이 제공한 데이터, 공개적으로 허용된 baseline/문서, 온라인에 공개되어 누구나 사용할 수 있는 pre-trained 모델만 사용한다.
- 외부 지식이나 pretrained embedding/model을 쓰는 경우, 사용 가능성·라이선스·재현 절차를 리포트에 남긴다.
- 부정행위로 판단될 수 있는 데이터 누수, hidden label 복원, 외부 정답 매칭, private test 역추적 시도는 금지한다.

## 5. 최종 제출/재현성 규칙

- 제출한 예측 결과 중 최종 **2개**를 선택할 수 있다.
- 선택한 최종 제출에 대해서는 동일한 결과가 재현되는 **코드, 조건, 실행 절차**를 eCampus에 반드시 제출해야 한다.
- 제출 결과가 복원되지 않으면 제출로 인정되지 않을 수 있다.
- 부정행위로 판단되는 경우 다른 제출과 관계없이 0점 처리될 수 있다.
- 따라서 final 후보는 다음 정보를 반드시 보존한다.
  - 후보 CSV 경로와 SHA256.
  - 생성 스크립트/명령어.
  - Python/패키지/OS/GPU 사용 여부 등 실행 환경.
  - train/pairs 원본 데이터 fingerprint.
  - 랜덤 seed와 split/negative sampling 조건.
  - 후처리/blending/top-half 변환 규칙.
  - 제출 당시 Git commit 또는 재현 가능한 patch 상태.

## 6. 현재 프로젝트 운영 해석

- 이 대회는 단순 score 제출 경쟁이 아니라, 최종 eCampus 재현성을 요구하는 과제형 Kaggle 대회다.
- Public LB는 테스트셋 절반만 반영하므로, Public 점수만 보고 같은 계열을 반복 제출하지 않는다.
- 현재 프로젝트는 다음 우선순위를 따른다.
  1. 제공 데이터 기반 EDA와 validation surrogate 품질 개선.
  2. user별 후보군/top-half 구조를 보존하는 ranking 모델 개선.
  3. popularity, BPR, ItemKNN, EASE, ALS, sequence/text/time feature 등 다양한 계열 검토.
  4. W&B/GitHub 리포트로 실험 재현성과 후보 lineage 기록.
  5. Kaggle 제출은 우현 승인 후 제한적으로 수행.
