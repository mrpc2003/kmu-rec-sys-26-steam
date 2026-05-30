# KMU RecSys 26 Steam — Validation Harness + Baseline/EASE/ItemKNN Prototype Report

작성 시각: 2026-05-30 KST.  
범위: validation harness, baseline 재현, EASE/item-item prototype.  
중요: Kaggle 제출은 수행하지 않았다.

## 1. 구현 파일

- `scripts/recsys_played_utils.py`
  - train/pairs loader, CSR matrix builder, per-user top-half prediction/evaluation, README popularity baseline reproduction helper.
- `scripts/build_validation_splits.py`
  - actual `pairs.csv`의 user별 candidate count를 기준으로 validation candidates 생성.
  - holdout: `random`, `recent`.
  - negative sampler: `uniform`, `sqrtpop`, `popbin`.
  - fold-train overlap, unknown user/item, per-user 1:1 candidate 구조를 safety check로 강제.
- `scripts/score_popularity_itemknn_ease.py`
  - popularity, item-item cosine/co-occurrence, hours-weighted itemKNN, EASE score 생성.
  - validation이면 per-user top-half Accuracy 계산.
  - test pairs면 no-submit candidate prediction CSV 생성.
- `scripts/evaluate_tophalf.py`
  - 임의 score CSV에 대해 per-user top-half rule로 재평가.

## 2. 생성한 validation splits

명령:

```bash
uv run --with numpy --with pandas --with scipy python scripts/build_validation_splits.py \
  --configs random:sqrtpop:42,random:uniform:42,random:popbin:42,recent:sqrtpop:42 \
  --out-root artifacts/validation
```

생성 결과:

| split | holdout | negative | rows | users | positive rows | skipped users | safety |
|---|---|---|---:|---:|---:|---:|---|
| `val_random_sqrtpop_seed42` | random | sqrtpop | 19,996 | 4,736 | 9,998 | 1 | no overlap, known users/items, even counts |
| `val_random_uniform_seed42` | random | uniform | 19,996 | 4,736 | 9,998 | 1 | no overlap, known users/items, even counts |
| `val_random_popbin_seed42` | random | popbin | 19,996 | 4,736 | 9,998 | 1 | no overlap, known users/items, even counts |
| `val_recent_sqrtpop_seed42` | recent | sqrtpop | 19,996 | 4,736 | 9,998 | 1 | no overlap, known users/items, even counts |

한 user `u57101927`은 train interaction이 1개라 holdout 후 known-user 조건을 유지할 수 없어 skip했다. 따라서 validation rows는 실제 test 19,998보다 2개 작다.

`sqrtpop` split의 negative item popularity median은 59로, fold-train 기준 actual pairs item popularity median 58과 거의 맞는다. uniform split median은 35로 더 쉽다.

## 3. Baseline 재현

제공 baseline 파일을 no-submit으로 실행했다.

명령 요약:

```bash
cd data/raw/public
cp data/pairs.csv data/pairs_Played.csv
uv run --with numpy --with scipy --with implicit python baseline.py
uv run --with numpy --with scipy --with implicit python baseline_bpr.py
```

결과:

| file | rows | label=1 | SHA256 |
|---|---:|---:|---|
| `data/raw/public/data/played_baseline.csv` | 19,998 | 6,289 | `2183fef4dc7a5194bd2458ca5d73f27424becd19bb0c903e219e2dd044bc99bb` |
| `data/raw/public/data/played_bpr.csv` | 19,998 | 9,999 | `03a8783320898fc66deace68fb7b2e1e80494ea9319ac9c79b0c47ab3806dc04` |

README raw popularity baseline의 Counter tie-order까지 맞추도록 `readme_popularity_labels()`를 수정했고, prototype에서 생성한 `candidate_readme_popularity_raw.csv`가 제공 `played_baseline.csv`와 동일 SHA로 일치함을 확인했다.

Kaggle public reference:

- README popularity baseline: public 0.68413.
- BPR+popularity per-user baseline: public 0.72194.

## 4. Prototype validation scores

각 split에서 `popularity`, `itemKNN`, `EASE(lambda=100,300,1000)`, 간단 z-score blends를 평가했다. 점수는 모두 per-user top-half rule 적용 후 row Accuracy / per-user mean Accuracy다.

| split | best score column | row acc | user mean acc | README raw baseline row acc |
|---|---|---:|---:|---:|
| `val_random_uniform_seed42` | `score_itemknn_sum` | 0.740648 | 0.764331 | 0.682737 |
| `val_random_sqrtpop_seed42` | `score_ease_lambda1000` | 0.646329 | 0.660149 | 0.607221 |
| `val_recent_sqrtpop_seed42` | `score_itemknn_top3` | 0.619424 | 0.629942 | 0.578616 |
| `val_random_popbin_seed42` | `score_ease_lambda300` | 0.563013 | 0.575285 | 0.504651 |

해석:

1. uniform negative split은 너무 쉽다. README raw baseline이 0.6827로 public 0.68413과 비슷해 보이지만, 이는 negative가 쉬워서 popularity가 유리한 환경이다.
2. sqrtpop split은 실제 candidate item popularity median과 잘 맞아 primary local gate로 쓰기 좋다.
3. popbin split은 positive와 negative popularity를 강하게 맞춘 hard split이라 전체 점수가 낮다. popularity만으로는 거의 무력화되므로 모델의 personalized ranking 능력을 보는 stress split으로 유용하다.
4. EASE는 sqrtpop/popbin에서 가장 안정적인 상위권이다. itemKNN은 uniform/recent에서 강하다.

## 5. Full test-pair no-submit candidate predictions

명령:

```bash
uv run --with numpy --with pandas --with scipy python scripts/score_popularity_itemknn_ease.py \
  --data-dir data/raw/public/data \
  --methods popularity,itemknn,ease \
  --ease-lambdas 1000 \
  --out-dir artifacts/scores/test_pairs_full_train_proto \
  --write-predictions
```

출력 위치:

- `artifacts/scores/test_pairs_full_train_proto/candidate_scores.csv`
- `artifacts/scores/test_pairs_full_train_proto/prediction_csv/*.csv`

검증:

- 모든 generated candidate CSV는 `ID,Label` header, 19,998 rows, contiguous ID, labels {0,1} 조건을 만족한다.
- per-user top-half 기반 candidate files는 label=1 총합이 9,999다.
- README raw popularity reproduction만 label=1 총합이 6,289이며, 제공 baseline과 동일하다.

주의: 이 파일들은 후보 산출물일 뿐이며 Kaggle 제출하지 않았다.

## 6. OpenCode review

OpenCode `openai/gpt-5.5`로 read-only correctness review를 실행했다.

- 로그: `reports/opencode_reviews/20260530_validation_harness_review_direct.jsonl`
- 주요 finding:
  - blocking leakage 없음.
  - per-user top-half evaluation, EASE formula, README baseline reproduction은 타당.
  - potential high issue: holdout positive game이 fold-train에서 사라질 경우 unknown item candidate가 될 수 있음.
- 대응:
  - 실제 4개 split을 검사했을 때 missing candidate item/user rows는 모두 0이었다.
  - 그래도 `build_validation_splits.py`에 fold-train known-user/known-item safety check를 추가해 앞으로는 즉시 실패하도록 보강했다.

## 7. Verification log

실행/검증 완료:

- `python3 -m py_compile` for all new scripts: PASS.
- validation split build: PASS.
- per-user 1:1/even candidate safety: PASS.
- fold-train overlap: 0.
- missing user/item vs fold train: 0.
- scoring/evaluation smoke: PASS.
- README baseline SHA reproduction: PASS.
- no Kaggle submit command / credential pattern scan: PASS.

## 8. 다음 단계 제안

1. `score_popularity_itemknn_ease.py`를 확장해 BM25/TF-IDF-normalized itemKNN과 EASE BM25 matrix를 추가한다.
2. BPR baseline도 같은 validation harness에서 fold별로 평가하도록 별도 scorer를 만든다.
3. sqrtpop split을 primary, popbin/recent를 stress로 두고 model acceptance gate를 정의한다.
4. 이후 ALS/BPR/EASE/itemKNN score files를 모두 candidate row order 기준으로 저장하고, rank/z-score/RRF blend를 탐색한다.
5. Kaggle 제출은 local gate 결과를 보고 우현 승인 후에만 진행한다.
