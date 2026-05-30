# KMU RecSys 26 Steam — Initial Data Profile
## Competition pages/API snapshot
- Task: Steam review data 기반 user-game played 여부 이진 분류.
- Public metric: Accuracy. Public LB는 test 절반 기준, private/final은 전체 test 기준.
- Key structure: test pairs are per-user positive:negative = 1:1; per-user ranking/top-half strategy is central.
- Rule reminders: no external Steam answer collection/reverse engineering; code/answers not shared; final selected submissions must be reproducible for eCampus; public pre-trained models allowed.

## Files
- `data/kmu-rec-sys-26-steam.zip` SHA256 `8b741ac2ef54ec403b2ec3c5bbdd74ae931c9ecce9abca857dd2ac79cd2cb572`
- `data/raw/public/data/train.json` SHA256 `2258fa550ae4835fe041d54bcbae4337c3bed8774396dee9ceafbe6a3041daa2`
- `data/raw/public/data/pairs.csv` SHA256 `58678f4a851709bc130a5c3f2a0a3a622ae8b41d74d255c7c517537a426c2bae`
- `data/raw/public/README.md` SHA256 `039a986734b47097be4cf0eea03ad3a8ce2adc2eaa56c920cbd3016c52f36576`
- `data/raw/public/baseline.py` SHA256 `300f93f3e5e429b2a8356f898ff1a10043011f8944242d8eca470c3a2c4c2448`
- `data/raw/public/baseline_bpr.py` SHA256 `ff3d44f32fedfdb24f9393a5f835ffb8f7f1c96e54a5dda5fe3afa4041916268`

## Train summary
- Records: 175,000
- Unique users/games: 6,710 users / 2,437 games
- Unique user-game pairs: 175,000; duplicate rows: 0
- Matrix density: 1.070188%
- Date range: 2010-10-15 ~ 2018-01-05
- Fields: {'hours': 175000, 'gameID': 175000, 'hours_transformed': 175000, 'early_access': 175000, 'date': 175000, 'text': 175000, 'userID': 175000, 'user_id': 54841, 'found_funny': 29938, 'compensation': 2870}
- Early access counts: {'False': 158332, 'True': 16668}
- Optional user_id present: 54,841; found_funny present: 29,938; found_helpful present: 0
- Hours median/mean/p95/max: 10.100 / 66.408 / 258.510 / 16539.900
- hours_transformed median/mean/p95/max: 3.472 / 3.718 / 8.020 / 14.014
- Text length median/mean/p95/max: 130.0 / 391.0 / 1637.0 / 8,000; blank rows: 838
- User degree median/mean/p95/max: 21.0 / 26.08 / 60.0 / 232
- Game degree median/mean/p95/max: 38.0 / 71.81 / 246.6 / 1092
- Popularity coverage:
  - 25% interactions covered by top 109 games (4.47% of games)
  - 50% interactions covered by top 351 games (14.40% of games)
  - 75% interactions covered by top 887 games (36.40% of games)
  - 90% interactions covered by top 1,568 games (64.34% of games)
  - 95% interactions covered by top 1,926 games (79.03% of games)
- Game popularity Gini: 0.5214
- Top games by train count:
  - `g10773791`: 1,092
  - `g05463839`: 943
  - `g75228197`: 897
  - `g40499587`: 794
  - `g11862712`: 746
  - `g89200271`: 724
  - `g15881340`: 646
  - `g29741733`: 633
  - `g89492775`: 613
  - `g77806076`: 606

## Pairs/test summary
- CSV rows parsed: 19,998; raw lines incl. header/blanks: 19,999; blank lines: 0
- ID range: 0..19997; contiguous 0-based: True; duplicate IDs: 0
- Unique users/games in pairs: 4,737 / 2,429
- Cold users/games vs train: 0 / 0
- Candidate rows with game unseen in train: 0
- Candidate user-game pairs already in train: 0
- Candidate count per user median/mean/p95/max: 4.0 / 4.22 / 10.0 / 38
- Users with odd candidate counts: 0
- If using floor(top-half) per user: label-1 total 9,999 (50.00%)
- Candidate user train-degree median/mean/p95/max: 27.0 / 39.08 / 104.0 / 232
- Candidate game train-popularity median/mean/p95/max: 62.0 / 129.46 / 483.0 / 1092
- Top candidate games by pair frequency:
  - `g10773791`: 70
  - `g05463839`: 63
  - `g11862712`: 60
  - `g75228197`: 50
  - `g40499587`: 50
  - `g82668863`: 44
  - `g89492775`: 44
  - `g89200271`: 44
  - `g68047320`: 40
  - `g15881340`: 40

## Immediate modeling implications
1. Submission must be `ID,Label`, labels in {0,1}; current downloaded file is `pairs.csv`, while Kaggle page/baseline text also mentions `pairs_Played.csv`, so local code should normalize this filename mismatch.
2. Because all pair users/items are expected to be mostly known, start with reproducible per-user rankers rather than global thresholding.
3. Popularity is a very strong axis; every CF/graph/text feature should be evaluated both alone and as a hybrid with calibrated popularity.
4. Build validation as user-wise leave-one-out positive + sampled negatives, then score by per-user ranking/top-half accuracy. Add multiple negative samplers to reduce public/private mismatch.
5. Candidate model ladder: popularity/top-half variants → BPR/ALS item2item hybrids → LightGCN → text/hours/date/user-history features → rank-ensemble. No submission before explicit user approval.
