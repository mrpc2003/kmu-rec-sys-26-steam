# Forced OTTO residual post-submission analysis

- Submitted file: `submissions/candidate_otto_coplay_top5_reverse_recent_w0090_w0040_forced_20260607T114059KST.csv`
- SHA256: `d70af0f1e325c5c59985ed0df6dbc4232950edf6971f0ad26c72f7da8205985e`
- Status: `SubmissionStatus.COMPLETE`
- Public score: **0.77815**
- Current live best before submit: **0.77825** (`submissions/candidate_rank_blend_emb128_emb192.csv`)
- Δ vs current best: **-0.00010**
- Δ vs emb128 public 0.77745: **+0.00070**
- Independent local mean Δ vs emb128: `+0.0006668000`
- Transfer ratio vs emb128: `1.050`

## Judgment

The rejection/no-escalation decision is confirmed for the live leaderboard: the candidate is a real weak positive relative to the emb128 backbone, but it is not strong enough to beat the current best rank-blend anchor.

## Diff vs current best

- Overall flips: `508` (`2.5403%`)
- Promoted old=0→new=1: `254`
- Demoted old=1→new=0: `254`

### User candidate-count bucket
- `5-6`: flips `122/3990` = `3.0576%`
- `11+`: flips `56/2550` = `2.1961%`
- `2`: flips `68/4282` = `1.5880%`
- `7-10`: flips `124/3948` = `3.1408%`
- `3-4`: flips `138/5228` = `2.6396%`

### Item-popularity quintile
- `Q4`: flips `108/3999` = `2.7007%`, mean_pop `138.46`
- `Q1_low`: flips `84/4000` = `2.1000%`, mean_pop `18.78`
- `Q3`: flips `168/4000` = `4.2000%`, mean_pop `64.21`
- `Q5_high`: flips `32/4000` = `0.8000%`, mean_pop `392.41`
- `Q2`: flips `116/3999` = `2.9007%`, mean_pop `33.39`

- Promoted item_pop mean/median: `90.15354330708661` / `64.0`
- Demoted item_pop mean/median: `66.65354330708661` / `48.0`
- All item_pop mean/median: `129.45599559955997` / `62.0`

## Diff vs emb128 base

- Overall flips: `294` (`1.4701%`)
- Promoted old=0→new=1: `147`
- Demoted old=1→new=0: `147`

### User candidate-count bucket
- `5-6`: flips `78/3990` = `1.9549%`
- `11+`: flips `50/2550` = `1.9608%`
- `2`: flips `0/4282` = `0.0000%`
- `7-10`: flips `74/3948` = `1.8744%`
- `3-4`: flips `92/5228` = `1.7598%`

### Item-popularity quintile
- `Q4`: flips `64/3999` = `1.6004%`, mean_pop `138.46`
- `Q1_low`: flips `52/4000` = `1.3000%`, mean_pop `18.78`
- `Q3`: flips `89/4000` = `2.2250%`, mean_pop `64.21`
- `Q5_high`: flips `31/4000` = `0.7750%`, mean_pop `392.41`
- `Q2`: flips `58/3999` = `1.4504%`, mean_pop `33.39`

- Promoted item_pop mean/median: `115.73469387755102` / `83.0`
- Demoted item_pop mean/median: `61.204081632653065` / `47.0`
- All item_pop mean/median: `129.45599559955997` / `62.0`
