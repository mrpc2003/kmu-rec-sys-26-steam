# KMURecSys26 Steam — OpenCode 재가동 후 후속 진행 체크포인트

- 작성 시각: 2026-06-12 22:55 KST
- 작업 경로: `/opt/data/kaggle/kmu-rec-sys-26-steam`
- 범위: OpenCode 복구 뒤 no-submit 탐색 재시도, 안전 검증, final-2 포장 후보 재확인

## 1. 이번에 이어서 한 일

OpenCode 쪽 문제를 고친 뒤, 같은 흐름을 다시 탔다. 먼저 repo 상태와 제출 안전 상태를 확인했다.

- OpenCode version: `1.15.11`
- OpenCode 기본 agent: `Hephaestus - Deep Agent`
- 기본 model route: `openai/gpt-5.5`
- preflight 시점 `submissions/*.csv` 개수: `23`
- live Kaggle submit process: 없음
- repo 관련 장기 실행 process: 없음

그다음 OpenCode adviser를 no-submit 조건으로 다시 실행했다.

- prompt: `reports/20260612T2246KST_opencode_axis_advisory_prompt.md`
- raw JSONL: `logs/20260612T2246KST_opencode_axis_advisory.jsonl`
- 추출 텍스트: `reports/20260612T2246KST_opencode_axis_advisory_raw_text.md`
- sentinel: `OPENCODE_AXIS_ADVISORY_DONE` 확인
- OpenCode error event: `0`

## 2. OpenCode adviser 판정

OpenCode의 최종 판정은 `NO_SAFE_INTERNAL_AXIS`였다.

핵심 내용은 짧다.

- 제공 데이터만 쓰는 내부 축 중 strict gate를 노릴 만한 새 probe가 없다.
- EASE/ItemKNN, GF-CF, Turbo-CF, ALS/WMF, stacker, hyperbolic, sequential, simple hours/date/text/popularity residual은 이미 약하거나 닫혔다.
- seed expansion이나 단순 ensemble retune은 안정화용이지 `+0.00970` public gap을 설명할 축이 아니다.
- 남은 “새 정보” 축은 Steam metadata지만, 교수자/사용자 승인 전에는 수집·학습하지 않는 것이 맞다.

Hermes 쪽에서 OpenCode 출력도 다시 확인했다.

```json
{
  "verdict": "NO_SAFE_INTERNAL_AXIS",
  "validation_only": true,
  "candidate_csv_written": false,
  "kaggle_submit_executed": false,
  "external_metadata_required": false,
  "recommended_probe_name": null
}
```

## 3. 안전 검증

OpenCode 실행 뒤 안전 상태를 다시 확인했다.

- `submissions/*.csv` 개수: 실행 전 `23`, 실행 후 `23`
- OpenCode prompt 작성 이후 새 `submissions/*.csv`: `0`
- live Kaggle submit process: 없음
- full-test 후보 CSV 생성: 없음
- hidden label 사용: 없음
- 외부 Steam metadata 수집: 없음

따라서 이번 OpenCode 재시도는 no-submit 범위 안에서 끝났다.

## 4. 왜 새 probe를 바로 띄우지 않았는가

이번에는 실행 가능한 probe가 없어서 멈춘 것이 아니다. OpenCode까지 다시 붙여서 보니, 지금 바로 돌릴 만한 내부 데이터 기반 probe가 strict gate를 통과할 가능성이 낮다는 쪽으로 정리됐다.

특히 최근에 다시 확인한 축은 이미 결과가 분명하다.

| 축 | 최근 확인 | 판정 |
|---|---:|---|
| EASE/ItemKNN wide audit | best mean `0.742849` | emb128 기준선보다 크게 낮음 |
| GF-CF spectral panel | best blend mean `0.763786` | emb128 `0.76505` 미달 |
| ALS/WMF residual | mean `+0.000800`, min `-0.000300`, p `0.129` | strict gate fail |
| rank-blend emb128+emb192 | public `0.77825` | 현 public best지만 validation 통계는 약함 |

그래서 새 validation-only probe는 이번 턴에서 띄우지 않았다. 같은 닫힌 축을 이름만 바꿔 다시 돌리면 시간과 로그만 늘어난다.

## 5. final-2 포장 후보 재확인

OpenCode 판정이 `NO_SAFE_INTERNAL_AXIS`라서, 다음으로 final packaging 쪽을 확인했다.

### 5.1 후보 A — public best rank-blend

- 역할: public best / manual-risk final slot 후보
- 파일: `submissions/candidate_rank_blend_emb128_emb192.csv`
- Public: `0.77825`
- SHA256: `835b8b90ce30116a3df7a7575e6ccdaec268af9c1acb01ca0c15c733b3152b2e`
- 컬럼: `ID,Played`
- rows: `19,998`
- label 분포: `1=9,999`, `0=9,999`
- ID 범위: `0..19997`
- ID unique: true
- user별 top-half 위반: `0`
- emb128 후보와 row diff: `368`

검증 근거:

- report: `reports/20260612_rank_blend_emb128_emb192_preflight_refresh.json`
- 생성 스크립트: `scripts/materialize_rank_blend_emb128_emb192.py`
- validation evidence: 3-split mean delta vs emb128 `+0.00083`, split deltas `[0.0017, 0.0003, 0.0005]`, Fisher p `0.3421`, strict gate `false`

해석: 점수는 가장 높지만, 통계적으로 강한 후보라기보다는 public best를 final slot에 보존하는 성격이다.

### 5.2 후보 B — emb128 stable backbone

- 역할: 재현 안정형 final slot 후보
- 파일: `artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv`
- Public: `0.77745`
- uniform surrogate: `0.76505`
- SHA256: `7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195`
- 컬럼: `ID,Label`
- rows: `19,998`
- label 분포: `1=9,999`, `0=9,999`
- ID 범위: `0..19997`
- ID unique: true
- user별 top-half 위반: `0`
- seeds: `42, 123, 2024, 7`
- config: emb_dim `128`, layers `4`, reg `1e-3`, lr `1e-3`, epochs `200`, batch `4096`

실행 검증:

```text
candidate : /opt/data/kaggle/kmu-rec-sys-26-steam/artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv
sha256    : 7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195
expected  : 7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195
SHA MATCH : YES reproducible
preflight : rows=19998 1/0=9999/9999 bad_users=0 id_ok=True
```

재현 명령:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy \
  python3 scripts/reproduce_submission_emb128.py --verify-existing
```

report: `reports/20260601_ecampus_repro_emb128_verification.json`

해석: public 점수는 rank-blend보다 `0.00080` 낮지만, byte-identical 재현과 validation 안정성은 가장 좋다.

## 6. 현재 결론

OpenCode 문제는 더 이상 blocker가 아니다. 복구된 Hephaestus 경로로 advisory를 다시 돌렸고, 내부 데이터만으로 바로 띄울 새 strict-gate probe는 없다는 판정을 받았다. Hermes 검증에서도 safety 위반은 없었다.

그래서 지금 상태는 다음 둘 중 하나로 갈린다.

1. **외부 Steam metadata 사용 가능 여부를 교수자에게 확인한다.** 허용되면 새 정보 축으로 다시 열린다.
2. **허용 전에는 final packaging으로 간다.** 현재 final-2 후보는 public best rank-blend와 stable emb128 backbone 조합이 가장 깔끔하다.

새 Kaggle 제출은 하지 않았다. 새 full-test 후보도 만들지 않았다.
