# KMURecSys26 Steam final packaging checkpoint

작성 시각: 2026-06-16 06:28 KST

## 현재 판단

boundary v1은 닫는다. forced public probe까지 태웠고, public 0.77705로 current best 0.77825를 넘지 못했다. OpenCode no-submit adviser도 `NO_SAFE_INTERNAL_AXIS`로 판정했다. 이제 새 speculative probe보다 최종 제출 파일·재현성·해시 검수에 집중한다.

## public 기준 후보

| slot | 파일 | public | SHA256 | 용도 |
|---:|---|---:|---|---|
| 1 | `final_package/final_slot1_rank_blend_emb128_emb192_LABEL.csv` | 0.77825 | `1d38c3edf7afae2bbcde9f3d548aa92276323379af2c0c143b66844f25cef052` | 최종 1순위 |
| 2 | `final_package/final_slot2_lightgcn_emb128L4r3_4seed_LABEL.csv` | 0.77745 | `7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195` | 안정 backup |
| 제외 | `submissions/candidate_boundary_v1_ridge_fast_panel20_forced.csv` | 0.77705 | `ebca52d0ea2c185c3dc4770dbf471553213b1fd1cf6a0a5d5a2d22668117f4a6` | boundary v1 실패 기록 |

## final_package 파일 검수

두 final package 파일 모두 다음 조건을 통과했다.

| 항목 | slot1 | slot2 |
|---|---:|---:|
| rows | 19,998 | 19,998 |
| columns | `ID,Label` | `ID,Label` |
| Label=1 | 9,999 | 9,999 |
| Label=0 | 9,999 | 9,999 |
| ID unique | true | true |
| ID contiguous 0..19997 | true | true |
| per-user top-half violation | 0 | 0 |

참고: slot1은 Kaggle에 `Played` header 후보로 public 0.77825가 이미 확인된 `candidate_rank_blend_emb128_emb192.csv`와 같은 예측이다. eCampus/최종 제출 편의를 위해 `ID,Label` header로 별도 복사했다. SHA가 Kaggle 기록의 `final_slot1_rank_blend_emb128_emb192_LABEL.csv` 메시지와 맞는다.

## 재현성 검증

### slot1 rank-blend

검증 명령:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --python 3.13 \
  python3 scripts/materialize_rank_blend_emb128_emb192.py \
  --out /tmp/candidate_rank_blend_emb128_emb192_verify.csv \
  --report /tmp/rankblend_verify.json
cmp -s /tmp/candidate_rank_blend_emb128_emb192_verify.csv \
  submissions/candidate_rank_blend_emb128_emb192.csv
```

결과:

```text
RANKBLEND_BYTE_IDENTICAL
sha256 = 835b8b90ce30116a3df7a7575e6ccdaec268af9c1acb01ca0c15c733b3152b2e
```

### slot2 emb128 LightGCN 4-seed

검증 명령:

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy --python 3.13 \
  python3 scripts/reproduce_submission_emb128.py --verify-existing
```

결과:

```text
SHA MATCH : YES ✅ reproducible
sha256 = 7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195
rows=19998 1/0=9999/9999 bad_users=0 id_ok=True
```

## boundary v1 종료 근거

| 항목 | 값 |
|---|---:|
| forced public score | 0.77705 |
| current best | 0.77825 |
| delta | -0.00120 |
| row diff vs current best | 176 |
| public changed rows 추정 | 88 |
| public net rows 추정 | -11.9988 |
| public implied precision 추정 | 0.431825 |
| scored panel20 mean flip precision | 0.54537 |
| top2 gate pass | false |
| top1 gate pass | false |

`reports/failed_axes.json`에는 `boundary_v1_ridge_fast_panel20_forced_20260615` 항목을 추가했다. 같은 boundary-family row flip은 재시도하지 않는다.

## OpenCode no-submit adviser

- prompt: `reports/20260615T_opencode_after_boundary_forced_no_submit_prompt.md`
- raw log: `reports/20260615T_opencode_after_boundary_forced_no_submit.jsonl`
- extracted text: `reports/20260615T_opencode_after_boundary_forced_no_submit_raw_text.md`
- sentinel: `BOUNDARY_AFTERCARE_ADVISORY_DONE`
- verdict: `NO_SAFE_INTERNAL_AXIS`

요지: boundary v1 public fail, panel20 gate fail, closed axes 대부분이 같은 family retune라서 더 밀면 public overfit 위험만 커진다는 판단이다.

## 남은 일

1. final package 두 파일을 제출용으로 고정한다.
2. `reports/failed_axes.json`, closure report, final packaging report만 선별 staging한다. raw artifact/log/cache 전체를 한 번에 stage하지 않는다.
3. 대회 최종 제출/과제 제출 UI에서 slot1은 0.77825 파일, slot2는 emb128 0.77745 파일로 고른다.
