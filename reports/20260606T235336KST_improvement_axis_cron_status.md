# KMURecSys26 Steam OpenCode-first no-submit axis loop — 2026-06-06 23:53:36 KST

## 결론

- **새 OpenCode/새 축은 실행하지 않았습니다.** 기존 validation-only probe인 `UserKNN gated residual fine-grid`가 아직 살아 있어 중복 실행을 피했습니다.
- **Jackknife expanded probe는 실패/미완료로 판정했습니다.** 이전 OpenCode가 띄운 `expanded` PID는 더 이상 없고, 기대 리포트(`reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.{json,md}`)와 산출 artifact가 없습니다. 로그도 12줄에서 `val_random_uniform_seed123` 중간에 멈췄습니다.
- **Jackknife smoke 결과는 기존대로 WEAK_SIGNAL입니다.** 최고 평균 Δ `+0.000367`, min Δ `-0.001200`, `2/3` positive, fixes/breaks `252/230`, p `0.3388`로 strict gate 통과가 아닙니다.

## Safety

- validation_only: `true`
- 이 tick에서 Kaggle submit 실행: `false`
- 실제 Kaggle submit 프로세스: `0`
- 이 tick에서 full-test candidate/submission CSV 생성: `false`
- hidden labels/private answers/external Steam scraping: `false`
- quarantine/guard logic 약화: `false`
- git stage/commit/push: `false`
- credential/token 출력: `false`

## OpenCode 상태

- OpenCode CLI/auth 확인: `/opt/data/home/.local/bin/opencode`, version `1.15.11`, credentials present(값은 출력하지 않음).
- 이전 OpenCode loop:
  - status: `reports/20260606T220406KST_opencode_improvement_axis_loop_status.md`
  - exit code: `0`
  - sentinel: `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
- 이번 tick에서는 새 OpenCode run을 띄우지 않았습니다. 이유: 이미 CPU-bound validation-only `UserKNN gated residual fine-grid`가 active이므로, 정책상 새 축을 중복 실행하지 않고 모니터링합니다.

## Active / completed process reconciliation

### 1) UserKNN gated residual fine-grid — RUNNING_UNCLASSIFIED

- PIDs: `18483`, `18804`, `18812`
- 관측 상태: PID `18812`가 약 100% CPU로 실행 중, elapsed 약 `10h28m`
- log: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
- 관측 log lines: `28289`
- 기대 리포트:
  - `reports/20260606T132450KST_userknn_gated_residual_fine.json` — 아직 없음
  - `reports/20260606T132450KST_userknn_gated_residual_fine.md` — 아직 없음
- artifact dir 파일: 아직 관측되지 않음
- 판정: **RUNNING_UNCLASSIFIED**
- 이유: 결과 JSON/MD가 아직 없어서 strict/weak/reject gate 적용 불가. 프로세스가 살아 있으므로 새 probe를 중복 실행하지 않음.

### 2) Jackknife uncertainty boundary expanded — FAILED_INCOMPLETE_NO_METRIC_REPORT

- 이전 expected pids: `28965`, `28974` — 현재 not live
- log: `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`
- 기대 리포트:
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` — 없음
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.md` — 없음
- artifact dir: 파일 없음
- 로그 상태: 872 bytes / 12 lines, `val_random_uniform_seed123` 진행 중간에서 종료
- 판정: **FAILED_INCOMPLETE_NO_METRIC_REPORT**
- 의미: complete metric이 없으므로 STRICT_PASS/WEAK_SIGNAL이 아님. Jackknife expanded axis는 현 상태에서 escalate하지 않습니다.

### 3) Jackknife uncertainty boundary smoke — WEAK_SIGNAL, not candidate

- report: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.md`
- JSON: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
- best variant: `vote_consensus__high_capacity_gap__B1__w0.1`
- mean Δ: `+0.0003667400`
- min Δ: `-0.0012002400`
- positive splits: `2/3`
- fixes/breaks: `252 / 230`
- pooled exact p: `0.338815`
- strict gate failures:
  - mean Δ `< +0.0015`
  - min Δ `< 0`
  - not `3/3` positive
  - p `>= 0.05`
- 판정: **WEAK_SIGNAL / escalation reject**

### 4) Existing aggressive quota runner

- PIDs: `7613`, `7630`
- 기존 프로세스이며 이번 tick에서 시작하지 않았습니다.
- 현재 실제 Kaggle submit 프로세스는 `0`개로 관측했습니다.

## GPU snapshot

- GPU0: `0 / 32768 MiB`, util `0%`
- GPU1: `0 / 32768 MiB`, util `0%`
- GPU2: `0 / 32768 MiB`, util `1%`
- GPU3: `4320 / 32768 MiB`, util `12%`

## Artifacts written this tick

- JSON: `reports/20260606T235336KST_improvement_axis_cron_status.json`
- Markdown: `reports/20260606T235336KST_improvement_axis_cron_status.md`

## Next action

다음 tick에서는 우선 `reports/20260606T132450KST_userknn_gated_residual_fine.json` 생성 여부와 PID `18812` 종료 여부를 확인해야 합니다. 생성/종료되면 아래 strict gate로 분류합니다:

- mean Δ `>= +0.0015`
- min Δ `>= 0`
- `3/3` positive splits
- fixes `>` breaks
- pooled exact p `< 0.05`
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`

UserKNN도 실패/종료되면 그때 새 OpenCode-first no-submit axis search를 1개만 띄우는 것이 다음 조치입니다.
