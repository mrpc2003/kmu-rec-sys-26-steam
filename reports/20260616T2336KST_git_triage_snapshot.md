# Git 반영 전 아티팩트 선별 기록 — 2026-06-16 23:36 KST

## 요약

- 기준 브랜치: `main`
- 기준 원격 커밋: `a11129f38cf9`
- Kaggle 제출 실행: 없음
- 새 후보/제출 CSV 생성: 없음
- 삭제: 0바이트 untracked placeholder 1개 제거

## 선별 정책

커밋 대상:

- `scripts/` 아래 재현·검증용 파이썬 스크립트
- `reports/` 아래 사람이 읽는 Markdown, JSON/CSV 집계, PNG 표/그림
- 기존 tracked ledger/state 업데이트
- `.gitignore` 보호 규칙

로컬/ignored 유지:

- `.omo/` 런타임 continuation 상태
- `data/`, `artifacts/`, `wandb/`, `wandb_runs/`, `submissions/`
- `final_package/*.csv` 최종 라벨 CSV 원본
- 반복 autorun/idle submission snapshot CSV, submit stdout/stderr, status jsonl/txt/디렉터리
- 비어 있는 중단 산출물 `reports/*_antifail_scan.json`

## 카운트

```text
visible_untracked_after_staging = 0
ignored_untracked_after_staging = 4488
staged_files = 442
ignored_final_package_count = 2
ignored_submission_count = 24
```

staged by top-level:

```text
reports: 396
scripts: 44
.gitignore: 1
state: 1
```

staged by extension:

```text
.md: 243
.json: 131
.py: 44
.csv: 17
.png: 6
<noext>: 1
```

## 주의

중요한 최종 라벨 CSV는 GitHub에 직접 올리지 않고, `reports/20260615T_final_package_file_preflight.json` 및 관련 Markdown 보고서로 경로·SHA·행 수를 기록하는 방식으로 보존한다. 필요하면 이후 `git add -f final_package/<파일>`로 명시 승인된 파일만 강제 추가한다.
