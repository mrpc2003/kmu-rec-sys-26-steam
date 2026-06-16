# Public-readiness audit — 2026-06-16 23:52 KST

## Verdict

`PASS_WITH_CAUTIONS` — 원격 저장소가 현재 private이면 public 전환 가능.

## What was checked

- GitHub visibility and auth
- Current tracked file inventory
- Current and historical forbidden prefixes: `data/`, `artifacts/`, `submissions/`, `wandb/`, `.omo/`, `final_package/`
- Current and historical credential patterns: GitHub/OpenAI/HF/AWS/Kaggle/private-key/signed URL/generic token assignments
- Largest historical blobs
- README and license presence

## Cleanup already applied

Before public conversion, two small metadata files under `artifacts/` were removed from all Git history with `git-filter-repo --invert-paths`:

- `artifacts/lightgcn_20260530/test_full_train/raw_save_meta_emb64_L3_reg1e-04.json`
- `artifacts/stacker_20260530/test_candidate/meta_emb64_L3_reg1e-04.json`

A local private backup bundle was created before the rewrite.

## Audit results

```text
tracked_count = 798
tracked_total_mb = 9.992
current_forbidden_prefix_count = 0
history_forbidden_prefix_count = 0
current_secret_hit_count = 0
history_secret_hit_count = 0
local_absolute_path_file_count = 234
license_file_present = False
readme_pair_present = True
```

Largest historical blob after cleanup is under 1 MB, so there is no GitHub large-file blocker.

## Cautions

- `LICENSE` is absent. Public visibility is still possible, but reuse rights remain unspecified until a code license is chosen.
- Some reports keep container-local paths such as `/opt/data/...` for reproducibility. They are not credentials, but they are machine-specific.
- Raw Kaggle data, generated submissions, final label CSVs, W&B local runs, and large score artifacts are not tracked and remain ignored/local.

## Public conversion decision

No raw dataset, submission CSV, credential, private key, or large artifact remains in current tree or reachable Git history after the rewrite. The repository can be made public.
