#!/usr/bin/env python3
"""OpenCode -> Hermes-review -> rejection-feedback improvement-axis loop.

This controller is intentionally no-submit. It repeatedly asks OpenCode to find
or launch a bounded validation-only improvement-axis probe, then applies a
Hermes-side strict review. If the result is rejected, the rejection reasons are
fed into the next OpenCode prompt so the next iteration must search a different
axis instead of repeating the same weak/stalled family.

Allowed side effects:
- reports/<timestamp>_axis_loop_iterNN_* prompts/reports/raw text
- logs/<timestamp>_axis_loop_iterNN_opencode.jsonl
- artifacts/opencode_hermes_axis_loop_<timestamp>/ metadata

Forbidden side effects enforced by prompt and checked by review:
- Kaggle submission commands
- full-test candidate/submission CSV creation under submissions/
- hidden labels/private answers/external Steam scraping
- git stage/commit/push
- recursive cron scheduling
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9), name="KST")
ROOT_DEFAULT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SENTINELS = [
    "OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND",
    "OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING",
    "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS",
    "OPENCODE_AXIS_LOOP_DONE_BLOCKED",
]
STRICT_MEAN_DELTA = 0.0015
STRICT_P = 0.05


@dataclass
class IterationResult:
    iteration: int
    timestamp_kst: str
    dry_run: bool
    prompt_path: str
    opencode_jsonl: str | None
    raw_text_path: str | None
    expected_report_json: str
    expected_report_md: str
    opencode_exit_code: int | None
    opencode_timed_out: bool
    sentinel: str | None
    hermes_verdict: str
    strict_pass: bool
    rejection_reasons: list[str]
    safety_issues: list[str]
    metrics_checked: list[dict[str, Any]]
    new_probe: dict[str, Any] | None
    new_submission_csvs: list[str]


def kst_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%dT%H%M%SKST")


def log(msg: str) -> None:
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}] {msg}", flush=True)


def ensure_dirs(root: Path) -> None:
    for rel in ("reports", "logs", "artifacts", "scripts"):
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_text(cmd: list[str], cwd: Path, timeout: int = 30) -> str:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return cp.stdout[-8000:]
    except Exception as exc:  # pragma: no cover - defensive runtime path
        return f"<command failed: {type(exc).__name__}: {exc}>"


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def recent_files(root: Path, patterns: list[str], limit: int = 12) -> list[str]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    uniq = sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return [relpath(p, root) for p in uniq[:limit]]


def tail_file(path: Path, max_chars: int = 3000) -> str:
    if not path.exists():
        return "<missing>"
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[-max_chars:]
    except Exception as exc:
        return f"<read failed: {exc}>"


def collect_state(root: Path, rejection_history: list[dict[str, Any]]) -> dict[str, Any]:
    ps = run_text(["ps", "-eo", "pid,ppid,etime,stat,pcpu,pmem,args"], root, timeout=20)
    filtered = []
    for line in ps.splitlines():
        low = line.lower()
        if any(x in low for x in ["opencode", "axis_loop", "userknn", "jackknife", "aggressive_quota", "kaggle", "submit", "train", "probe"]):
            if "opencode_hermes_axis_rejection_loop.py" in line and str(os.getpid()) in line:
                continue
            filtered.append(line[:1000])
    latest_reports = recent_files(
        root,
        [
            "reports/*opencode*axis*loop*.json",
            "reports/*improvement_axis*status*.json",
            "reports/*otto*confirmation*.json",
            "reports/*otto*post_submission_analysis.json",
            "reports/*otto*reconciliation*.md",
            "reports/*external_dacon_kaggle_methodology_scan.md",
            "reports/*repeat_until_candidate_status*.json",
            "reports/*userknn*gated*probe*.json",
            "reports/*jackknife*probe*.json",
            "reports/*dns*panel*.json",
            "reports/*last_slot*.json",
        ],
        limit=16,
    )
    latest_logs = recent_files(
        root,
        [
            "logs/*opencode*axis*loop*.jsonl",
            "logs/*userknn*gated*.log",
            "logs/*jackknife*.log",
        ],
        limit=10,
    )
    state_files = recent_files(root, ["state/*.json"], limit=10)
    userknn_log = root / "logs/userknn_gated_residual_fine_20260606T132450KST.log"
    jack_log = root / "logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log"
    failed_axes_path = root / "reports/failed_axes.json"
    failed_axes_tail: Any = "<missing>"
    if failed_axes_path.exists():
        try:
            failed_axes_obj = json.loads(failed_axes_path.read_text(encoding="utf-8"))
            if isinstance(failed_axes_obj, list):
                failed_axes_tail = failed_axes_obj[-12:]
            else:
                failed_axes_tail = failed_axes_obj
        except Exception as exc:
            failed_axes_tail = f"<failed_axes_json_parse_failed:{type(exc).__name__}:{exc}>"
    return {
        "timestamp_kst": datetime.now(KST).isoformat(),
        "active_process_lines": filtered[:30],
        "latest_reports": latest_reports,
        "latest_logs": latest_logs,
        "state_files": state_files,
        "known_incomplete": {
            "userknn_gated_residual_fine": {
                "expected_json": "reports/20260606T132450KST_userknn_gated_residual_fine.json",
                "expected_md": "reports/20260606T132450KST_userknn_gated_residual_fine.md",
                "expected_exist": (root / "reports/20260606T132450KST_userknn_gated_residual_fine.json").exists(),
                "log_tail": tail_file(userknn_log, 1200),
            },
            "jackknife_uncertainty_boundary_expanded": {
                "expected_json": "reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json",
                "expected_md": "reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.md",
                "expected_exist": (root / "reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json").exists(),
                "log_tail": tail_file(jack_log, 1200),
            },
        },
        "latest_failed_axes_tail": failed_axes_tail,
        "recent_rejections": rejection_history[-8:],
    }


def json_dumps_compact(obj: Any, max_chars: int = 12000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated for prompt>"


def build_prompt(
    *,
    root: Path,
    run_ts: str,
    iteration: int,
    state: dict[str, Any],
    report_json: str,
    report_md: str,
    artifact_dir: str,
) -> str:
    iter_tag = f"{run_ts}_axis_loop_iter{iteration:02d}"
    return f"""# KMURecSys26 Steam — OpenCode axis-finding iteration {iteration:02d}

You are OpenCode acting as a constrained validation-only worker inside `{root}`.

CRITICAL EXECUTION RULES:
- Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- Your job: find a fresh independent improvement axis OR launch exactly one bounded validation-only probe.
- Do NOT run `kaggle competitions submit`.
- Do NOT create full-test candidate/submission/uploadable CSVs. Do NOT write under `submissions/`.
- Do NOT use hidden labels/private answers. Do NOT scrape external Steam data.
- Do NOT print credentials/tokens/secrets.
- Do NOT weaken quarantine/guard logic.
- Do NOT git add/stage/commit/push.
- Do NOT schedule cron jobs.
- Validation-only scripts, split artifacts, logs, and reports are allowed.
- If launching a probe, it must be bounded with explicit timeout <= 3600 seconds and report/log paths. Prefer <= 20 minutes.

Objective:
- Find a real axis beyond current public-best behavior.
- Current public best reference: `candidate_rank_blend_emb128_emb192.csv`, public `0.77825`.
- Main internal uniform reference: emb128 4-seed, around `0.76505`.
- Weak one-split or tiny `+0.000x` blips are NOT candidates; they require stronger multi-split evidence.

Strict gate Hermes will apply:
- validation_only=true
- candidate/full-test/submission CSV written=false
- kaggle_submit_executed=false
- mean_delta >= +0.0015
- min_delta >= 0
- 3/3 positive validation splits
- fixes > breaks
- pooled exact/McNemar p < 0.05 when available
- no quarantine/near-duplicate/public-negative-family conflict

Closed/stalled families to avoid unless you introduce a materially new bounded design:
- OTTO/source-separated co-visitation family already forced-tested after user approval: independent strict failed (`mean_delta +0.0006668`, `min_delta -0.0006001`, positive `2/3`, `p=0.1700`); one-off public score `0.77815`, below current live best `0.77825`. Do not spend more work on coplay_top5/reverse_recent/nearby weight retunes unless adding genuinely new independent information against the current-best rankblend anchor.
- broad UserKNN gated fine-grid: stalled/incomplete, repeated invalid-divide warnings, no metric report
- jackknife uncertainty expanded: failed/incomplete, no metric report
- jackknife smoke: weak, split-negative, p non-significant
- previous UserKNN smoke: weak, below mean and p gates
- DNS pool=1: split-specific/public-noise risk
- hours-confidence, exact-K, temporal compatibility, boundary covariate/residual, SL@K-lite, last-slot sparse agreement
- capacity/frontier/emb192 marginal public noise
- rankblend/ALS/BPR residual, boundary scoreblend/pairwise, TAG-CF public-negative/quarantined families
- raw semantic/text/README/LM probes weak or redundant

Hermes rejection feedback and current state are below. You must not repeat rejected axes unless materially narrowed and justified.

```json
{json_dumps_compact(state)}
```

Required outputs for this iteration:
- Markdown report: `{report_md}`
- JSON report: `{report_json}`
- Optional validation artifacts only under `{artifact_dir}`

Required JSON shape (you may add fields):
```json
{{
  "safety_flags": {{
    "validation_only": true,
    "candidate_csv_written": false,
    "full_test_candidate_or_submission_csv_created": false,
    "kaggle_submit_executed": false,
    "hidden_labels_used": false,
    "private_answers_used": false,
    "external_steam_scraping_used": false,
    "credentials_or_tokens_printed": false,
    "quarantine_or_guard_logic_weakened": false,
    "git_stage_commit_push_executed": false,
    "recursive_cron_scheduled": false
  }},
  "axis_decision": "fresh axis tested / probe launched / no safe axis / blocked",
  "new_probe": {{
    "launched": false,
    "status": "not_launched | running | completed | failed",
    "command": null,
    "pid_file": null,
    "log": null,
    "report_json": null,
    "report_md": null,
    "artifact_dir": null
  }},
  "best_or_top_metrics": {{
    "variant": null,
    "mean_delta_vs_base": null,
    "min_delta_vs_base": null,
    "positive_splits": null,
    "fixes": null,
    "breaks": null,
    "pooled_p_exact": null,
    "quarantine_conflict": false
  }},
  "ranked_next_axis_hypotheses": [],
  "verdict": "STRICT_PASS | WEAK_SIGNAL | REJECT | NEXT_PROBE_RUNNING | NO_SAFE_AXIS | BLOCKED"
}}
```

Final response requirements:
- Summarize what you did and why.
- Include report paths.
- Include safety flags.
- End with exactly one sentinel line:
  - OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND
  - OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING
  - OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS
  - OPENCODE_AXIS_LOOP_DONE_BLOCKED
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_opencode(opencode_bin: str, prompt: str, jsonl_path: Path, cwd: Path, timeout_sec: int) -> tuple[int | None, bool]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [opencode_bin, "run", prompt, "--format", "json"]
    with jsonl_path.open("w", encoding="utf-8") as f:
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            return cp.returncode, False
        except subprocess.TimeoutExpired:
            f.write(f"\n__HERMES_OPENCODE_TIMEOUT__ timeout_sec={timeout_sec}\n")
            return None, True


def extract_text_from_jsonl(jsonl_path: Path, raw_text_path: Path) -> tuple[str, str | None]:
    texts: list[str] = []
    sentinel: str | None = None
    if not jsonl_path.exists():
        write_text(raw_text_path, "<missing jsonl>\n")
        return "", None
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for s in SENTINELS:
            if s in line:
                sentinel = s
        try:
            obj = json.loads(line)
        except Exception:
            if line.strip() and not line.startswith("{"):
                texts.append(line)
            continue
        part = obj.get("part") if isinstance(obj, dict) else None
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif isinstance(obj, dict) and obj.get("type") == "text" and isinstance(obj.get("text"), str):
            texts.append(obj["text"])
    text = "\n\n".join(texts)
    write_text(raw_text_path, text + ("\n" if text else ""))
    for s in SENTINELS:
        if s in text:
            sentinel = s
    return text, sentinel


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "report_json_missing"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"json_parse_failed:{type(exc).__name__}:{exc}"
    if not isinstance(obj, dict):
        return None, "json_report_not_object"
    return obj, None


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1", "ok", "pass"}:
            return True
        if low in {"false", "no", "0", "none", "null"}:
            return False
    return None


def get_path(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safety_issues_from_report(data: dict[str, Any] | None) -> list[str]:
    if data is None:
        return ["missing_opencode_json_report"]
    safety = data.get("safety_flags") or data.get("safety") or {}
    if not isinstance(safety, dict):
        return ["safety_flags_missing_or_invalid"]
    issues: list[str] = []
    required_true = ["validation_only"]
    required_false = [
        "candidate_csv_written",
        "full_test_candidate_or_submission_csv_created",
        "kaggle_submit_executed",
        "hidden_labels_used",
        "private_answers_used",
        "external_steam_scraping_used",
        "credentials_or_tokens_printed",
        "quarantine_or_guard_logic_weakened",
        "git_stage_commit_push_executed",
        "recursive_cron_scheduled",
    ]
    aliases_false = {
        "full_test_candidate_or_submission_csv_created": ["full_test_candidate_csv_written", "submission_csv_written"],
        "hidden_labels_used": ["hidden_test_read", "hidden_labels_or_external_scraping"],
        "private_answers_used": ["private_answers_used"],
    }
    for key in required_true:
        val = boolish(safety.get(key))
        if val is not True:
            issues.append(f"safety.{key}_not_true")
    for key in required_false:
        value = safety.get(key)
        if value is None:
            for alt in aliases_false.get(key, []):
                if alt in safety:
                    value = safety.get(alt)
                    break
        val = boolish(value)
        if val is not False:
            # Missing credentials/quarantine keys are softer, but keep them visible.
            issues.append(f"safety.{key}_not_false_or_missing")
    return issues


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def positive_count(value: Any) -> int | None:
    if isinstance(value, str) and "/" in value:
        first = value.split("/", 1)[0].strip()
        try:
            return int(first)
        except Exception:
            return None
    f = as_float(value)
    return int(f) if f is not None else None


def first_present(d: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def collect_metric_dicts(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        metric_keys = {
            "mean_delta_vs_base", "top_mean_delta_vs_base", "mean_delta", "min_delta_vs_base", "top_min_delta_vs_base",
            "positive_splits", "fixes", "breaks", "pooled_p_exact", "mcnemar_p_exact", "paired_p", "p_value_exact",
        }
        if any(k in obj for k in metric_keys):
            out.append(obj)
        for v in obj.values():
            collect_metric_dicts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_metric_dicts(v, out)


def evaluate_metric_dict(d: dict[str, Any]) -> dict[str, Any]:
    mean_delta = as_float(first_present(d, ["mean_delta_vs_base", "top_mean_delta_vs_base", "mean_delta", "delta_mean", "mean_delta_vs_ref"]))
    min_delta = as_float(first_present(d, ["min_delta_vs_base", "top_min_delta_vs_base", "min_delta", "min_delta_vs_ref"]))
    pos = positive_count(first_present(d, ["positive_splits", "top_positive_splits", "positive_split_count"]))
    fixes = as_float(first_present(d, ["fixes", "top_fixes", "b_variant_fixes"]))
    breaks = as_float(first_present(d, ["breaks", "top_breaks", "c_variant_breaks"]))
    p = as_float(first_present(d, ["pooled_p_exact", "top_pooled_p_exact", "mcnemar_p_exact", "paired_p", "p_value_exact", "p_value", "pooled_mcnemar_p"]))
    q_conflict = boolish(first_present(d, ["quarantine_conflict", "near_duplicate_conflict", "blocked_by_quarantine"]))
    no_conflict = boolish(first_present(d, ["no_quarantine_or_near_duplicate_conflict", "no_near_duplicate_or_quarantine_conflict"]))
    if q_conflict is None and no_conflict is not None:
        q_conflict = not no_conflict
    pass_mean = mean_delta is not None and mean_delta >= STRICT_MEAN_DELTA
    pass_min = min_delta is not None and min_delta >= 0
    pass_pos = pos is not None and pos >= 3
    pass_fb = fixes is not None and breaks is not None and fixes > breaks
    pass_p = True if p is None else p < STRICT_P
    pass_q = q_conflict is not True
    strict = pass_mean and pass_min and pass_pos and pass_fb and pass_p and pass_q
    return {
        "variant": first_present(d, ["variant", "top_variant", "name", "axis"]),
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "positive_splits": pos,
        "fixes": fixes,
        "breaks": breaks,
        "p": p,
        "quarantine_conflict": q_conflict,
        "strict_pass": strict,
        "failed": [
            name for name, ok in [
                ("mean_delta", pass_mean),
                ("min_delta", pass_min),
                ("positive_splits", pass_pos),
                ("fixes_gt_breaks", pass_fb),
                ("p_value", pass_p),
                ("quarantine", pass_q),
            ] if not ok
        ],
    }


def new_submission_csvs_since(root: Path, start_time: float) -> list[str]:
    sub = root / "submissions"
    if not sub.exists():
        return []
    out = []
    for p in sub.glob("*.csv"):
        try:
            if p.stat().st_mtime >= start_time:
                out.append(relpath(p, root))
        except FileNotFoundError:
            pass
    return sorted(out)


def active_kaggle_submit_processes() -> list[str]:
    try:
        cp = subprocess.run(["ps", "-eo", "pid,args"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return ["<ps_failed>"]
    rows = []
    self_pid = os.getpid()
    for line in cp.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            pid = -1
        if pid == self_pid:
            continue
        low = line.lower()
        if "kaggle" in low and "competitions" in low and "submit" in low:
            # Exclude harmless report/prompt text in controller/OpenCode command lines.
            # The actual dangerous process is the Kaggle CLI, not an agent prompt that
            # contains the forbidden phrase as part of the safety contract.
            if "opencode_hermes_axis_rejection_loop.py" in low or "opencode run" in low or "/opencode run" in low:
                continue
            rows.append(line[:1000])
    return rows


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def maybe_wait_for_probe(root: Path, new_probe: dict[str, Any] | None, wait_sec: int) -> dict[str, Any]:
    if not isinstance(new_probe, dict) or not boolish(new_probe.get("launched")):
        return {"waited": False, "reason": "no_probe_launched"}
    report_json_val = new_probe.get("report_json")
    report_md_val = new_probe.get("report_md")
    pid_file_val = new_probe.get("pid_file")
    report_json = root / str(report_json_val) if report_json_val else None
    report_md = root / str(report_md_val) if report_md_val else None
    pid_file = root / str(pid_file_val) if pid_file_val else None
    start = time.time()
    while time.time() - start < wait_sec:
        if report_json and report_json.exists():
            return {"waited": True, "status": "report_json_ready", "elapsed_sec": int(time.time() - start)}
        if report_md and report_md.exists() and not report_json:
            return {"waited": True, "status": "report_md_ready", "elapsed_sec": int(time.time() - start)}
        if pid_file and pid_file.exists():
            try:
                txt = pid_file.read_text(encoding="utf-8", errors="replace").strip().split()[0]
                pid = int(txt)
            except Exception:
                pid = -1
            if pid > 0 and not pid_alive(pid):
                return {"waited": True, "status": "pid_dead_no_report", "elapsed_sec": int(time.time() - start), "pid": pid}
        time.sleep(min(30, max(5, wait_sec // 20 if wait_sec else 5)))
    return {"waited": True, "status": "probe_still_running_or_no_report_after_wait", "elapsed_sec": int(time.time() - start)}


def review_iteration(root: Path, report_json: Path, iter_start: float, sentinel: str | None, wait_sec: int) -> tuple[str, bool, list[str], list[str], list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    data, err = load_json(report_json)
    rejection_reasons: list[str] = []
    if err:
        rejection_reasons.append(err)
    safety_issues = safety_issues_from_report(data)
    new_csvs = new_submission_csvs_since(root, iter_start)
    if new_csvs:
        safety_issues.append("new_submission_csvs_created_since_iteration_start")
    submit_procs = active_kaggle_submit_processes()
    if submit_procs:
        safety_issues.append("active_kaggle_submit_process_detected")
    if sentinel is None:
        rejection_reasons.append("opencode_sentinel_missing")
    metrics_checked: list[dict[str, Any]] = []
    strict_pass = False
    new_probe: dict[str, Any] | None = None
    if data is not None:
        npv = data.get("new_probe")
        if isinstance(npv, dict):
            new_probe = dict(npv)
            wait_info = maybe_wait_for_probe(root, new_probe, wait_sec)
            new_probe["hermes_wait_info"] = wait_info
            # If a report appeared after waiting, prefer reviewing that probe report too.
            probe_report = new_probe.get("report_json")
            if probe_report and (root / str(probe_report)).exists() and (root / str(probe_report)) != report_json:
                probe_data, probe_err = load_json(root / str(probe_report))
                if probe_err:
                    rejection_reasons.append(f"probe_report_{probe_err}")
                elif probe_data:
                    collect_metric_dicts(probe_data, metrics_checked)
                    safety_issues.extend([f"probe.{x}" for x in safety_issues_from_report(probe_data) if x != "safety.credentials_or_tokens_printed_not_false_or_missing"])
            if wait_info.get("status") == "probe_still_running_or_no_report_after_wait":
                return "PROBE_STILL_RUNNING", False, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs
        collect_metric_dicts(data, metrics_checked)
    metric_evals = [evaluate_metric_dict(m) for m in metrics_checked]
    # de-duplicate compactly
    compact: list[dict[str, Any]] = []
    seen = set()
    for ev in metric_evals:
        key = (ev.get("variant"), ev.get("mean_delta"), ev.get("min_delta"), ev.get("positive_splits"), ev.get("fixes"), ev.get("breaks"), ev.get("p"))
        if key in seen:
            continue
        seen.add(key)
        compact.append(ev)
    metrics_checked[:] = compact[:25]
    strict_pass = any(ev.get("strict_pass") for ev in compact)
    if not compact:
        rejection_reasons.append("no_strict_gate_metrics_found")
    else:
        best = sorted(compact, key=lambda x: (x.get("strict_pass") is True, x.get("mean_delta") or -999), reverse=True)[0]
        if not best.get("strict_pass"):
            rejection_reasons.append("best_metrics_failed:" + ",".join(best.get("failed") or []))
    if safety_issues:
        strict_pass = False
    if strict_pass and sentinel == "OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND":
        return "STRICT_PASS", True, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs
    if strict_pass:
        # allow strict metrics even if sentinel says weak/no-safe, but mark as candidate-like for Hermes follow-up
        return "STRICT_PASS_METRICS_SENTINEL_MISMATCH", True, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs
    if sentinel == "OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING" and new_probe and boolish(new_probe.get("launched")):
        return "NEXT_PROBE_REVIEWED_OR_WAITED_BUT_NOT_STRICT", False, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs
    if sentinel == "OPENCODE_AXIS_LOOP_DONE_BLOCKED":
        return "BLOCKED_REJECTED_FOR_LOOP_CONTINUATION", False, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs
    return "REJECTED_CONTINUE", False, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs


def write_summary(root: Path, run_ts: str, results: list[IterationResult], final_verdict: str, dry_run: bool) -> tuple[Path, Path]:
    summary_json = root / "reports" / f"{run_ts}_opencode_hermes_axis_rejection_loop_summary.json"
    summary_md = root / "reports" / f"{run_ts}_opencode_hermes_axis_rejection_loop_summary.md"
    payload = {
        "timestamp_kst": datetime.now(KST).isoformat(),
        "run_ts": run_ts,
        "dry_run": dry_run,
        "final_verdict": final_verdict,
        "safety_contract": {
            "no_kaggle_submit": True,
            "no_full_test_candidate_or_submission_csv": True,
            "no_hidden_labels_or_private_answers": True,
            "no_external_steam_scraping": True,
            "no_git_stage_commit_push": True,
            "no_recursive_cron": True,
        },
        "iterations": [asdict(r) for r in results],
    }
    write_text(summary_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# OpenCode↔Hermes rejection loop summary — {run_ts}",
        "",
        f"- final_verdict: `{final_verdict}`",
        f"- dry_run: `{str(dry_run).lower()}`",
        f"- iterations: `{len(results)}`",
        "",
        "## Safety contract",
        "",
        "- no Kaggle submit",
        "- no full-test candidate/submission CSV creation",
        "- no hidden labels/private answers/external Steam scraping",
        "- no git stage/commit/push",
        "- no recursive cron scheduling",
        "",
        "## Iterations",
        "",
        "| iter | sentinel | Hermes verdict | strict | rejection reasons | report |",
        "|---:|---|---|---:|---|---|",
    ]
    for r in results:
        reasons = "; ".join((r.rejection_reasons + r.safety_issues)[:5])
        lines.append(
            f"| {r.iteration} | `{r.sentinel}` | `{r.hermes_verdict}` | `{str(r.strict_pass).lower()}` | {reasons or '-'} | `{r.expected_report_json}` |"
        )
    lines.extend(["", "## Artifacts", "", f"- JSON: `{relpath(summary_json, root)}`", f"- Markdown: `{relpath(summary_md, root)}`"])
    write_text(summary_md, "\n".join(lines) + "\n")
    return summary_json, summary_md


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--opencode-timeout-sec", type=int, default=900)
    ap.add_argument("--probe-wait-sec", type=int, default=3600)
    ap.add_argument("--sleep-between-iters-sec", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--opencode-bin", default="opencode")
    ap.add_argument("--workdir", default=str(ROOT_DEFAULT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.workdir).resolve()
    if not root.exists():
        raise SystemExit(f"workdir does not exist: {root}")
    ensure_dirs(root)
    run_ts = kst_stamp()
    artifact_dir = root / "artifacts" / f"opencode_hermes_axis_loop_{run_ts}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results: list[IterationResult] = []
    rejection_history: list[dict[str, Any]] = []
    final_verdict = "MAX_ITERS_REACHED_NO_STRICT_PASS"
    log(f"controller start run_ts={run_ts} max_iters={args.max_iters} dry_run={args.dry_run}")

    for iteration in range(1, args.max_iters + 1):
        iter_start = time.time()
        state = collect_state(root, rejection_history)
        iter_tag = f"{run_ts}_axis_loop_iter{iteration:02d}"
        prompt_path = root / "reports" / f"{iter_tag}_prompt.md"
        report_json = root / "reports" / f"{iter_tag}_opencode.json"
        report_md = root / "reports" / f"{iter_tag}_opencode.md"
        raw_text_path = root / "reports" / f"{iter_tag}_opencode_raw_text.md"
        jsonl_path = root / "logs" / f"{iter_tag}_opencode.jsonl"
        prompt = build_prompt(
            root=root,
            run_ts=run_ts,
            iteration=iteration,
            state=state,
            report_json=relpath(report_json, root),
            report_md=relpath(report_md, root),
            artifact_dir=relpath(artifact_dir, root),
        )
        write_text(prompt_path, prompt)
        log(f"iter {iteration}: prompt={relpath(prompt_path, root)}")
        exit_code: int | None = None
        timed_out = False
        sentinel: str | None = None
        if args.dry_run:
            fake = {
                "safety_flags": {"validation_only": True, "candidate_csv_written": False, "full_test_candidate_or_submission_csv_created": False, "kaggle_submit_executed": False, "hidden_labels_used": False, "private_answers_used": False, "external_steam_scraping_used": False, "credentials_or_tokens_printed": False, "quarantine_or_guard_logic_weakened": False, "git_stage_commit_push_executed": False, "recursive_cron_scheduled": False},
                "axis_decision": "dry_run_no_opencode_called",
                "new_probe": {"launched": False, "status": "dry_run", "command": None, "pid_file": None, "log": None, "report_json": None, "report_md": None, "artifact_dir": None},
                "best_or_top_metrics": {"variant": None, "mean_delta_vs_base": None, "min_delta_vs_base": None, "positive_splits": None, "fixes": None, "breaks": None, "pooled_p_exact": None, "quarantine_conflict": False},
                "ranked_next_axis_hypotheses": [],
                "verdict": "DRY_RUN",
            }
            write_text(report_json, json.dumps(fake, ensure_ascii=False, indent=2) + "\n")
            write_text(report_md, f"# Dry run iteration {iteration}\n\nNo OpenCode call was made.\n")
            write_text(raw_text_path, "DRY_RUN\nOPENCODE_AXIS_LOOP_DONE_BLOCKED\n")
            sentinel = "OPENCODE_AXIS_LOOP_DONE_BLOCKED"
            exit_code = 0
            timed_out = False
        else:
            log(f"iter {iteration}: running OpenCode timeout={args.opencode_timeout_sec}s log={relpath(jsonl_path, root)}")
            exit_code, timed_out = run_opencode(args.opencode_bin, prompt, jsonl_path, root, args.opencode_timeout_sec)
            _, sentinel = extract_text_from_jsonl(jsonl_path, raw_text_path)
        verdict, strict_pass, rejection_reasons, safety_issues, metrics_checked, new_probe, new_csvs = review_iteration(
            root, report_json, iter_start, sentinel, args.probe_wait_sec
        )
        result = IterationResult(
            iteration=iteration,
            timestamp_kst=datetime.now(KST).isoformat(),
            dry_run=args.dry_run,
            prompt_path=relpath(prompt_path, root),
            opencode_jsonl=None if args.dry_run else relpath(jsonl_path, root),
            raw_text_path=relpath(raw_text_path, root),
            expected_report_json=relpath(report_json, root),
            expected_report_md=relpath(report_md, root),
            opencode_exit_code=exit_code,
            opencode_timed_out=timed_out,
            sentinel=sentinel,
            hermes_verdict=verdict,
            strict_pass=strict_pass,
            rejection_reasons=rejection_reasons,
            safety_issues=safety_issues,
            metrics_checked=metrics_checked,
            new_probe=new_probe,
            new_submission_csvs=new_csvs,
        )
        results.append(result)
        rejection_history.append({
            "iteration": iteration,
            "sentinel": sentinel,
            "hermes_verdict": verdict,
            "rejection_reasons": rejection_reasons[:8],
            "safety_issues": safety_issues[:8],
            "metrics_checked": metrics_checked[:3],
        })
        log(f"iter {iteration}: sentinel={sentinel} hermes_verdict={verdict} strict={strict_pass}")
        if strict_pass:
            final_verdict = verdict
            break
        if verdict == "PROBE_STILL_RUNNING":
            final_verdict = verdict
            break
        if args.dry_run:
            final_verdict = "DRY_RUN_COMPLETE"
            break
        if iteration < args.max_iters:
            time.sleep(max(0, args.sleep_between_iters_sec))
    else:
        final_verdict = "MAX_ITERS_REACHED_NO_STRICT_PASS"

    summary_json, summary_md = write_summary(root, run_ts, results, final_verdict, args.dry_run)
    log(f"controller complete final_verdict={final_verdict}")
    log(f"summary_json={relpath(summary_json, root)}")
    log(f"summary_md={relpath(summary_md, root)}")
    return 0 if final_verdict not in {"STRICT_PASS"} else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("received KeyboardInterrupt; exiting")
        raise SystemExit(130)
