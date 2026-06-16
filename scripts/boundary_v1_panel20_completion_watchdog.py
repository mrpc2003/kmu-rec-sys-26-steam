#!/usr/bin/env python3
"""Wait for boundary v1 panel20 score coverage, then run scored cross-fit eval.

NO-SUBMIT watchdog.  It only observes score artifacts/status and runs
`scripts/boundary_v1_scored_crossfit_eval.py` after all coverage jobs complete.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = ROOT / "reports/boundary_v1_panel20_score_coverage_status"
SUMMARY = STATUS_DIR / "summary.json"
WATCHDOG_JSONL = STATUS_DIR / "completion_watchdog.jsonl"
DONE = STATUS_DIR / "completion_watchdog.done"
CROSSFIT_CMD = [
    sys.executable,
    str(ROOT / "scripts/boundary_v1_scored_crossfit_eval.py"),
    "--min-splits",
    "20",
]


def append(payload: dict[str, Any]) -> None:
    WATCHDOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def read_summary() -> dict[str, Any] | None:
    if not SUMMARY.exists():
        return None
    try:
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    except Exception as exc:
        append({"event": "summary_read_error", "error": str(exc)})
        return None


def main() -> None:
    poll_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    append({"event": "watchdog_start", "summary": str(SUMMARY), "poll_seconds": poll_seconds})
    while True:
        summary = read_summary()
        if summary:
            total = int(summary.get("jobs_total", 0) or 0)
            complete = int(summary.get("jobs_complete", 0) or 0)
            append({"event": "poll", "jobs_total": total, "jobs_complete": complete})
            if total > 0 and complete >= total:
                break
        time.sleep(poll_seconds)
    append({"event": "coverage_complete", "cmd": CROSSFIT_CMD})
    log_path = STATUS_DIR / "crossfit_after_completion.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(CROSSFIT_CMD, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
    payload = {"event": "crossfit_done", "returncode": proc.returncode, "log": str(log_path)}
    append(payload)
    DONE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
