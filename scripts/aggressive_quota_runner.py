#!/usr/bin/env python3
"""Continuous aggressive quota runner for KMURecSys26 Steam.

This runner is enabled only because the user explicitly granted standing approval to
submit any candidate with even a slight validation-positive signal through the 2026-06-15
deadline and to consume the daily Kaggle quota. It still enforces hard safety gates:
preflight schema/order/top-half checks, no duplicate variant, no hidden-label access, no
external scraping, no credential printing, and GitHub/W&B logging for each submission.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import materialize_readme_rankblend_residual as mat  # noqa: E402

COMP = "kmu-rec-sys-26-steam"
STATE_PATH = ROOT / "state/aggressive_quota_runner_state.json"
POLICY_PATH = ROOT / "state/standing_approval_policy.json"
KST = ZoneInfo("Asia/Seoul")
UTC = dt.timezone.utc
WANDB_ENTITY = "mrpc2003-kookmin-university"
WANDB_PROJECT = "kmu-rec-sys-26-steam"


def now_kst() -> dt.datetime:
    return dt.datetime.now(KST)


def stamp() -> str:
    return now_kst().strftime("%Y%m%dT%H%M%SKST")


def log(msg: str) -> None:
    print(f"[{now_kst().isoformat(timespec='seconds')}] {msg}", flush=True)


def sh(cmd: list[str], *, timeout: int = 300, cwd: Path = ROOT, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("HOME", "/opt/data/home")
    env.setdefault("PATH", "/opt/data/home/.local/bin:/usr/local/bin:/usr/bin:/bin")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"submitted_variants": [], "submission_results": [], "notes": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_submissions(ts: str) -> tuple[Path, list[dict[str, str]]]:
    out = ROOT / f"reports/{ts}_autorun_submissions.csv"
    cp = sh(["/opt/data/home/.local/bin/kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "200"], timeout=120)
    out.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode != 0:
        (ROOT / f"reports/{ts}_autorun_submissions_stderr.txt").write_text(cp.stderr, encoding="utf-8")
        raise RuntimeError(f"kaggle submissions failed: {cp.returncode}")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    return out, rows


def quota_and_best(rows: list[dict[str, str]]) -> dict[str, Any]:
    today_utc = dt.datetime.now(UTC).date()
    used = 0
    best_score = None
    best_row = None
    for r in rows:
        ds = (r.get("date") or "")[:10]
        try:
            d = dt.date.fromisoformat(ds)
        except Exception:
            d = None
        if d == today_utc and r.get("status") in {"SubmissionStatus.COMPLETE", "SubmissionStatus.PENDING"}:
            used += 1
        try:
            s = float(r.get("publicScore") or "nan")
        except Exception:
            continue
        if s == s and (best_score is None or s > best_score):
            best_score = s
            best_row = r
    return {"used_today_utc": used, "remaining": max(0, 5 - used), "best_score": best_score, "best_row": best_row}


def safe_variant_filename(variant: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant)[:150]
    return f"candidate_autorun_{safe}.csv"


def choose_next_variant(state: dict[str, Any], submitted_names: set[str]) -> dict[str, Any] | None:
    validation = mat.validate_variants()
    tried = set(state.get("submitted_variants", []))
    for v in validation["all_variants"]:
        if not v.get("manual_risk_signal"):
            continue
        if v["variant"] in tried:
            continue
        if safe_variant_filename(v["variant"]) in submitted_names:
            state.setdefault("submitted_variants", []).append(v["variant"])
            continue
        return v
    # If all manual-risk variants are consumed, still keep only validation-positive variants.
    for v in validation["all_variants"]:
        if v["variant"] in tried:
            continue
        if v.get("mean_delta_vs_rankblend", 0.0) <= 0:
            continue
        if safe_variant_filename(v["variant"]) in submitted_names:
            state.setdefault("submitted_variants", []).append(v["variant"])
            continue
        return v
    return None


def preflight_file(path: Path) -> dict[str, Any]:
    sub = pd.read_csv(path)
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    merged = pairs[["ID", "userID"]].merge(sub, on="ID", validate="one_to_one")
    g = merged.groupby("userID", sort=False)["Played"].agg(["sum", "count"])
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "columns": sub.columns.tolist(),
        "rows": int(len(sub)),
        "expected_rows": int(len(pairs)),
        "id_unique": bool(sub["ID"].is_unique),
        "id_order_matches_pairs": bool(sub["ID"].tolist() == pairs["ID"].tolist()),
        "labels_binary": bool(set(sub["Played"].astype(int).unique()).issubset({0, 1})),
        "label_1": int(sub["Played"].sum()),
        "label_0": int(len(sub) - sub["Played"].sum()),
        "bad_users_tophalf": int(((g["count"] % 2 != 0) | (g["sum"] != g["count"] // 2)).sum()),
    }


def preflight_ok(pf: dict[str, Any]) -> bool:
    return (
        pf["columns"] == ["ID", "Played"]
        and pf["rows"] == pf["expected_rows"] == 19998
        and pf["id_unique"]
        and pf["id_order_matches_pairs"]
        and pf["labels_binary"]
        and pf["label_1"] == pf["label_0"] == 9999
        and pf["bad_users_tophalf"] == 0
    )


def submit_candidate(path: Path, variant: dict[str, Any], q: dict[str, Any], ts: str) -> dict[str, Any]:
    msg = (
        f"AUTO aggressive quota: {variant['variant']}. "
        f"3split Δvs rankblend {variant['mean_delta_vs_rankblend']:+.6f}, "
        f"pos {variant['positive_splits']}/3, fixes/breaks {variant['fixes']}/{variant['breaks']}, "
        f"p={variant['pooled_p_exact']:.4g}, SHA {sha256_file(path)[:8]}."
    )[:240]
    stdout_path = ROOT / f"reports/{ts}_autorun_submit_stdout.txt"
    stderr_path = ROOT / f"reports/{ts}_autorun_submit_stderr.txt"
    cp = sh(["/opt/data/home/.local/bin/kaggle", "competitions", "submit", "-c", COMP, "-f", str(path), "-m", msg], timeout=240)
    stdout_path.write_text(cp.stdout, encoding="utf-8")
    stderr_path.write_text(cp.stderr, encoding="utf-8")
    if cp.returncode != 0:
        raise RuntimeError(f"kaggle submit failed rc={cp.returncode}; see {stderr_path}")
    print("KAGGLE_SUBMITTED " + path.name, flush=True)
    return {"submit_stdout": str(stdout_path), "submit_stderr": str(stderr_path), "message": msg}


def poll_result(filename: str, ts: str, *, max_wait_s: int = 600) -> tuple[Path, dict[str, Any]]:
    deadline = time.time() + max_wait_s
    post = ROOT / f"reports/{ts}_autorun_post_submissions.csv"
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        _, rows = refresh_submissions(ts + "_poll")
        # Copy latest poll csv to canonical post path.
        latest = ROOT / f"reports/{ts}_poll_autorun_submissions.csv"
        if latest.exists():
            post.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
        for r in rows:
            if r.get("fileName") == filename:
                last = r
                break
        if last:
            status = str(last.get("status"))
            log(f"poll {filename}: {status} public={last.get('publicScore')}")
            if status in {"SubmissionStatus.COMPLETE", "SubmissionStatus.ERROR"}:
                return post, last
        time.sleep(10)
    if last:
        return post, last
    raise TimeoutError(f"No submission result found for {filename}")


def log_wandb(result: dict[str, Any], report_files: list[Path]) -> str | None:
    try:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"autorun-submit-{result['timestamp_kst']}",
            group="aggressive-quota-autorun",
            job_type="kaggle-submission",
            tags=["kaggle", "steam", "submission", "autorun", "manual-risk", "standing-approval"],
            notes="Aggressive quota autorun submission result. No secrets logged.",
            config={
                "variant": result.get("variant"),
                "candidate_file": result.get("candidate_file"),
                "sha256": result.get("sha256"),
            },
        )
        ev = result.get("validation_evidence", {})
        pf = result.get("preflight", {})
        metrics = {
            "public/score": result.get("publicScore"),
            "public/previous_best": result.get("previous_public_best"),
            "public/delta_vs_previous_best": result.get("delta_vs_previous_best"),
            "public/beat_previous_best": int(bool(result.get("beat_previous_best"))),
            "validation/mean_delta_vs_rankblend": ev.get("mean_delta_vs_rankblend"),
            "validation/fixes": ev.get("fixes"),
            "validation/breaks": ev.get("breaks"),
            "validation/pooled_p_exact": ev.get("pooled_p_exact"),
            "preflight/rows": pf.get("rows"),
            "preflight/bad_users_tophalf": pf.get("bad_users_tophalf"),
        }
        wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float, bool)) and v == v})
        art = wandb.Artifact(f"autorun_submission_{result['timestamp_kst']}", type="submission-report", metadata={"sha256": result.get("sha256"), "publicScore": result.get("publicScore")})
        for p in report_files:
            if p.exists() and p.is_file():
                art.add_file(str(p), name=str(p.relative_to(ROOT)))
        run.log_artifact(art)
        url = run.url
        run.finish()
        return url
    except Exception as exc:
        log(f"W&B logging failed: {exc}")
        return None


def git_commit(paths: list[Path], message: str) -> None:
    existing = [str(p.relative_to(ROOT)) for p in paths if p.exists()]
    if not existing:
        return
    # Avoid committing submission CSVs/artifacts; reports/scripts/state only.
    allow = [p for p in existing if p.startswith("reports/") or p.startswith("scripts/") or p.startswith("state/")]
    if not allow:
        return
    sh(["git", "add", *allow], timeout=120)
    cp = sh(["git", "diff", "--cached", "--quiet"], timeout=60)
    if cp.returncode == 0:
        return
    commit = sh(["git", "commit", "-m", message], timeout=180)
    if commit.returncode != 0:
        log(f"git commit failed: {commit.stderr[:500]}")
        return
    push = sh(["git", "push", "origin", "main"], timeout=240)
    if push.returncode != 0:
        log(f"git push failed: {push.stderr[:500]}")


def run_one_submission(state: dict[str, Any]) -> bool:
    ts = stamp()
    subs_path, rows = refresh_submissions(ts)
    q = quota_and_best(rows)
    log(f"quota used={q['used_today_utc']}/5 remaining={q['remaining']} best={q['best_score']} file={q['best_row'].get('fileName') if q['best_row'] else None}")
    if q["remaining"] <= 0:
        return False
    submitted_names = {r.get("fileName", "") for r in rows}
    variant = choose_next_variant(state, submitted_names)
    if variant is None:
        log("No validation-positive unsubmitted variant available; waiting for new experiment axes.")
        return False
    filename = safe_variant_filename(variant["variant"])
    out_path = ROOT / "submissions" / filename
    mat_info = mat.materialize(variant["variant"], out_path)
    pf = preflight_file(out_path)
    pre_path = ROOT / f"reports/{ts}_autorun_preflight.json"
    pre_payload = {"variant": variant, "materialized": mat_info, "preflight": pf, "quota": q, "pre_submissions_csv": str(subs_path)}
    pre_path.write_text(json.dumps(pre_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not preflight_ok(pf):
        state.setdefault("notes", []).append(f"preflight failed {variant['variant']} at {ts}")
        save_state(state)
        log(f"preflight failed for {filename}: {pf}")
        return False
    submit_meta = submit_candidate(out_path, variant, q, ts)
    post_path, row = poll_result(filename, ts)
    public_score = None
    try:
        public_score = float(row.get("publicScore") or "nan")
        if public_score != public_score:
            public_score = None
    except Exception:
        public_score = None
    prev = q.get("best_score")
    beat = public_score is not None and prev is not None and public_score > prev
    result = {
        "kind": "aggressive_quota_autorun_submission_result",
        "timestamp_kst": ts,
        "variant": variant["variant"],
        "candidate_file": str(out_path),
        "fileName": filename,
        "sha256": pf["sha256"],
        "status": row.get("status"),
        "date_utc": row.get("date"),
        "publicScore": public_score,
        "previous_public_best": prev,
        "delta_vs_previous_best": (public_score - prev) if public_score is not None and prev is not None else None,
        "beat_previous_best": beat,
        "validation_evidence": {
            "mean_delta_vs_rankblend": variant["mean_delta_vs_rankblend"],
            "positive_splits": variant["positive_splits"],
            "fixes": variant["fixes"],
            "breaks": variant["breaks"],
            "pooled_p_exact": variant["pooled_p_exact"],
            "fisher_p": variant.get("fisher_p"),
        },
        "preflight": pf,
        "submit_meta": submit_meta,
        "post_submissions_csv": str(post_path),
        "safety": {"hidden_label_access": False, "external_steam_scraping": False, "standing_user_approval": True},
    }
    res_json = ROOT / f"reports/{ts}_autorun_submission_result.json"
    res_md = ROOT / f"reports/{ts}_autorun_submission_result.md"
    res_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    res_md.write_text(
        "\n".join([
            "# Aggressive quota autorun submission result",
            "",
            f"- variant: `{variant['variant']}`",
            f"- file: `{filename}`",
            f"- status: `{row.get('status')}`",
            f"- publicScore: `{public_score}`",
            f"- previous best: `{prev}`",
            f"- delta vs previous best: `{result['delta_vs_previous_best']}`",
            f"- validation mean Δ vs rankblend: `{variant['mean_delta_vs_rankblend']:+.6f}`",
            f"- fixes/breaks: `{variant['fixes']}/{variant['breaks']}`",
            f"- pooled p: `{variant['pooled_p_exact']:.4g}`",
        ]) + "\n",
        encoding="utf-8",
    )
    wandb_url = log_wandb(result, [pre_path, res_json, res_md, subs_path, post_path])
    if wandb_url:
        result["wandb_url"] = wandb_url
        res_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    state.setdefault("submitted_variants", []).append(variant["variant"])
    state.setdefault("submission_results", []).append({
        "variant": variant["variant"], "fileName": filename, "publicScore": public_score,
        "previous_public_best": prev, "delta_vs_previous_best": result["delta_vs_previous_best"],
        "timestamp_kst": ts, "sha256": pf["sha256"], "wandb_url": wandb_url,
    })
    state["last_run_kst"] = now_kst().isoformat()
    state["last_public_best_seen"] = max([x for x in [prev, public_score] if x is not None], default=prev)
    save_state(state)
    git_commit([pre_path, res_json, res_md, subs_path, post_path, STATE_PATH], f"chore: log autorun submission {ts}")
    if beat:
        print(f"NEW_PUBLIC_BEST {public_score} {filename}", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deadline-kst", default="2026-06-15T23:59:59+09:00")
    ap.add_argument("--sleep-no-quota", type=int, default=300)
    ap.add_argument("--sleep-no-candidate", type=int, default=600)
    ap.add_argument("--max-submissions", type=int, default=None, help="Optional cap for this process; omit for continuous until deadline.")
    args = ap.parse_args()
    if not POLICY_PATH.exists():
        raise SystemExit(f"Missing standing approval policy: {POLICY_PATH}")
    deadline = dt.datetime.fromisoformat(args.deadline_kst)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=KST)
    state = load_state()
    submissions_done = 0
    log(f"runner start deadline={deadline.isoformat()} max_submissions={args.max_submissions}")
    while now_kst() <= deadline:
        try:
            did = run_one_submission(state)
            if did:
                submissions_done += 1
                if args.max_submissions is not None and submissions_done >= args.max_submissions:
                    log("max submissions reached; exiting")
                    return
                # Immediately re-check quota and continue; user asked for quota burn, not hourly cadence.
                time.sleep(5)
            else:
                # Determine whether quota is exhausted or no candidate; do not spin hot.
                ts = stamp()
                _, rows = refresh_submissions(ts + "_idle")
                q = quota_and_best(rows)
                delay = args.sleep_no_quota if q["remaining"] <= 0 else args.sleep_no_candidate
                log(f"idle remaining={q['remaining']} sleep={delay}s")
                time.sleep(delay)
        except Exception as exc:
            print(f"RUNNER_ERROR {type(exc).__name__}: {exc}", flush=True)
            state = load_state()
            state.setdefault("errors", []).append({"time_kst": now_kst().isoformat(), "error": repr(exc)})
            save_state(state)
            time.sleep(120)
    log("deadline reached; runner exiting")


if __name__ == "__main__":
    main()
