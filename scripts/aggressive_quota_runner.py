#!/usr/bin/env python3
"""Continuous aggressive quota runner for KMURecSys26 Steam.

This runner is an autonomous submission worker for the 2026-06-15 deadline. Kaggle submit
runs automatically when candidate, quota, and safety gates pass. It still never burns the
daily quota in one rapid batch, never submits exact/near-identical tuned CSVs, and always
runs post-submission calibration before the next submission cycle. Hard safety gates remain
mandatory: schema/order/top-half preflight, duplicate/similarity guards, no hidden-label
access, no external scraping, no credential printing, and GitHub/W&B logging for each
submission.
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
POLICY_PATH = ROOT / "state/autonomous_submission_policy.json"
KST = ZoneInfo("Asia/Seoul")
UTC = dt.timezone.utc
WANDB_ENTITY = "mrpc2003-kookmin-university"
WANDB_PROJECT = "kmu-rec-sys-26-steam"
MIN_AUTONOMOUS_ROW_DIFF = 500
HARD_DUPLICATE_ROW_DIFF = 200
DEFAULT_SLEEP_AFTER_SUBMIT = 6 * 60 * 60
CURRENT_PUBLIC_BEST_FALLBACK = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"


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


def load_validation_cache() -> dict[str, Any]:
    """Load cached validation variants; recompute only if the cache is unavailable.

    The full validation grid is deterministic and already persisted by
    materialize_readme_rankblend_residual.py. Reusing it prevents the long-lived runner from
    spending minutes in repeated pandas groupby work between quota submissions.
    """
    p = ROOT / "reports/20260602_readme_rankblend_residual_materialization.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        validation = data.get("validation")
        if isinstance(validation, dict) and validation.get("all_variants"):
            return validation
    return mat.validate_variants()


def choose_next_variant(state: dict[str, Any], submitted_names: set[str]) -> dict[str, Any] | None:
    validation = load_validation_cache()
    skipped = state.setdefault("skipped_variants", [])
    tried = set(state.get("submitted_variants", [])) | set(skipped)
    quarantined = state.get("quarantined_families", {})
    for v in validation["all_variants"]:
        if not v.get("manual_risk_signal"):
            continue
        if v["variant"] in tried:
            continue
        if variant_family(v["variant"]) in quarantined:
            if v["variant"] not in skipped:
                skipped.append(v["variant"])
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
        if variant_family(v["variant"]) in quarantined:
            if v["variant"] not in skipped:
                skipped.append(v["variant"])
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


def variant_family(variant: str) -> str:
    """Group nearby weight/popularity-only tunes so one failed probe blocks siblings.

    Example:
    rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.025
      -> rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa*
    """
    base = re.sub(r"_w\d+(?:\.\d+)?$", "", variant)
    return re.sub(r"_popa\d+$", "_popa*", base)


def read_submission_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    label_col = "Played" if "Played" in df.columns else "Label" if "Label" in df.columns else None
    if label_col is None:
        raise ValueError(f"No Played/Label column in {path}")
    out = df[["ID", label_col]].copy()
    out.columns = ["ID", "Played"]
    out["Played"] = out["Played"].astype(int)
    return out


def row_diff_count(a: Path, b: Path) -> int:
    left = read_submission_labels(a)
    right = read_submission_labels(b)
    merged = left.merge(right, on="ID", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(left):
        raise ValueError(f"Row mismatch while comparing {a} and {b}")
    return int((merged["Played_a"] != merged["Played_b"]).sum())


def local_submission_path(file_name: str | None) -> Path | None:
    if not file_name:
        return None
    p = ROOT / "submissions" / file_name
    return p if p.exists() else None


def reference_paths(state: dict[str, Any], q: dict[str, Any], *, exclude_sha: str | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add(label: str, path: Path | None, meta: dict[str, Any] | None = None) -> None:
        if path is None or not path.exists():
            return
        key = str(path.resolve())
        if key in seen_paths:
            return
        try:
            sha = sha256_file(path)
        except Exception:
            sha = None
        if exclude_sha and sha == exclude_sha:
            return
        seen_paths.add(key)
        refs.append({"label": label, "path": path, "sha256": sha, "meta": meta or {}})

    best = q.get("best_row") or {}
    add("live_public_best", local_submission_path(best.get("fileName")), best)
    add("fallback_public_best", CURRENT_PUBLIC_BEST_FALLBACK, {"fileName": CURRENT_PUBLIC_BEST_FALLBACK.name})
    for r in state.get("submission_results", []):
        p = Path(r["candidate_file"]) if r.get("candidate_file") else local_submission_path(r.get("fileName"))
        add(f"prior_submit:{r.get('variant') or r.get('fileName')}", p, r)
    return refs


def similarity_guard(path: Path, pf: dict[str, Any], variant: dict[str, Any], state: dict[str, Any], q: dict[str, Any]) -> dict[str, Any]:
    family = variant_family(variant["variant"])
    quarantined = state.get("quarantined_families", {})
    report: dict[str, Any] = {
        "variant": variant["variant"],
        "family": family,
        "sha256": pf["sha256"],
        "min_required_row_diff": MIN_AUTONOMOUS_ROW_DIFF,
        "hard_duplicate_row_diff": HARD_DUPLICATE_ROW_DIFF,
        "blocked": False,
        "block_reasons": [],
        "row_diffs": [],
    }
    if family in quarantined:
        report["blocked"] = True
        report["block_reasons"].append("family_quarantined_after_negative_transfer")
        report["quarantine"] = quarantined[family]

    seen_hashes = {r.get("sha256") for r in state.get("submission_results", []) if r.get("sha256")}
    if pf["sha256"] in seen_hashes:
        report["blocked"] = True
        report["block_reasons"].append("exact_sha_duplicate_of_prior_submission")

    for ref in reference_paths(state, q, exclude_sha=pf["sha256"]):
        try:
            diff = row_diff_count(path, ref["path"])
        except Exception as exc:
            report["row_diffs"].append({"label": ref["label"], "file": str(ref["path"]), "error": repr(exc)})
            continue
        item = {"label": ref["label"], "file": str(ref["path"]), "sha256": ref.get("sha256"), "row_diff": diff}
        report["row_diffs"].append(item)
        if diff < HARD_DUPLICATE_ROW_DIFF:
            report["blocked"] = True
            report["block_reasons"].append(f"hard_near_duplicate_{diff}_rows_vs_{ref['label']}")
        elif diff < MIN_AUTONOMOUS_ROW_DIFF:
            report["blocked"] = True
            report["block_reasons"].append(f"near_duplicate_{diff}_rows_vs_{ref['label']}")
    return report


def write_post_submission_analysis(
    *,
    ts: str,
    result: dict[str, Any],
    variant: dict[str, Any],
    state: dict[str, Any],
    q: dict[str, Any],
    candidate_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    family = variant_family(variant["variant"])
    validation_delta = float(variant.get("mean_delta_vs_rankblend") or 0.0)
    public_delta = result.get("delta_vs_previous_best")
    transfer_ratio = None
    if public_delta is not None and validation_delta:
        transfer_ratio = float(public_delta) / validation_delta
    row_diffs = []
    for ref in reference_paths(state, q, exclude_sha=result.get("sha256")):
        try:
            diff = row_diff_count(candidate_path, ref["path"])
            row_diffs.append({"label": ref["label"], "file": str(ref["path"]), "sha256": ref.get("sha256"), "row_diff": diff})
        except Exception as exc:
            row_diffs.append({"label": ref["label"], "file": str(ref["path"]), "error": repr(exc)})

    quarantine = False
    quarantine_reason = None
    if result.get("publicScore") is not None and public_delta is not None and public_delta <= 0:
        quarantine = True
        quarantine_reason = "non_improving_public_transfer"
    if transfer_ratio is not None and transfer_ratio <= 0:
        quarantine = True
        quarantine_reason = "negative_transfer_ratio"

    analysis = {
        "kind": "autorun_post_submission_calibration",
        "timestamp_kst": ts,
        "variant": variant["variant"],
        "family": family,
        "candidate_file": str(candidate_path),
        "sha256": result.get("sha256"),
        "publicScore": result.get("publicScore"),
        "previous_public_best": result.get("previous_public_best"),
        "public_delta_vs_previous_best": public_delta,
        "validation_mean_delta_vs_rankblend": validation_delta,
        "transfer_ratio_public_delta_over_validation_delta": transfer_ratio,
        "beat_previous_best": result.get("beat_previous_best"),
        "row_diffs_vs_references": row_diffs,
        "quarantine_family": quarantine,
        "quarantine_reason": quarantine_reason,
        "next_submission_policy": {
            "same_family_allowed": False if quarantine else "only_after_fresh_analysis",
            "minimum_row_diff_vs_any_prior_csv": MIN_AUTONOMOUS_ROW_DIFF,
            "rapid_batch_quota_burn_allowed": False,
        },
    }
    if quarantine:
        state.setdefault("quarantined_families", {})[family] = {
            "time_kst": now_kst().isoformat(),
            "reason": quarantine_reason,
            "source_variant": variant["variant"],
            "public_delta_vs_previous_best": public_delta,
            "validation_mean_delta_vs_rankblend": validation_delta,
            "transfer_ratio": transfer_ratio,
            "sha256": result.get("sha256"),
        }

    js = ROOT / f"reports/{ts}_autorun_post_submission_calibration.json"
    md = ROOT / f"reports/{ts}_autorun_post_submission_calibration.md"
    js.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Autorun post-submission calibration",
        "",
        f"- variant: `{variant['variant']}`",
        f"- family: `{family}`",
        f"- publicScore: `{result.get('publicScore')}`",
        f"- previous best: `{result.get('previous_public_best')}`",
        f"- public Δ vs previous best: `{public_delta}`",
        f"- validation mean Δ: `{validation_delta:+.6f}`",
        f"- transfer ratio: `{transfer_ratio}`",
        f"- quarantine family: `{quarantine}`",
        f"- quarantine reason: `{quarantine_reason}`",
        f"- rapid batch quota burn allowed: `False`",
        "",
        "## Row diffs vs references",
    ]
    for item in row_diffs:
        lines.append(f"- `{item.get('label')}`: row_diff=`{item.get('row_diff')}`, file=`{item.get('file')}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return js, md, analysis


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
            tags=["kaggle", "steam", "submission", "autorun", "manual-risk", "autonomous-submit"],
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
            "post_analysis/transfer_ratio": (result.get("post_analysis") or {}).get("transfer_ratio_public_delta_over_validation_delta"),
            "similarity/min_row_diff": min(
                [x.get("row_diff") for x in (result.get("similarity_guard") or {}).get("row_diffs", []) if isinstance(x.get("row_diff"), int)],
                default=None,
            ),
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
    sim = similarity_guard(out_path, pf, variant, state, q)
    pre_path = ROOT / f"reports/{ts}_autorun_preflight.json"
    sim_path = ROOT / f"reports/{ts}_autorun_similarity_guard.json"
    pre_payload = {
        "variant": variant,
        "materialized": mat_info,
        "preflight": pf,
        "similarity_guard": sim,
        "quota": q,
        "pre_submissions_csv": str(subs_path),
    }
    pre_path.write_text(json.dumps(pre_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sim_path.write_text(json.dumps(sim, indent=2, ensure_ascii=False), encoding="utf-8")
    if not preflight_ok(pf):
        state.setdefault("notes", []).append(f"preflight failed {variant['variant']} at {ts}")
        save_state(state)
        log(f"preflight failed for {filename}: {pf}")
        return False
    if sim.get("blocked"):
        state.setdefault("skipped_variants", []).append(variant["variant"])
        state.setdefault("notes", []).append(
            f"similarity guard blocked {variant['variant']} at {ts}: {sim.get('block_reasons')}"
        )
        save_state(state)
        git_commit([pre_path, sim_path, STATE_PATH], f"chore: block near-duplicate autorun candidate {ts}")
        log(f"similarity guard blocked {filename}: {sim.get('block_reasons')}")
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
        "family": variant_family(variant["variant"]),
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
        "similarity_guard": sim,
        "submit_meta": submit_meta,
        "post_submissions_csv": str(post_path),
        "safety": {
            "hidden_label_access": False,
            "external_steam_scraping": False,
            "autonomous_submission_enabled": True,
            "manual_gate_required": False,
        },
    }
    analysis_json, analysis_md, analysis = write_post_submission_analysis(
        ts=ts,
        result=result,
        variant=variant,
        state=state,
        q=q,
        candidate_path=out_path,
    )
    result["post_analysis"] = analysis
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
    wandb_url = log_wandb(result, [pre_path, sim_path, analysis_json, analysis_md, res_json, res_md, subs_path, post_path])
    if wandb_url:
        result["wandb_url"] = wandb_url
        res_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    state.setdefault("submitted_variants", []).append(variant["variant"])
    state.setdefault("submission_results", []).append({
        "variant": variant["variant"], "family": variant_family(variant["variant"]), "fileName": filename,
        "candidate_file": str(out_path), "publicScore": public_score,
        "previous_public_best": prev, "delta_vs_previous_best": result["delta_vs_previous_best"],
        "validation_mean_delta_vs_rankblend": variant.get("mean_delta_vs_rankblend"),
        "transfer_ratio": analysis.get("transfer_ratio_public_delta_over_validation_delta"),
        "post_analysis_json": str(analysis_json), "post_analysis_md": str(analysis_md),
        "timestamp_kst": ts, "sha256": pf["sha256"], "wandb_url": wandb_url,
    })
    state["last_run_kst"] = now_kst().isoformat()
    state["last_public_best_seen"] = max([x for x in [prev, public_score] if x is not None], default=prev)
    save_state(state)
    git_commit([pre_path, sim_path, analysis_json, analysis_md, res_json, res_md, subs_path, post_path, STATE_PATH], f"chore: log autorun submission {ts}")
    if beat:
        print(f"NEW_PUBLIC_BEST {public_score} {filename}", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deadline-kst", default="2026-06-15T23:59:59+09:00")
    ap.add_argument("--sleep-no-quota", type=int, default=300)
    ap.add_argument("--sleep-no-candidate", type=int, default=600)
    ap.add_argument("--sleep-after-submit", type=int, default=DEFAULT_SLEEP_AFTER_SUBMIT,
                    help="Cooldown after any completed submission; prevents rapid multi-submit batches.")
    ap.add_argument("--max-submissions", type=int, default=None, help="Optional cap for this process; omit for continuous until deadline.")
    args = ap.parse_args()
    if not POLICY_PATH.exists():
        raise SystemExit(f"Missing autonomous submission policy: {POLICY_PATH}")
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
                # User correction on 2026-06-02: do not burn multiple similar quota slots in
                # one rapid batch. Post-analysis is written inside run_one_submission(); now
                # cool down before any further candidate search/submission cycle.
                log(f"post-submission cooldown sleep={args.sleep_after_submit}s")
                time.sleep(args.sleep_after_submit)
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
