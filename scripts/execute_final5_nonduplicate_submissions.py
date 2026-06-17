#!/usr/bin/env python3
"""Execute the approved final-five non-duplicate Kaggle submissions.

This script is intentionally narrow and competition-specific. It materializes five
preselected, already-computed label files into upload-safe `ID,Label` CSVs, checks
live Kaggle submissions for filename/SHA/label-duplicate guards, submits one file at
a time, polls the public score, and writes JSON/Markdown reports.

It does not create new model predictions and does not use external metadata or hidden
labels. It only consumes the user-approved remaining quota slots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_BIN = Path("/opt/data/home/.local/bin/kaggle")
COMPETITION = "kmu-rec-sys-26-steam"
EXPECTED_ROWS = 19998
DAILY_QUOTA = 5
PUBLIC_BEST = 0.77825

SELECTED = [
    {
        "slot": "slot1",
        "source": "submissions/candidate_smoke_tagcf_fulltest.csv",
        "upload_name": "final5_slot1_tagcf_smoke_LABEL.csv",
        "role": "TAGCF smoke graph-family lottery; largest non-duplicate TAGCF variant",
    },
    {
        "slot": "slot2",
        "source": "artifacts/scores/test_pairs_full_train_stage2_cf/prediction_csv/candidate_score_als_f32_it30_alpha20_popa2.csv",
        "upload_name": "final5_slot2_pure_als_popa2_LABEL.csv",
        "role": "Pure ALS CF baseline distinct from rankblend/ALS-residual submissions",
    },
    {
        "slot": "slot3",
        "source": "artifacts/scores/test_pairs_full_train_stage2_itemease/prediction_csv/candidate_score_itemknn_bm25_max.csv",
        "upload_name": "final5_slot3_itemknn_bm25_max_LABEL.csv",
        "role": "Pure itemKNN BM25 max baseline; high row-diff classical CF hedge",
    },
    {
        "slot": "slot4",
        "source": "artifacts/scores/test_pairs_full_train_stage2_itemease/prediction_csv/candidate_score_rrf_pop_itemknn_ease.csv",
        "upload_name": "final5_slot4_rrf_pop_itemknn_ease_LABEL.csv",
        "role": "RRF popularity+itemKNN+EASE hybrid, distinct from pure ALS/itemKNN",
    },
    {
        "slot": "slot5",
        "source": "artifacts/scores/test_pairs_full_train_stage2_blend/prediction_csv/candidate_score_blend_median_z.csv",
        "upload_name": "final5_slot5_stage2_median_z_LABEL.csv",
        "role": "Stage2 median-z blend baseline distinct from submitted mean-z baseline",
    },
]


@dataclass
class LiveSubmission:
    fileName: str
    date: str
    description: str
    status: str
    publicScore: str
    privateScore: str


class SubmissionError(RuntimeError):
    pass


def now_kst_stamp() -> str:
    # Avoid requiring pytz/zoneinfo details; KST is UTC+9 and the host normally has TZ data.
    return subprocess.check_output(["bash", "-lc", "TZ=Asia/Seoul date +%Y%m%dT%H%M%SKST"], text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = "/opt/data/home"
    return env


def run_kaggle(args: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    if not KAGGLE_BIN.exists():
        raise FileNotFoundError(KAGGLE_BIN)
    cmd = [str(KAGGLE_BIN), *args]
    return subprocess.run(cmd, cwd=ROOT, env=kaggle_env(), text=True, capture_output=True, timeout=timeout)


def parse_submissions_csv_text(text: str) -> list[LiveSubmission]:
    lines = text.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.startswith("fileName,"))
    except StopIteration as exc:
        raise SubmissionError(f"Could not find Kaggle CSV header in output: {text[:500]!r}") from exc
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for r in reader:
        rows.append(
            LiveSubmission(
                fileName=r.get("fileName", ""),
                date=r.get("date", ""),
                description=r.get("description", ""),
                status=r.get("status", ""),
                publicScore=r.get("publicScore", ""),
                privateScore=r.get("privateScore", ""),
            )
        )
    return rows


def fetch_live_submissions(out_path: Path) -> list[LiveSubmission]:
    cp = run_kaggle(["competitions", "submissions", "-c", COMPETITION, "-v", "--page-size", "200"], timeout=180)
    combined = (cp.stdout or "") + ("\n--- STDERR ---\n" + cp.stderr if cp.stderr else "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")
    if cp.returncode != 0:
        raise SubmissionError(f"kaggle submissions failed rc={cp.returncode}; see {out_path}")
    return parse_submissions_csv_text(cp.stdout)


def count_today_utc(rows: list[LiveSubmission]) -> int:
    today = datetime.now(timezone.utc).date()
    count = 0
    for r in rows:
        if r.status not in {"SubmissionStatus.COMPLETE", "SubmissionStatus.PENDING"}:
            continue
        if not r.date:
            continue
        # Kaggle CLI date has no explicit timezone; treat it as UTC, matching quota reset semantics.
        try:
            dt = datetime.fromisoformat(r.date.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.date() == today:
            count += 1
    return count


def read_submission_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise SubmissionError(f"{path} missing ID column")
    label_col = "Label" if "Label" in df.columns else "Played" if "Played" in df.columns else None
    if label_col is None:
        raise SubmissionError(f"{path} missing Label/Played column")
    if len(df) != EXPECTED_ROWS:
        raise SubmissionError(f"{path} has {len(df)} rows, expected {EXPECTED_ROWS}")
    out = df[["ID", label_col]].rename(columns={label_col: "Label"}).copy()
    out = out.sort_values("ID", kind="mergesort").reset_index(drop=True)
    expected_ids = list(range(EXPECTED_ROWS))
    if out["ID"].tolist() != expected_ids:
        raise SubmissionError(f"{path} IDs are not contiguous 0..{EXPECTED_ROWS - 1}")
    try:
        out["Label"] = out["Label"].astype(int)
    except Exception as exc:
        raise SubmissionError(f"{path} labels are not integer-like") from exc
    uniq = set(out["Label"].unique().tolist())
    if not uniq.issubset({0, 1}):
        raise SubmissionError(f"{path} labels are not binary: {sorted(uniq)}")
    return out


def materialize_upload(source: Path, upload: Path, pairs: pd.DataFrame) -> dict[str, Any]:
    labels = read_submission_labels(source)
    if len(pairs) != EXPECTED_ROWS or pairs["ID"].tolist() != list(range(EXPECTED_ROWS)):
        raise SubmissionError("pairs.csv shape/order mismatch")
    chk = pairs[["ID", "userID"]].merge(labels, on="ID", validate="one_to_one")
    per_user = chk.groupby("userID", sort=False)["Label"].agg(["sum", "count"])
    bad_users = int((per_user["sum"] != (per_user["count"] // 2)).sum())
    upload.parent.mkdir(parents=True, exist_ok=True)
    tmp = upload.with_suffix(upload.suffix + ".tmp")
    labels.to_csv(tmp, index=False)
    tmp.replace(upload)
    return {
        "source": str(source.relative_to(ROOT)),
        "upload": str(upload.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "upload_sha256": sha256_file(upload),
        "rows": int(len(labels)),
        "label_1": int(labels["Label"].sum()),
        "label_0": int(len(labels) - labels["Label"].sum()),
        "bad_users_tophalf": bad_users,
    }


def local_live_predictions(live_names: set[str]) -> list[dict[str, Any]]:
    roots = [ROOT / "submissions", ROOT / "final_package", ROOT / "artifacts"]
    candidates: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        if base.name == "artifacts":
            candidates.extend(base.glob("**/candidate*.csv"))
        else:
            candidates.extend(base.glob("*.csv"))
    records = []
    for p in sorted(set(candidates)):
        if p.name not in live_names:
            continue
        try:
            labels = read_submission_labels(p)
        except Exception:
            continue
        records.append({"name": p.name, "path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "labels": labels["Label"].to_numpy(dtype="int8")})
    return records


def duplicate_checks(upload: Path, live_rows: list[LiveSubmission], live_preds: list[dict[str, Any]]) -> dict[str, Any]:
    live_names = {r.fileName for r in live_rows}
    labels = read_submission_labels(upload)["Label"].to_numpy(dtype="int8")
    exact_name = upload.name in live_names
    upload_sha = sha256_file(upload)
    exact_sha = any(upload_sha == r["sha256"] for r in live_preds)
    diffs = []
    for r in live_preds:
        diffs.append({"name": r["name"], "rowdiff": int((labels != r["labels"]).sum())})
    diffs.sort(key=lambda x: x["rowdiff"])
    return {
        "filename_already_submitted": exact_name,
        "exact_upload_sha_already_submitted_locally": exact_sha,
        "min_rowdiff_vs_live": diffs[0]["rowdiff"] if diffs else None,
        "nearest_live_filename": diffs[0]["name"] if diffs else None,
    }


def submit_one(upload: Path, message: str, stdout_path: Path) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    cp = run_kaggle(["competitions", "submit", "-c", COMPETITION, "-f", str(upload), "-m", message], timeout=300)
    stdout_path.write_text(
        "COMMAND: kaggle competitions submit -c {comp} -f {file} -m <message>\n".format(comp=COMPETITION, file=upload)
        + "RETURN_CODE: "
        + str(cp.returncode)
        + "\n\nSTDOUT:\n"
        + (cp.stdout or "")
        + "\nSTDERR:\n"
        + (cp.stderr or ""),
        encoding="utf-8",
    )
    return cp


def poll_result(upload_name: str, prefix: Path, *, max_attempts: int = 30, sleep_s: int = 20) -> dict[str, Any]:
    history = []
    for attempt in range(1, max_attempts + 1):
        out_csv = prefix.with_name(prefix.name + f"_poll{attempt:02d}.csv")
        rows = fetch_live_submissions(out_csv)
        matches = [r for r in rows if r.fileName == upload_name]
        row = matches[0] if matches else None
        entry = {"attempt": attempt, "found": bool(row), "status": row.status if row else None, "publicScore": row.publicScore if row else None, "date": row.date if row else None, "csv": str(out_csv.relative_to(ROOT))}
        history.append(entry)
        if row and row.status in {"SubmissionStatus.COMPLETE", "SubmissionStatus.ERROR"}:
            return {"final": entry, "history": history}
        time.sleep(sleep_s)
    return {"final": history[-1] if history else {}, "history": history, "timeout": True}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = []
    lines.append("# Final 5 Non-Duplicate Kaggle Submission Run")
    lines.append("")
    lines.append("User-approved execution of the remaining five KMURecSys26 Steam submissions. This run does not create new model predictions; it only materializes and submits the approved precomputed label files.")
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    for k, v in payload["safety"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| slot | upload | SHA | status | public | Δ vs 0.77825 | min diff vs prior live |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for r in payload["results"]:
        final = r.get("poll", {}).get("final", {})
        public = final.get("publicScore")
        try:
            delta = float(public) - PUBLIC_BEST if public not in {None, ""} else None
        except Exception:
            delta = None
        lines.append(
            f"| {r['slot']} | `{Path(r['upload']).name}` | `{r['upload_sha256'][:8]}` | `{final.get('status')}` | "
            f"{public if public not in {None, ''} else ''} | {delta:+.5f} | {r['duplicate_checks'].get('min_rowdiff_vs_live')} |"
            if delta is not None
            else f"| {r['slot']} | `{Path(r['upload']).name}` | `{r['upload_sha256'][:8]}` | `{final.get('status')}` |  |  | {r['duplicate_checks'].get('min_rowdiff_vs_live')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Exact filename and exact local upload SHA duplicates were blocked before each submit.")
    lines.append("- The row-diff guard was checked against locally available live-submission predictions before each submit.")
    lines.append("- Final pair remains emb128 4-seed + rankblend unless a result is separately promoted after review.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Actually call kaggle competitions submit")
    ap.add_argument("--sleep-seconds", type=int, default=20)
    args = ap.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to run without --execute")

    ts = now_kst_stamp()
    report_root = ROOT / "reports"
    upload_dir = ROOT / "submissions" / "final5_nonduplicate"
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    payload: dict[str, Any] = {
        "timestamp_kst": ts,
        "competition": COMPETITION,
        "safety": {
            "user_approved_remaining_5_execution": True,
            "new_model_predictions_created": False,
            "external_metadata_used": False,
            "hidden_label_access": False,
            "submit_now": True,
        },
        "results": [],
    }

    # Initial live preflight.
    initial_live_path = report_root / f"{ts}_final5_initial_live_submissions.csv"
    live = fetch_live_submissions(initial_live_path)
    today_count = count_today_utc(live)
    payload["initial_live"] = {"csv": str(initial_live_path.relative_to(ROOT)), "today_count_complete_or_pending": today_count, "live_rows": len(live)}
    if today_count >= DAILY_QUOTA:
        raise SubmissionError(f"Daily quota already exhausted: {today_count}/{DAILY_QUOTA}")

    for i, item in enumerate(SELECTED, start=1):
        slot = item["slot"]
        source = ROOT / item["source"]
        upload = upload_dir / item["upload_name"]
        live_check_path = report_root / f"{ts}_final5_{slot}_pre_live_submissions.csv"
        live = fetch_live_submissions(live_check_path)
        today_count = count_today_utc(live)
        if today_count >= DAILY_QUOTA:
            payload["stopped"] = {"reason": "daily_quota_exhausted", "today_count": today_count, "before_slot": slot}
            break
        live_names = {r.fileName for r in live}
        live_preds = local_live_predictions(live_names)
        materialized = materialize_upload(source, upload, pairs)
        dup = duplicate_checks(upload, live, live_preds)
        if materialized["bad_users_tophalf"] != 0:
            raise SubmissionError(f"{slot} has bad_users_tophalf={materialized['bad_users_tophalf']}")
        if dup["filename_already_submitted"] or dup["exact_upload_sha_already_submitted_locally"]:
            raise SubmissionError(f"{slot} duplicate blocked: {dup}")
        if dup["min_rowdiff_vs_live"] is not None and dup["min_rowdiff_vs_live"] < 500:
            raise SubmissionError(f"{slot} rowdiff guard failed: {dup}")
        msg = f"FINAL5 non-duplicate burn {slot}; {item['role']}; SHA {materialized['upload_sha256'][:8]}; no external metadata/hidden labels."
        submit_log = report_root / f"{ts}_final5_{slot}_submit_stdout.txt"
        cp = submit_one(upload, msg, submit_log)
        rec: dict[str, Any] = {
            **item,
            **materialized,
            "duplicate_checks": dup,
            "quota_count_before_submit_utc_day": today_count,
            "submit_message": msg,
            "submit_stdout": str(submit_log.relative_to(ROOT)),
            "submit_returncode": cp.returncode,
        }
        if cp.returncode != 0:
            rec["submit_failed"] = True
            payload["results"].append(rec)
            payload["stopped"] = {"reason": "submit_returncode_nonzero", "slot": slot}
            break
        poll_prefix = report_root / f"{ts}_final5_{slot}"
        rec["poll"] = poll_result(upload.name, poll_prefix, sleep_s=args.sleep_seconds)
        payload["results"].append(rec)
        final = rec["poll"].get("final", {})
        if final.get("status") == "SubmissionStatus.ERROR":
            payload["stopped"] = {"reason": "submission_status_error", "slot": slot}
            break
    # Final live snapshot.
    final_live_path = report_root / f"{ts}_final5_final_live_submissions.csv"
    try:
        final_live = fetch_live_submissions(final_live_path)
        payload["final_live"] = {"csv": str(final_live_path.relative_to(ROOT)), "today_count_complete_or_pending": count_today_utc(final_live), "live_rows": len(final_live)}
    except Exception as exc:  # keep report even if final fetch fails
        payload["final_live_error"] = repr(exc)

    out_json = report_root / f"{ts}_final5_submission_run.json"
    out_md = report_root / f"{ts}_final5_submission_run.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"report_json": str(out_json.relative_to(ROOT)), "report_md": str(out_md.relative_to(ROOT)), "submitted": len(payload["results"]), "stopped": payload.get("stopped")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
