#!/usr/bin/env python3
"""Generate LightGCN score coverage for boundary v1 panel20 splits (NO-SUBMIT).

This is a scheduler/worker script.  It writes validation score artifacts only under
`artifacts/boundary_v1_panel20_score_coverage/` and never creates a Kaggle candidate
CSV or runs a Kaggle submit command.

Examples
--------
Smoke one cheap cell:
  uv run --with numpy --with pandas --with scipy --with torch==2.10.0 \
    python3 scripts/boundary_v1_panel20_lightgcn_score_coverage.py \
    --emb-dims 128 --seeds 42 --splits val_random_uniform_seed101 \
    --epochs 1 --out-root artifacts/boundary_v1_panel20_score_coverage_smoke

Full panel20 launch:
  uv run --with numpy --with pandas --with scipy --with torch==2.10.0 \
    python3 scripts/boundary_v1_panel20_lightgcn_score_coverage.py \
    --emb-dims 128,192 --seeds 42,123,2024,7 --epochs 200 --max-parallel 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import build_user_item_matrix, evaluate_tophalf, load_pairs_csv, load_train_interactions, write_json  # noqa: E402
from lightgcn_train import score_candidates, train_lightgcn  # noqa: E402

PANEL20_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
DEFAULT_OUT_ROOT = ROOT / "artifacts/boundary_v1_panel20_score_coverage"
DEFAULT_STATUS_DIR = ROOT / "reports/boundary_v1_panel20_score_coverage_status"


@dataclass(frozen=True)
class Job:
    split: str
    emb_dim: int
    seed: int
    device: str

    @property
    def job_id(self) -> str:
        return f"emb{self.emb_dim}/{self.split}/seed{self.seed}"


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def available_panel_splits(panel_root: Path) -> list[str]:
    return sorted(p.name for p in panel_root.iterdir() if p.is_dir() and p.name.startswith("val_random_uniform_seed"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def score_col(df: pd.DataFrame) -> str:
    for col in ("score_lightgcn", "score", "score_layermix_uniform"):
        if col in df.columns:
            return col
    raise ValueError(f"No score column in {df.columns.tolist()}")


def validate_split_dir(split_dir: Path) -> None:
    for name in ("train_interactions.csv", "candidates.csv"):
        if not (split_dir / name).exists():
            raise FileNotFoundError(split_dir / name)


def run_worker(args: argparse.Namespace) -> int:
    started = time.time()
    panel_root = Path(args.panel_root)
    out_root = Path(args.out_root)
    split_dir = panel_root / args.worker_split
    validate_split_dir(split_dir)
    out_dir = out_root / f"emb{args.worker_emb_dim}" / args.worker_split / f"seed{args.worker_seed}"
    out_file = out_dir / "lightgcn_scores.csv"
    summary_file = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and summary_file.exists() and not args.force:
        print(json.dumps({"event": "skip_existing", "out_file": str(out_file), "summary_file": str(summary_file)}, ensure_ascii=False), flush=True)
        return 0

    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    cand = load_pairs_csv(split_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(train_df, binary=True)
    print(
        json.dumps(
            {
                "event": "worker_start",
                "split": args.worker_split,
                "emb_dim": args.worker_emb_dim,
                "seed": args.worker_seed,
                "device": args.worker_device,
                "train_rows": len(train_df),
                "candidate_rows": len(cand),
                "users": len(users),
                "items": len(items),
                "nnz": int(mat.nnz),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    user_emb, item_emb, meta = train_lightgcn(
        mat,
        len(users),
        len(items),
        emb_dim=args.worker_emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.worker_device,
        seed=args.worker_seed,
    )
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, user_emb, item_emb, u2i, i2i)
    summary, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out_file, index=False)
    payload = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "split": args.worker_split,
        "emb_dim": args.worker_emb_dim,
        "seed": args.worker_seed,
        "device": args.worker_device,
        "row_accuracy": float(summary["row_accuracy"]),
        "candidate_rows": int(len(cand)),
        "train_rows": int(len(train_df)),
        "meta": meta,
        "elapsed_seconds_total": round(time.time() - started, 2),
        "out_file": str(out_file),
    }
    write_json(summary_file, payload)
    print(json.dumps({"event": "worker_done", **payload}, ensure_ascii=False), flush=True)
    return 0


def summarize_completed(out_root: Path, status_dir: Path, jobs: list[Job]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        summary_file = out_root / f"emb{job.emb_dim}" / job.split / f"seed{job.seed}" / "summary.json"
        score_file = out_root / f"emb{job.emb_dim}" / job.split / f"seed{job.seed}" / "lightgcn_scores.csv"
        row = {**asdict(job), "summary_file": str(summary_file), "score_file": str(score_file), "complete": summary_file.exists() and score_file.exists()}
        if summary_file.exists():
            try:
                payload = json.loads(summary_file.read_text(encoding="utf-8"))
                row["row_accuracy"] = payload.get("row_accuracy")
                row["train_seconds"] = (payload.get("meta") or {}).get("train_seconds")
            except Exception as exc:  # pragma: no cover
                row["summary_error"] = str(exc)
        rows.append(row)
    df = pd.DataFrame(rows)
    status_dir.mkdir(parents=True, exist_ok=True)
    csv_path = status_dir / "job_status.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    aggregate = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "jobs_total": len(jobs),
        "jobs_complete": int(df["complete"].sum()) if len(df) else 0,
        "job_status_csv": str(csv_path),
        "out_root": str(out_root),
    }
    write_json(status_dir / "summary.json", aggregate)
    return aggregate


def run_scheduler(args: argparse.Namespace) -> int:
    panel_root = Path(args.panel_root)
    out_root = Path(args.out_root)
    status_dir = Path(args.status_dir)
    if args.splits == "all":
        splits = available_panel_splits(panel_root)
    else:
        splits = parse_strs(args.splits)
    emb_dims = parse_ints(args.emb_dims)
    seeds = parse_ints(args.seeds)
    devices = parse_strs(args.devices)
    if not devices:
        devices = ["cuda:0"]
    jobs: list[Job] = []
    for split in splits:
        validate_split_dir(panel_root / split)
        for emb_dim in emb_dims:
            for seed in seeds:
                device = devices[len(jobs) % len(devices)]
                jobs.append(Job(split=split, emb_dim=emb_dim, seed=seed, device=device))
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    status_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = status_dir / "status.jsonl"
    append_jsonl(status_jsonl, {"event": "scheduler_start", "jobs_total": len(jobs), "splits": splits, "emb_dims": emb_dims, "seeds": seeds, "devices": devices, "out_root": str(out_root), "time": time.time()})

    running: list[tuple[Job, subprocess.Popen[str], Path]] = []
    pending = jobs.copy()
    completed = 0
    failed = 0
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            score_file = out_root / f"emb{job.emb_dim}" / job.split / f"seed{job.seed}" / "lightgcn_scores.csv"
            summary_file = out_root / f"emb{job.emb_dim}" / job.split / f"seed{job.seed}" / "summary.json"
            if score_file.exists() and summary_file.exists() and not args.force:
                completed += 1
                append_jsonl(status_jsonl, {"event": "skip_existing", "job": job.job_id, "time": time.time()})
                continue
            log_path = status_dir / "logs" / f"emb{job.emb_dim}_{job.split}_seed{job.seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-split", job.split,
                "--worker-emb-dim", str(job.emb_dim),
                "--worker-seed", str(job.seed),
                "--worker-device", job.device,
                "--panel-root", str(panel_root),
                "--out-root", str(out_root),
                "--epochs", str(args.epochs),
                "--n-layers", str(args.n_layers),
                "--reg", str(args.reg),
                "--lr", str(args.lr),
                "--batch-size", str(args.batch_size),
            ]
            if args.force:
                cmd.append("--force")
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT), env=env)
            running.append((job, proc, log_path))
            append_jsonl(status_jsonl, {"event": "job_start", "job": job.job_id, "pid": proc.pid, "device": job.device, "log": str(log_path), "time": time.time()})
        time.sleep(args.poll_seconds)
        still: list[tuple[Job, subprocess.Popen[str], Path]] = []
        for job, proc, log_path in running:
            ret = proc.poll()
            if ret is None:
                still.append((job, proc, log_path))
                continue
            if ret == 0:
                completed += 1
                event = "job_done"
            else:
                failed += 1
                event = "job_failed"
            append_jsonl(status_jsonl, {"event": event, "job": job.job_id, "returncode": ret, "log": str(log_path), "time": time.time()})
            summarize_completed(out_root, status_dir, jobs)
        running = still
    aggregate = summarize_completed(out_root, status_dir, jobs)
    append_jsonl(status_jsonl, {"event": "scheduler_done", "completed": completed, "failed": failed, "aggregate": aggregate, "time": time.time()})
    print(json.dumps({"completed": completed, "failed": failed, **aggregate}, indent=2, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 2


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="Run one training cell; scheduler uses this internally.")
    ap.add_argument("--worker-split")
    ap.add_argument("--worker-emb-dim", type=int)
    ap.add_argument("--worker-seed", type=int)
    ap.add_argument("--worker-device", default="cuda:0")
    ap.add_argument("--panel-root", default=str(PANEL20_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--status-dir", default=str(DEFAULT_STATUS_DIR))
    ap.add_argument("--splits", default="all", help="all or comma-separated split names")
    ap.add_argument("--emb-dims", default="128")
    ap.add_argument("--seeds", default="42,123,2024,7")
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--max-jobs", type=int)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--force", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.worker:
        if not args.worker_split or args.worker_emb_dim is None or args.worker_seed is None:
            raise SystemExit("worker mode requires --worker-split, --worker-emb-dim, --worker-seed")
        raise SystemExit(run_worker(args))
    raise SystemExit(run_scheduler(args))


if __name__ == "__main__":
    main()
