#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Independent uniform-split confirmation for the OTTO co-visitation residual axis.

Safety contract:
- validation-only; build synthetic validation splits from public train.json/pairs.csv only
- no full-test pairs, no uploadable candidate/submission CSV, no Kaggle submit
- no hidden/private labels, no external Steam scraping, no git stage/commit/push

The main mode builds fresh random/uniform validation splits, trains the canonical
emb128 L4 reg1e-3 LightGCN 4-seed baseline for those splits, and evaluates a
pre-registered OTTO residual family selected from the earlier 3-split panel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_validation_splits import SplitConfig, build_split  # noqa: E402
from otto_source_covisit_smoke import FEATURE_COLS, build_indices, build_sources, score_candidates  # noqa: E402
from recsys_played_utils import (  # noqa: E402
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    load_train_json,
    normalize_within_user,
    write_json,
)

MODEL_SEEDS = [42, 123, 2024, 7]
DEFAULT_SPLIT_SEEDS = [314, 2025, 2718]

# The only strict-confirmation row: fixed before seeing the fresh independent panel.
STRICT_VARIANT = (
    "pre_registered_old_panel_top_coplay_top5_reverse_recent",
    [("score_coplay_top5_mean", 0.090), ("score_reverse_recent", 0.040)],
)

# Extra diagnostics are reported but do not trigger candidate escalation by themselves.
EXTRA_DIAGNOSTIC_VARIANTS = [
    ("primary_smoke_coplay_top5_w0.200", [("score_coplay_top5_mean", 0.200)]),
    ("followup_top_coplay_top5_w0.120_last5_forward_w0.030", [("score_coplay_top5_mean", 0.120), ("score_last5_forward", 0.030)]),
    ("looso_seed42_choice_coplay_top5_w0.100_reverse_recent_w0.070", [("score_coplay_top5_mean", 0.100), ("score_reverse_recent", 0.070)]),
    ("looso_seed7_choice_coplay_top5_w0.095_reverse_recent_w0.035", [("score_coplay_top5_mean", 0.095), ("score_reverse_recent", 0.035)]),
]

SAFETY_FLAGS = {
    "validation_only": True,
    "candidate_csv_written": False,
    "full_test_candidate_or_submission_csv_created": False,
    "kaggle_submit_executed": False,
    "hidden_labels_used": False,
    "private_answers_used": False,
    "external_steam_scraping_used": False,
    "credentials_or_tokens_printed": False,
    "git_stage_commit_push_executed": False,
    "recursive_cron_scheduled": False,
}


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError(f"empty int list: {raw!r}")
    return vals


def exact_binom_two_sided(fixes: int, breaks: int) -> float | None:
    n = fixes + breaks
    if n <= 0:
        return None
    k = min(fixes, breaks)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * cdf))


def train_one_cell(args: argparse.Namespace) -> None:
    """Worker mode: train exactly one (split, model seed) LightGCN cell."""
    # Import torch-dependent module only in worker mode so dry-runs and report parsing
    # do not require a torch environment.
    from lightgcn_train import score_candidates as score_lgcn_candidates  # noqa: WPS433
    from lightgcn_train import train_lightgcn  # noqa: WPS433
    from recsys_played_utils import build_user_item_matrix  # noqa: WPS433

    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    print(
        f"[worker] split={args.split_name} model_seed={args.model_seed} "
        f"device={args.device} users={len(users)} items={len(items)} nnz={mat.nnz}",
        flush=True,
    )
    user_emb, item_emb, meta = train_lightgcn(
        mat,
        len(users),
        len(items),
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.model_seed,
    )
    scored = candidates.copy()
    scored["score_lightgcn"] = score_lgcn_candidates(scored, user_emb, item_emb, user_to_idx, item_to_idx)
    summary, _ = evaluate_tophalf(scored, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    scored[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out_dir / "lightgcn_scores.csv", index=False)
    write_json(
        out_dir / "summary.json",
        {
            "split": args.split_name,
            "model_seed": args.model_seed,
            "emb_dim": args.emb_dim,
            "n_layers": args.n_layers,
            "reg": args.reg,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "row_accuracy": float(cast(Any, summary["row_accuracy"])),
            "summary": summary,
            "train_meta": meta,
            "safety_flags": SAFETY_FLAGS,
        },
    )
    print(
        f"[worker] done split={args.split_name} model_seed={args.model_seed} "
        f"row_acc={float(cast(Any, summary['row_accuracy'])):.8f} train_seconds={meta.get('train_seconds')}",
        flush=True,
    )


def build_validation_panel(split_seeds: list[int], validation_root: Path) -> list[dict[str, Any]]:
    data_dir = ROOT / "data" / "raw" / "public" / "data"
    train_df = load_train_json(data_dir / "train.json")
    pairs_df = load_pairs_csv(data_dir / "pairs.csv")
    summaries = []
    ensure_dir(validation_root)
    for seed in split_seeds:
        cfg = SplitConfig(holdout="random", negative="uniform", seed=seed)
        split_dir = validation_root / cfg.name
        if (split_dir / "train_interactions.csv").exists() and (split_dir / "candidates.csv").exists():
            summary_path = split_dir / "summary.json"
            if summary_path.exists():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            else:
                summaries.append({"name": cfg.name, "out_dir": str(split_dir), "status": "existing_without_summary"})
            print(f"[split] reuse existing {cfg.name}", flush=True)
            continue
        print(f"[split] build {cfg.name} -> {split_dir}", flush=True)
        summaries.append(build_split(train_df, pairs_df, cfg, validation_root))
    write_json(validation_root / "validation_splits_summary.json", summaries)
    return summaries


def expected_score_path(base_root: Path, split: str, model_seed: int) -> Path:
    return base_root / split / f"seed{model_seed}" / "lightgcn_scores.csv"


def launch_train_panel(
    split_names: list[str],
    validation_root: Path,
    base_root: Path,
    log_root: Path,
    run_ts: str,
    gpus: list[int],
    epochs: int,
    emb_dim: int,
    n_layers: int,
    reg: float,
    lr: float,
    batch_size: int,
    max_parallel: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for split in split_names:
        for seed in MODEL_SEEDS:
            out_dir = base_root / split / f"seed{seed}"
            score_path = out_dir / "lightgcn_scores.csv"
            if score_path.exists():
                tasks.append({"split": split, "model_seed": seed, "status": "skipped_existing", "score_path": str(score_path)})
                print(f"[train] skip existing {split} seed{seed}", flush=True)
                continue
            tasks.append({"split": split, "model_seed": seed, "status": "pending", "score_path": str(score_path)})

    pending = [t for t in tasks if t["status"] == "pending"]
    if not pending:
        return tasks

    ensure_dir(log_root)
    active: list[dict[str, Any]] = []
    completed = 0
    failed: list[dict[str, Any]] = []
    max_parallel = max(1, min(max_parallel, len(gpus)))

    def start_task(task: dict[str, Any], gpu: int) -> dict[str, Any]:
        split = str(task["split"])
        seed = int(task["model_seed"])
        out_dir = base_root / split / f"seed{seed}"
        ensure_dir(out_dir)
        log_path = log_root / f"{run_ts}_indep_train_{split}_seed{seed}_gpu{gpu}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-ts",
            run_ts,
            "--worker-train",
            "--split-name",
            split,
            "--split-dir",
            str(validation_root / split),
            "--model-seed",
            str(seed),
            "--device",
            f"cuda:{gpu}",
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--emb-dim",
            str(emb_dim),
            "--n-layers",
            str(n_layers),
            "--reg",
            str(reg),
            "--lr",
            str(lr),
            "--batch-size",
            str(batch_size),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)
        task.update({"status": "running", "gpu": gpu, "pid": proc.pid, "log": str(log_path), "started_at": time.time()})
        print(f"[train] start split={split} seed{seed} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
        return {"task": task, "proc": proc, "log_file": log_f, "gpu": gpu}

    free_gpus = gpus[:]
    pending_iter = iter(pending)
    while True:
        while len(active) < max_parallel and free_gpus:
            try:
                task = next(pending_iter)
            except StopIteration:
                break
            active.append(start_task(task, free_gpus.pop(0)))
        if not active:
            break
        time.sleep(20)
        still = []
        for item in active:
            proc = item["proc"]
            ret = proc.poll()
            if ret is None:
                still.append(item)
                continue
            item["log_file"].close()
            task = item["task"]
            free_gpus.append(item["gpu"])
            elapsed = time.time() - float(task.get("started_at", time.time()))
            if ret == 0 and Path(str(task["score_path"])).exists():
                task.update({"status": "completed", "exit_code": ret, "elapsed_seconds": round(elapsed, 1)})
                completed += 1
                print(f"[train] completed split={task['split']} seed{task['model_seed']} elapsed={elapsed:.1f}s", flush=True)
            else:
                task.update({"status": "failed", "exit_code": ret, "elapsed_seconds": round(elapsed, 1)})
                failed.append(task)
                print(f"[train] FAILED split={task['split']} seed{task['model_seed']} ret={ret} log={task.get('log')}", flush=True)
        active = still
        if failed:
            for item in active:
                item["proc"].terminate()
                item["log_file"].close()
            raise RuntimeError(f"training failed: {failed[:3]}")
        print(f"[train] progress completed={completed}/{len(pending)} active={len(active)} free_gpus={free_gpus}", flush=True)
    return tasks


def load_base_scores(base_root: Path, split: str) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    cols: list[str] = []
    for seed in MODEL_SEEDS:
        path = expected_score_path(base_root, split, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
        col = f"base_seed{seed}"
        df = df.rename(columns={"score_lightgcn": col})
        base = df if base is None else base.merge(df[["ID", col]], on="ID", validate="one_to_one")
        cols.append(col)
    assert base is not None
    base["score_base"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "score_base"]]


def predict_from_scores(df: pd.DataFrame, score_col: str) -> np.ndarray:
    return evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")[1]["Pred"].to_numpy(np.int8)


def evaluate_variant(scored_by_split: dict[str, pd.DataFrame], variant: str, terms: list[tuple[str, float]]) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    accs: dict[str, float] = {}
    base_accs: dict[str, float] = {}
    fixes = 0
    breaks = 0
    for split, df0 in scored_by_split.items():
        df = df0.copy()
        score = df["z_score_base"].to_numpy(np.float64).copy()
        for feature, weight in terms:
            col = feature if feature == "score_source_mean_z" else f"z_{feature}"
            if col not in df.columns:
                raise ValueError(f"missing column {col} for {variant} on {split}")
            score += weight * df[col].to_numpy(np.float64)
        score_col = f"score_{variant}"
        df[score_col] = score
        base_summary, base_pred_df = evaluate_tophalf(df, "score_base", label_col="Label", user_col="userID", id_col="ID")
        new_summary, new_pred_df = evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")
        y = df["Label"].to_numpy(np.int8)
        base_pred = base_pred_df["Pred"].to_numpy(np.int8)
        new_pred = new_pred_df["Pred"].to_numpy(np.int8)
        base_acc = float(cast(Any, base_summary["row_accuracy"]))
        acc = float(cast(Any, new_summary["row_accuracy"]))
        base_accs[split] = base_acc
        accs[split] = acc
        deltas[split] = acc - base_acc
        fixes += int(((new_pred == y) & (base_pred != y)).sum())
        breaks += int(((new_pred != y) & (base_pred == y)).sum())
    vals = list(deltas.values())
    return {
        "variant": variant,
        "terms": terms,
        "mean_delta_vs_base": float(np.mean(vals)),
        "min_delta_vs_base": float(np.min(vals)),
        "max_delta_vs_base": float(np.max(vals)),
        "positive_splits": int(sum(v > 0 for v in vals)),
        "num_splits": len(vals),
        "fixes": int(fixes),
        "breaks": int(breaks),
        "pooled_p_exact": exact_binom_two_sided(fixes, breaks),
        "split_deltas": deltas,
        "split_accs": accs,
        "base_accs": base_accs,
    }


def score_otto_sources(split_names: list[str], validation_root: Path, base_root: Path, artifact_root: Path) -> dict[str, pd.DataFrame]:
    scored_by_split: dict[str, pd.DataFrame] = {}
    for split in split_names:
        split_dir = validation_root / split
        train = load_train_interactions(split_dir / "train_interactions.csv")
        candidates = load_pairs_csv(split_dir / "candidates.csv")
        base = load_base_scores(base_root, split)
        candidates = candidates.merge(base[["ID", "score_base"]], on="ID", validate="one_to_one")
        _, item_to_idx, _ = build_indices(train)
        sources = build_sources(train, item_to_idx)
        scored = score_candidates(train, candidates, item_to_idx, sources)
        scored = normalize_within_user(scored, ["score_base"], user_col="userID")
        keep_cols = ["ID", "userID", "gameID", "Label", "score_base"] + FEATURE_COLS + [
            f"z_{c}" for c in FEATURE_COLS if c != "score_source_mean_z"
        ] + ["z_score_base"]
        out_dir = ensure_dir(artifact_root / split)
        scored[keep_cols].to_csv(out_dir / "validation_otto_source_scores.csv", index=False)
        scored_by_split[split] = scored
        base_summary = evaluate_tophalf(scored, "score_base", label_col="Label", user_col="userID", id_col="ID")[0]
        base_acc = float(cast(Any, base_summary["row_accuracy"]))
        print(f"[otto] scored {split} base_acc={base_acc:.8f} -> {out_dir / 'validation_otto_source_scores.csv'}", flush=True)
    return scored_by_split


def strict_gate(row: dict[str, Any], num_splits: int) -> bool:
    return (
        row["mean_delta_vs_base"] >= 0.0015
        and row["min_delta_vs_base"] >= 0
        and row["positive_splits"] == num_splits
        and row["fixes"] > row["breaks"]
        and row["pooled_p_exact"] is not None
        and row["pooled_p_exact"] < 0.05
    )


def write_markdown(payload: dict[str, Any], out_md: Path) -> None:
    strict_row = payload["strict_confirmation_row"]
    lines = [
        "# OTTO independent uniform confirmation",
        "",
        f"- Timestamp: {payload['timestamp_kst']}",
        "- Safety: validation-only; no full-test pairs; no candidate/submission CSV; no Kaggle submit; no hidden/private labels; no external scraping.",
        f"- Fresh uniform split seeds: `{payload['split_seeds']}`",
        f"- Model seeds per split: `{payload['model_seeds']}`",
        f"- Verdict: `{payload['verdict']}`",
        "",
        "## Strict pre-registered row",
        "",
        f"- Variant: `{strict_row['variant']}`",
        f"- terms: `{strict_row['terms']}`",
        f"- mean Δ vs base: {strict_row['mean_delta_vs_base']:+.10f}",
        f"- min split Δ: {strict_row['min_delta_vs_base']:+.10f}",
        f"- positive splits: {strict_row['positive_splits']}/{strict_row['num_splits']}",
        f"- fixes/breaks: {strict_row['fixes']}/{strict_row['breaks']}",
        f"- pooled exact p: {strict_row['pooled_p_exact']}",
        f"- split deltas: `{strict_row['split_deltas']}`",
        "",
        "## Diagnostic rows",
        "",
    ]
    for i, row in enumerate(payload["diagnostic_rows"], 1):
        lines.append(
            f"{i}. `{row['variant']}` meanΔ={row['mean_delta_vs_base']:+.10f}, "
            f"minΔ={row['min_delta_vs_base']:+.10f}, pos={row['positive_splits']}/{row['num_splits']}, "
            f"fixes/breaks={row['fixes']}/{row['breaks']}, p={row['pooled_p_exact']}, deltas={row['split_deltas']}"
        )
    lines += ["", "## Artifacts", ""]
    for k, v in payload["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--split-seeds", default=",".join(str(x) for x in DEFAULT_SPLIT_SEEDS))
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--root-artifact-dir", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--dry-run", action="store_true")

    # Worker mode args.
    ap.add_argument("--worker-train", action="store_true")
    ap.add_argument("--split-name")
    ap.add_argument("--split-dir")
    ap.add_argument("--model-seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    if args.worker_train:
        train_one_cell(args)
        return

    split_seeds = parse_int_list(args.split_seeds)
    gpus = parse_int_list(args.gpus)
    run_root = ROOT / (args.root_artifact_dir or f"artifacts/otto_independent_uniform_{args.run_ts}")
    validation_root = run_root / "validation"
    base_root = run_root / "lightgcn_emb128"
    otto_root = run_root / "otto_source_covisit"
    log_root = ROOT / "logs" / f"otto_independent_uniform_{args.run_ts}"
    out_json = ROOT / (args.out_json or f"reports/{args.run_ts}_otto_independent_uniform_confirmation.json")
    out_md = ROOT / (args.out_md or f"reports/{args.run_ts}_otto_independent_uniform_confirmation.md")

    split_names = [f"val_random_uniform_seed{s}" for s in split_seeds]
    if args.dry_run:
        payload = {
            "timestamp_kst": args.run_ts,
            "split_seeds": split_seeds,
            "split_names": split_names,
            "model_seeds": MODEL_SEEDS,
            "strict_variant": STRICT_VARIANT,
            "extra_diagnostic_variants": EXTRA_DIAGNOSTIC_VARIANTS,
            "artifacts": {
                "run_root": str(run_root.relative_to(ROOT)),
                "validation_root": str(validation_root.relative_to(ROOT)),
                "base_root": str(base_root.relative_to(ROOT)),
                "otto_root": str(otto_root.relative_to(ROOT)),
                "out_json": str(out_json.relative_to(ROOT)),
                "out_md": str(out_md.relative_to(ROOT)),
            },
            "safety_flags": SAFETY_FLAGS,
            "verdict": "DRY_RUN_OK",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    ensure_dir(run_root)
    ensure_dir(base_root)
    ensure_dir(otto_root)
    split_summaries = build_validation_panel(split_seeds, validation_root)
    train_tasks = launch_train_panel(
        split_names=split_names,
        validation_root=validation_root,
        base_root=base_root,
        log_root=log_root,
        run_ts=args.run_ts,
        gpus=gpus,
        epochs=args.epochs,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        reg=args.reg,
        lr=args.lr,
        batch_size=args.batch_size,
        max_parallel=args.max_parallel,
    )
    scored_by_split = score_otto_sources(split_names, validation_root, base_root, otto_root)
    all_rows = [evaluate_variant(scored_by_split, name, terms) for name, terms in [STRICT_VARIANT] + EXTRA_DIAGNOSTIC_VARIANTS]
    strict_row = next(row for row in all_rows if row["variant"] == STRICT_VARIANT[0])
    rows = [row for row in all_rows if row["variant"] != STRICT_VARIANT[0]]
    rows.sort(key=lambda r: (r["mean_delta_vs_base"], r["min_delta_vs_base"], r["fixes"] - r["breaks"]), reverse=True)
    strict_pass = strict_gate(strict_row, len(split_names))
    diagnostic_strict = [row for row in rows if strict_gate(row, len(split_names))]
    if strict_pass:
        verdict = "INDEPENDENT_STRICT_CONFIRMATION_PASS"
    elif strict_row["positive_splits"] == len(split_names) and strict_row["mean_delta_vs_base"] > 0:
        verdict = "INDEPENDENT_WEAK_POSITIVE_STRICT_FAIL"
    elif rows and rows[0]["mean_delta_vs_base"] > 0:
        verdict = "INDEPENDENT_DIAGNOSTIC_ONLY_POSITIVE_STRICT_FAIL"
    else:
        verdict = "INDEPENDENT_REJECT"

    payload: dict[str, Any] = {
        "timestamp_kst": args.run_ts,
        "safety_flags": SAFETY_FLAGS,
        "split_seeds": split_seeds,
        "split_names": split_names,
        "model_seeds": MODEL_SEEDS,
        "lightgcn_params": {
            "emb_dim": args.emb_dim,
            "n_layers": args.n_layers,
            "reg": args.reg,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "strict_gate": {
            "mean_delta_minimum": 0.0015,
            "min_delta_minimum": 0.0,
            "positive_splits_required": len(split_names),
            "fixes_gt_breaks": True,
            "pooled_p_exact_lt": 0.05,
            "only_variant_allowed_for_candidate_escalation": STRICT_VARIANT[0],
            "note": "The strict row was fixed from the previous panel before this fresh split panel was built/trained/evaluated.",
        },
        "split_summaries": split_summaries,
        "train_tasks": train_tasks,
        "strict_confirmation_row": strict_row,
        "diagnostic_rows": rows,
        "diagnostic_strict_pass_count": len(diagnostic_strict),
        "diagnostic_strict_rows": diagnostic_strict,
        "verdict": verdict,
        "candidate_escalation_allowed": bool(strict_pass),
        "artifacts": {
            "run_root": str(run_root.relative_to(ROOT)),
            "validation_root": str(validation_root.relative_to(ROOT)),
            "base_root": str(base_root.relative_to(ROOT)),
            "otto_root": str(otto_root.relative_to(ROOT)),
            "log_root": str(log_root.relative_to(ROOT)),
            "out_json": str(out_json.relative_to(ROOT)),
            "out_md": str(out_md.relative_to(ROOT)),
        },
    }
    write_json(out_json, payload)
    write_markdown(payload, out_md)
    print(json.dumps({"verdict": verdict, "candidate_escalation_allowed": strict_pass, "strict_row": strict_row, "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
