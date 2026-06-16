#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Independent confirmation for the current-best ALS residual atlas row.

Safety contract:
- validation-only
- reuse public-train fresh uniform splits from the previous OTTO independent run
- train/evaluate emb192 validation scores only as needed
- write ALS candidate_scores artifacts for validation splits only
- no full-test candidate/submission CSV and no Kaggle submit
- no hidden/private labels, no external Steam scraping, no git stage/commit/push
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
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import materialize_readme_rankblend_residual as rb  # noqa: E402
import otto_independent_uniform_confirmation as oi  # noqa: E402

MODEL_SEEDS = [42, 123, 2024, 7]
SPLIT_SEEDS = [314, 2025, 2718]
SPLIT_NAMES = [f"val_random_uniform_seed{s}" for s in SPLIT_SEEDS]
SOURCE_INDEP_ROOT = ROOT / "artifacts" / "otto_independent_uniform_20260607T095549KST"
SOURCE_VALIDATION_ROOT = SOURCE_INDEP_ROOT / "validation"
SOURCE_EMB128_ROOT = SOURCE_INDEP_ROOT / "lightgcn_emb128"
STRICT_VARIANT = {
    "variant": "pre_registered_atlas_top_als_f32_popa4_w0.30_band2",
    "feature": "score_als_f32_it30_alpha20_popa4",
    "weight": 0.30,
    "band": 2,
}
DIAGNOSTIC_VARIANTS = [
    STRICT_VARIANT,
    {"variant": "atlas_als_f32_popa4_w0.30_band1", "feature": "score_als_f32_it30_alpha20_popa4", "weight": 0.30, "band": 1},
    {"variant": "atlas_als_f32_popa4_w0.30_all", "feature": "score_als_f32_it30_alpha20_popa4", "weight": 0.30, "band": None},
    {"variant": "atlas_als_f32_popa4_w0.30_band3", "feature": "score_als_f32_it30_alpha20_popa4", "weight": 0.30, "band": 3},
    {"variant": "atlas_als_f32_popa4_w0.20_band1", "feature": "score_als_f32_it30_alpha20_popa4", "weight": 0.20, "band": 1},
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def emb_seed_paths(root: Path, split: str) -> list[Path]:
    return [root / split / f"seed{s}" / "lightgcn_scores.csv" for s in MODEL_SEEDS]


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def ensemble(paths: list[Path], name: str) -> pd.DataFrame:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {name}: {missing[:5]}")
    first = pd.read_csv(paths[0])
    sc = score_col(first)
    out = first[["ID", "userID", "gameID", "Label"]].copy()
    out[name] = first[sc].astype(float).to_numpy()
    for p in paths[1:]:
        d = pd.read_csv(p)
        sc2 = score_col(d)
        before = len(out)
        out = out.merge(d[["ID", sc2]].rename(columns={sc2: "_score"}), on="ID", validate="one_to_one")
        if len(out) != before:
            raise RuntimeError(f"row mismatch merging {p}: {before}->{len(out)}")
        out[name] += out.pop("_score").astype(float).to_numpy()
    out[name] /= len(paths)
    return out


def top_half_pred(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    values = np.asarray(score, dtype=float)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        k = len(idx) // 2
        order = np.lexsort((ids[idx], -values[idx]))
        pred[idx[order[:k]]] = 1
    return pred


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    values = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        vals = values[idx]
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        if sd > 1e-12:
            out[idx] = (vals - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def base_rank_margin(df: pd.DataFrame) -> np.ndarray:
    margin = np.zeros(len(df), dtype=float)
    score = df["score_rankblend"].to_numpy(dtype=float)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        k = len(idx) // 2
        order = np.lexsort((ids[idx], -score[idx]))
        rank = np.empty(len(idx), dtype=float)
        rank[order] = np.arange(1, len(idx) + 1, dtype=float)
        margin[idx] = np.abs(rank - (k + 0.5))
    return margin


def exact_p(fixes: int, breaks: int) -> float:
    n = fixes + breaks
    if n <= 0:
        return 1.0
    return float(binomtest(min(fixes, breaks), n, 0.5, alternative="two-sided").pvalue)


def eval_variant(df: pd.DataFrame, feature: str, weight: float, band: int | None) -> dict[str, Any]:
    y = df["Label"].to_numpy(dtype=np.int8)
    base_score = df["score_rankblend"].to_numpy(dtype=float)
    base_pred = top_half_pred(df, base_score)
    base_ok = base_pred == y
    z_base = within_user_z(df, "score_rankblend")
    z_feat = within_user_z(df, feature)
    if band is None:
        gated = z_feat
    else:
        gated = z_feat * (df["base_margin"].to_numpy(dtype=float) <= band)
    score = z_base + float(weight) * gated
    pred = top_half_pred(df, score)
    ok = pred == y
    fixes = int((ok & ~base_ok).sum())
    breaks = int((~ok & base_ok).sum())
    return {
        "base_accuracy": float(base_ok.mean()),
        "accuracy": float(ok.mean()),
        "delta_vs_rankblend": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": fixes + breaks,
        "changed": int((pred != base_pred).sum()),
        "p_exact": exact_p(fixes, breaks),
    }


def strict_gate(row: dict[str, Any]) -> bool:
    return (
        row["mean_delta_vs_rankblend"] >= 0.0015
        and row["min_delta_vs_rankblend"] >= 0
        and row["positive_splits"] == len(SPLIT_NAMES)
        and row["fixes"] > row["breaks"]
        and row["pooled_p_exact"] < 0.05
    )


def run_als_scores(run_ts: str, als_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        out_dir = ensure_dir(als_root / f"{split}_readme_bprals")
        out_csv = out_dir / "candidate_scores.csv"
        if out_csv.exists():
            tasks.append({"split": split, "status": "skipped_existing", "out_csv": str(out_csv)})
            continue
        log_path = ROOT / "logs" / f"{run_ts}_als_{split}.log"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "score_bpr_als.py"),
            "--split-dir",
            str(SOURCE_VALIDATION_ROOT / split),
            "--out-dir",
            str(out_dir),
            "--models",
            "als",
            "--factors",
            "32",
            "--als-iterations",
            "30",
            "--als-alpha",
            "20",
            "--pop-alphas",
            "4",
            "--seed",
            "42",
        ]
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        tasks.append({
            "split": split,
            "status": "completed" if proc.returncode == 0 and out_csv.exists() else "failed",
            "exit_code": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 1),
            "out_csv": str(out_csv),
            "log": str(log_path),
        })
        if proc.returncode != 0 or not out_csv.exists():
            raise RuntimeError(f"ALS scoring failed for {split}; see {log_path}")
    return tasks


def load_split_frame(split: str, emb192_root: Path, als_root: Path) -> pd.DataFrame:
    e128 = ensemble(emb_seed_paths(SOURCE_EMB128_ROOT, split), "score_emb128")
    e192 = ensemble(emb_seed_paths(emb192_root, split), "score_emb192")[["ID", "score_emb192"]]
    df = e128.merge(e192, on="ID", validate="one_to_one")
    df["rank_emb128"] = rb.within_user_rank_high(df, df["score_emb128"].to_numpy(dtype=float))
    df["rank_emb192"] = rb.within_user_rank_high(df, df["score_emb192"].to_numpy(dtype=float))
    df["score_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    df["base_margin"] = base_rank_margin(df)
    als_path = als_root / f"{split}_readme_bprals" / "candidate_scores.csv"
    als = pd.read_csv(als_path)
    keep = ["ID", STRICT_VARIANT["feature"]]
    missing = [c for c in keep if c not in als.columns]
    if missing:
        raise ValueError(f"missing {missing} in {als_path}")
    df = df.merge(als[keep], on="ID", validate="one_to_one")
    if df[["score_emb128", "score_emb192", STRICT_VARIANT["feature"]]].isna().any().any():
        raise RuntimeError(f"NaN in merged frame for {split}")
    return df


def write_md(payload: dict[str, Any], md_path: Path) -> None:
    strict = payload["strict_confirmation_row"]
    lines = [
        "# Current-best ALS residual independent confirmation",
        "",
        f"- Timestamp: `{payload['timestamp_kst']}`",
        "- Safety: validation-only; no full-test candidate CSV; no Kaggle submit; no hidden/private labels; no external scraping.",
        f"- Verdict: `{payload['verdict']}`",
        "",
        "## Strict pre-registered atlas row",
        "",
        f"- Variant: `{strict['variant']}`",
        f"- feature/weight/band: `{strict['feature']}` / `{strict['weight']}` / `{strict['band']}`",
        f"- mean Δ vs current-best rankblend: {strict['mean_delta_vs_rankblend']:+.10f}",
        f"- min split Δ: {strict['min_delta_vs_rankblend']:+.10f}",
        f"- positive splits: {strict['positive_splits']}/{strict['num_splits']}",
        f"- fixes/breaks: {strict['fixes']}/{strict['breaks']}",
        f"- pooled exact p: {strict['pooled_p_exact']}",
        f"- split deltas: `{strict['split_deltas']}`",
        "",
        "## Diagnostic variants",
        "",
    ]
    for i, row in enumerate(payload["diagnostic_rows"], 1):
        lines.append(
            f"{i}. `{row['variant']}` meanΔ={row['mean_delta_vs_rankblend']:+.10f}, "
            f"minΔ={row['min_delta_vs_rankblend']:+.10f}, pos={row['positive_splits']}/{row['num_splits']}, "
            f"fixes/breaks={row['fixes']}/{row['breaks']}, p={row['pooled_p_exact']}, deltas={row['split_deltas']}"
        )
    lines += ["", "## Artifacts", ""]
    for k, v in payload["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Safety flags", ""]
    for k, v in payload["safety_flags"].items():
        lines.append(f"- {k}: `{str(v).lower()}`")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_root = ROOT / "artifacts" / f"current_best_als_independent_{args.run_ts}"
    emb192_root = run_root / "lightgcn_emb192"
    als_root = run_root / "als_scores"
    log_root = ROOT / "logs" / f"current_best_als_independent_{args.run_ts}"
    out_json = ROOT / (args.out_json or f"reports/{args.run_ts}_current_best_als_independent_confirmation.json")
    out_md = ROOT / (args.out_md or f"reports/{args.run_ts}_current_best_als_independent_confirmation.md")
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    if args.dry_run:
        print(json.dumps(clean({
            "timestamp_kst": args.run_ts,
            "source_independent_root": SOURCE_INDEP_ROOT,
            "split_names": SPLIT_NAMES,
            "strict_variant": STRICT_VARIANT,
            "diagnostic_variants": DIAGNOSTIC_VARIANTS,
            "artifacts": {"run_root": run_root, "emb192_root": emb192_root, "als_root": als_root, "out_json": out_json, "out_md": out_md},
            "safety_flags": SAFETY_FLAGS,
            "verdict": "DRY_RUN_OK",
        }), indent=2, ensure_ascii=False))
        return

    for split in SPLIT_NAMES:
        if not (SOURCE_VALIDATION_ROOT / split / "candidates.csv").exists():
            raise FileNotFoundError(SOURCE_VALIDATION_ROOT / split / "candidates.csv")
        if not (SOURCE_EMB128_ROOT / split / "seed42" / "lightgcn_scores.csv").exists():
            raise FileNotFoundError(SOURCE_EMB128_ROOT / split / "seed42" / "lightgcn_scores.csv")

    ensure_dir(run_root)
    ensure_dir(emb192_root)
    ensure_dir(als_root)
    ensure_dir(log_root)

    train_tasks = oi.launch_train_panel(
        split_names=SPLIT_NAMES,
        validation_root=SOURCE_VALIDATION_ROOT,
        base_root=emb192_root,
        log_root=log_root,
        run_ts=args.run_ts,
        gpus=gpus,
        epochs=args.epochs,
        emb_dim=192,
        n_layers=4,
        reg=1e-3,
        lr=1e-3,
        batch_size=4096,
        max_parallel=args.max_parallel,
    )
    als_tasks = run_als_scores(args.run_ts, als_root)

    by_variant: dict[str, list[dict[str, Any]]] = {str(v["variant"]): [] for v in DIAGNOSTIC_VARIANTS}
    base_rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        df = load_split_frame(split, emb192_root, als_root)
        base_pred = top_half_pred(df, df["score_rankblend"].to_numpy(dtype=float))
        y = df["Label"].to_numpy(dtype=np.int8)
        base_rows.append({"split": split, "base_rankblend_accuracy": float((base_pred == y).mean()), "rows": int(len(df)), "users": int(df["userID"].nunique())})
        for var in DIAGNOSTIC_VARIANTS:
            ev = eval_variant(df, str(var["feature"]), float(var["weight"]), var["band"] if var["band"] is None else int(var["band"]))
            ev.update({"split": split, **var})
            by_variant[str(var["variant"])].append(ev)

    agg_rows: list[dict[str, Any]] = []
    for var in DIAGNOSTIC_VARIANTS:
        name = str(var["variant"])
        rs = by_variant[name]
        deltas = [float(r["delta_vs_rankblend"]) for r in rs]
        fixes = int(sum(int(r["fixes"]) for r in rs))
        breaks = int(sum(int(r["breaks"]) for r in rs))
        row = {
            **var,
            "mean_delta_vs_rankblend": float(np.mean(deltas)),
            "min_delta_vs_rankblend": float(np.min(deltas)),
            "max_delta_vs_rankblend": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "num_splits": len(deltas),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": fixes + breaks,
            "pooled_p_exact": exact_p(fixes, breaks),
            "changed": int(sum(int(r["changed"]) for r in rs)),
            "split_deltas": {str(r["split"]): float(r["delta_vs_rankblend"]) for r in rs},
            "split_rows": rs,
        }
        row["strict_gate_pass"] = strict_gate(row)
        agg_rows.append(row)
    agg_rows.sort(key=lambda r: (r["strict_gate_pass"], r["mean_delta_vs_rankblend"], r["fixes"] - r["breaks"]), reverse=True)
    strict_row = next(r for r in agg_rows if r["variant"] == STRICT_VARIANT["variant"])
    if bool(strict_row["strict_gate_pass"]):
        verdict = "INDEPENDENT_STRICT_PASS_NEEDS_HERMES_REVIEW_NO_CANDIDATE_YET"
    elif strict_row["mean_delta_vs_rankblend"] > 0 and strict_row["positive_splits"] >= 2 and strict_row["fixes"] > strict_row["breaks"]:
        verdict = "INDEPENDENT_WEAK_POSITIVE_STRICT_FAIL"
    else:
        verdict = "INDEPENDENT_REJECT_STRICT_FAIL"

    payload = clean({
        "timestamp_kst": args.run_ts,
        "safety_flags": SAFETY_FLAGS,
        "source_independent_run": str(SOURCE_INDEP_ROOT.relative_to(ROOT)),
        "base": "rank_blend_emb128_emb192_public_best_style_on_fresh_uniform_panel",
        "split_names": SPLIT_NAMES,
        "model_seeds": MODEL_SEEDS,
        "lightgcn_params": {"emb_dim_added": 192, "n_layers": 4, "reg": 1e-3, "lr": 1e-3, "epochs": args.epochs, "batch_size": 4096},
        "strict_gate": {"mean_delta_minimum": 0.0015, "min_delta_minimum": 0.0, "positive_splits_required": 3, "fixes_gt_breaks": True, "pooled_p_exact_lt": 0.05, "only_variant_allowed": STRICT_VARIANT["variant"]},
        "base_rows": base_rows,
        "strict_confirmation_row": strict_row,
        "diagnostic_rows": agg_rows,
        "train_tasks": train_tasks,
        "als_tasks": als_tasks,
        "artifacts": {"run_root": str(run_root.relative_to(ROOT)), "emb192_root": str(emb192_root.relative_to(ROOT)), "als_root": str(als_root.relative_to(ROOT)), "log_root": str(log_root.relative_to(ROOT)), "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT))},
        "verdict": verdict,
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({"json": str(out_json.relative_to(ROOT)), "md": str(out_md.relative_to(ROOT)), "verdict": verdict, "strict_row": strict_row}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
