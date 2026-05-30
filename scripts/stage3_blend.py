#!/usr/bin/env python3
"""Stage3 blend: merge Stage2 scores + new time-decay BM25/CW-lite axes.

Validation-only. No Kaggle submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import evaluate_tophalf, normalize_within_user, ensure_dir, write_json


def merge_scores(stage2_path: Path, nextstep_path: Path) -> pd.DataFrame:
    s2 = pd.read_csv(stage2_path)
    ns = pd.read_csv(nextstep_path)
    # Merge on ID (both have same candidate rows)
    new_cols = [c for c in ns.columns if c.startswith("score_time_itemknn_bm25") or c == "score_cw_weighted_implicit_logit" or c.startswith("score_review_pseudocat") or c == "score_graph_svd_k64"]
    merged = s2.merge(ns[["ID"] + new_cols], on="ID", how="left", suffixes=("", "_ns"))
    return merged


def build_stage3_blends(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    # Core Stage2 axes
    s2_axes = ["score_itemknn_bm25_top3", "score_ease_lambda1000", "score_als_f32_it30_alpha20_popa2"]
    # New axes from next-step
    new_axes = ["score_time_itemknn_bm25_hl90_top3", "score_cw_weighted_implicit_logit", "score_review_pseudocat_affinity", "score_graph_svd_k64"]
    all_axes = [c for c in s2_axes + new_axes if c in out.columns]

    # Within-user z-score normalize all axes
    out = normalize_within_user(out, all_axes)
    z_cols = [f"z_{c}" for c in all_axes if f"z_{c}" in out.columns]
    s2_z = [f"z_{c}" for c in s2_axes if f"z_{c}" in out.columns]
    new_z = [f"z_{c}" for c in new_axes if f"z_{c}" in out.columns]

    # Stage3 blends
    blend_cols = []

    # 1. Stage2 mean-z (baseline anchor)
    if s2_z:
        out["score_stage2_mean_z"] = out[s2_z].mean(axis=1).astype(np.float32)
        blend_cols.append("score_stage2_mean_z")

    # 2. Stage3 all-axes mean-z
    if z_cols:
        out["score_stage3_all_mean_z"] = out[z_cols].mean(axis=1).astype(np.float32)
        blend_cols.append("score_stage3_all_mean_z")

    # 3. Stage3 weighted: heavier on proven axes
    weights = {}
    for c in s2_z:
        weights[c] = 0.30 / max(len(s2_z), 1)
    for c in new_z:
        weights[c] = 0.10 / max(len(new_z), 1)
    # Boost time-decay BM25 (proven strong)
    bm25_z = [c for c in new_z if "bm25" in c]
    for c in bm25_z:
        weights[c] = 0.15
    # Normalize weights
    total_w = sum(weights.values())
    if total_w > 0:
        out["score_stage3_weighted_z"] = sum(out[c] * (w / total_w) for c, w in weights.items()).astype(np.float32)
        blend_cols.append("score_stage3_weighted_z")

    # 4. Stage3 top-4 axes only (ItemKNN BM25, EASE, ALS, time-decay BM25)
    top4 = [c for c in ["z_score_itemknn_bm25_top3", "z_score_ease_lambda1000", "z_score_als_f32_it30_alpha20_popa2", "z_score_time_itemknn_bm25_hl90_top3"] if c in out.columns]
    if top4:
        out["score_stage3_top4_mean_z"] = out[top4].mean(axis=1).astype(np.float32)
        blend_cols.append("score_stage3_top4_mean_z")

    # 5. Stage3 top-4 + CW-lite (if CW passes gate)
    cw_z = "z_score_cw_weighted_implicit_logit"
    if cw_z in out.columns:
        top4_cw = top4 + [cw_z]
        out["score_stage3_top4_cw_mean_z"] = out[top4_cw].mean(axis=1).astype(np.float32)
        blend_cols.append("score_stage3_top4_cw_mean_z")

    # Also keep individual axes for evaluation
    eval_cols = all_axes + blend_cols
    return out, eval_cols


def run_split(stage2_dir: Path, nextstep_dir: Path, split: str, out_dir: Path) -> dict[str, object]:
    stage2_path = stage2_dir / f"{split}_stage2_blend" / "merged_blend_scores.csv"
    nextstep_path = nextstep_dir / split / "next_step_scores.csv"
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)
    if not nextstep_path.exists():
        raise FileNotFoundError(nextstep_path)

    merged = merge_scores(stage2_path, nextstep_path)
    scored, eval_cols = build_stage3_blends(merged)

    summaries = []
    for col in eval_cols:
        if col not in scored.columns:
            continue
        summary, _ = evaluate_tophalf(scored, col, label_col="Label", user_col="userID", id_col="ID")
        summaries.append(summary)
    summaries = sorted(summaries, key=lambda s: (s["row_accuracy"], s["per_user_mean_accuracy"]), reverse=True)

    split_out = ensure_dir(out_dir / split)
    keep = ["ID", "userID", "gameID", "Label"] + eval_cols
    scored[[c for c in keep if c in scored.columns]].to_csv(split_out / "stage3_scores.csv", index=False)
    result = {"split": split, "rows": int(len(scored)), "summaries": summaries}
    write_json(split_out / "summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2-dir", default="artifacts/scores")
    ap.add_argument("--nextstep-dir", default="artifacts/paper_guided_next_steps_20260530")
    ap.add_argument("--out-dir", default="artifacts/stage3_blend_20260530")
    ap.add_argument("--splits", nargs="*", default=[
        "val_random_sqrtpop_seed42",
        "val_recent_sqrtpop_seed42",
        "val_random_popbin_seed42",
    ])
    ap.add_argument("--report-json", default="reports/20260530_stage3_blend.json")
    ap.add_argument("--report-md", default="reports/20260530_stage3_blend.md")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    results = []
    for split in args.splits:
        print(f"[stage3] {split}", flush=True)
        results.append(run_split(Path(args.stage2_dir), Path(args.nextstep_dir), split, out_dir))

    # Report
    payload = {"note": "Stage3 blend validation-only. No Kaggle submission.", "results": results}
    write_json(args.report_json, payload)

    lines = [
        "# KMU RecSys 26 Steam — Stage3 blend validation",
        "",
        "Stage2 scores + new time-decay BM25 / CW-lite / review pseudo-cat axes merged and re-evaluated. No Kaggle submission.",
        "",
        "## Best by split",
        "",
        "| split | best score | row acc | per-user mean acc |",
        "|---|---|---:|---:|",
    ]
    for r in results:
        b = r["summaries"][0]
        lines.append(f"| `{r['split']}` | `{b['score_col']}` | {b['row_accuracy']:.6f} | {b['per_user_mean_accuracy']:.6f} |")

    # Stage2 anchor comparison
    lines.extend([
        "",
        "## Stage2 anchor comparison",
        "",
        "| split | Stage2 best (score_blend_mean_z) | Stage3 best |",
        "|---|---:|---:|",
    ])
    stage2_anchors = {"val_random_sqrtpop_seed42": 0.659732, "val_recent_sqrtpop_seed42": 0.626025, "val_random_popbin_seed42": 0.590818}
    for r in results:
        b = r["summaries"][0]
        anchor = stage2_anchors.get(r["split"], 0)
        delta = b["row_accuracy"] - anchor
        lines.append(f"| `{r['split']}` | {anchor:.6f} | {b['row_accuracy']:.6f} ({delta:+.6f}) |")

    lines.extend(["", "## Full tables", ""])
    for r in results:
        lines.extend([f"### {r['split']}", "", "| rank | score | row acc | per-user mean acc |", "|---:|---|---:|---:|"])
        for i, s in enumerate(r["summaries"][:15], 1):
            lines.append(f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
        lines.append("")

    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_md": args.report_md, "report_json": args.report_json, "splits": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
