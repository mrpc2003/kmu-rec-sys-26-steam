#!/usr/bin/env python3
"""Aggregate validation-only pseudo-label transduction probe summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    run_root = ROOT / args.root
    rows = []
    for p in sorted(run_root.glob("*/topn*_margin*/student_seed*/summary.json")):
        data = json.loads(p.read_text())
        cfg = data.get("config", {})
        pm = data.get("pseudo_meta", {})
        rows.append({
            "split": data.get("split"),
            "top_n": cfg.get("top_n"),
            "min_margin": cfg.get("min_margin"),
            "student_seed": cfg.get("student_seed"),
            "teacher_acc": data.get("teacher_4seed_acc_recomputed"),
            "student_acc": data.get("student_acc"),
            "delta": data.get("delta_vs_teacher_recomputed"),
            "pseudo_edges": pm.get("pseudo_edges"),
            "pseudo_precision_diag": pm.get("pseudo_label_precision_diagnostic"),
            "tier": data.get("tier"),
            "summary_path": str(p.relative_to(ROOT)),
            "validation_only": data.get("validation_only"),
            "candidate_csv_written": data.get("candidate_csv_written"),
            "no_kaggle_submit": data.get("no_kaggle_submit"),
            "external_metadata_used": data.get("external_metadata_used"),
        })
    if not rows:
        raise SystemExit(f"No summary files under {run_root}")
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["top_n", "min_margin", "split"], dropna=False)
        .agg(
            n=("student_seed", "count"),
            mean_student_acc=("student_acc", "mean"),
            mean_teacher_acc=("teacher_acc", "mean"),
            mean_delta=("delta", "mean"),
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
            mean_pseudo_precision=("pseudo_precision_diag", "mean"),
        )
        .reset_index()
    )
    overall = (
        df.groupby(["top_n", "min_margin"], dropna=False)
        .agg(
            n=("student_seed", "count"),
            splits=("split", "nunique"),
            mean_student_acc=("student_acc", "mean"),
            mean_teacher_acc=("teacher_acc", "mean"),
            mean_delta=("delta", "mean"),
            min_delta=("delta", "min"),
            max_delta=("delta", "max"),
            positive_runs=("delta", lambda s: int((s > 0).sum())),
            mean_pseudo_precision=("pseudo_precision_diag", "mean"),
        )
        .reset_index()
        .sort_values(["mean_delta", "min_delta"], ascending=[False, False])
    )
    best = overall.iloc[0].to_dict()
    gate = {
        "mean_delta_required": 0.005,
        "min_delta_required": -0.0015,
        "all_splits_required": 3,
        "at_least_two_seed_groups": True,
    }
    best_pass = bool(
        best["mean_delta"] >= gate["mean_delta_required"]
        and best["min_delta"] >= gate["min_delta_required"]
        and best["splits"] >= gate["all_splits_required"]
    )
    payload = {
        "run_root": args.root,
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "external_metadata_used": False,
        "gate": gate,
        "best_overall": best,
        "gate_pass": best_pass,
        "overall": overall.to_dict(orient="records"),
        "by_split": grouped.to_dict(orient="records"),
        "runs": df.to_dict(orient="records"),
    }
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    lines = [
        "# Pseudo-label transduction probe aggregate",
        "",
        f"- run root: `{args.root}`",
        "- validation only: true",
        "- Kaggle submit: false",
        "- candidate CSV written: false",
        "- external metadata: false",
        "",
        "## Overall",
        "",
        "| top_n | margin | runs | splits | mean student | mean teacher | mean Δ | min Δ | max Δ | +runs | pseudo precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overall.to_dict(orient="records"):
        lines.append(
            f"| {int(r['top_n'])} | {float(r['min_margin']):.4g} | {int(r['n'])} | {int(r['splits'])} "
            f"| {float(r['mean_student_acc']):.6f} | {float(r['mean_teacher_acc']):.6f} "
            f"| {float(r['mean_delta']):+.6f} | {float(r['min_delta']):+.6f} | {float(r['max_delta']):+.6f} "
            f"| {int(r['positive_runs'])} | {float(r['mean_pseudo_precision']):.4f} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- pass: `{best_pass}`",
        f"- required: mean Δ >= {gate['mean_delta_required']:+.4f}, min Δ >= {gate['min_delta_required']:+.4f}, splits >= {gate['all_splits_required']}",
        "",
        "## Best row",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "gate_pass": best_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()
