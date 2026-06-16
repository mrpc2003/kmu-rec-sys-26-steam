#!/usr/bin/env python3
"""Aggregate validation-only LightGCN checkpoint prediction averaging probes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))

from recsys_played_utils import evaluate_tophalf  # noqa: E402

BASELINE_4SEED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
SEEDS = [42, 123, 2024, 7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    run_root = ROOT / args.root
    score_files = sorted(run_root.glob("*/seed*/checkpoint_scores.csv"))
    if not score_files:
        raise SystemExit(f"No checkpoint_scores.csv under {run_root}")

    # Determine common variants present across all seed files.
    variants: set[str] | None = None
    file_rows = []
    for p in score_files:
        df_head = pd.read_csv(p, nrows=3)
        cols = {c for c in df_head.columns if c.startswith("score_")}
        variants = cols if variants is None else variants & cols
        file_rows.append({"path": p, "split": p.parts[-3], "seed": int(p.parts[-2].replace("seed", ""))})
    variants = set() if variants is None else variants
    if not variants:
        raise SystemExit("No common score variants found")

    by_split = []
    run_details = []
    for split in sorted({r["split"] for r in file_rows}):
        split_files = [r for r in file_rows if r["split"] == split]
        present_seeds = sorted(r["seed"] for r in split_files)
        for variant in sorted(variants):
            merged = None
            seed_cols = []
            for r in sorted(split_files, key=lambda x: x["seed"]):
                df = pd.read_csv(r["path"])
                seed_col = f"{variant}_seed{r['seed']}"
                if merged is None:
                    merged = df[["ID", "userID", "gameID", "Label", variant]].rename(columns={variant: seed_col})
                else:
                    merged = merged.merge(df[["ID", variant]].rename(columns={variant: seed_col}), on="ID", how="inner")
                seed_cols.append(seed_col)
            assert merged is not None
            merged["ensemble_score"] = merged[seed_cols].mean(axis=1)
            summ, _ = evaluate_tophalf(merged, "ensemble_score", label_col="Label", user_col="userID", id_col="ID")
            acc = float(summ["row_accuracy"])
            delta = acc - BASELINE_4SEED[split]
            by_split.append({
                "split": split,
                "variant": variant,
                "seeds_present": present_seeds,
                "n_seeds": len(present_seeds),
                "ensemble_acc": acc,
                "baseline_4seed_acc": BASELINE_4SEED[split],
                "delta_vs_baseline": delta,
                "rows": int(summ["rows"]),
                "users": int(summ["users"]),
            })
    df = pd.DataFrame(by_split)
    overall = (
        df.groupby("variant")
        .agg(
            splits=("split", "nunique"),
            mean_acc=("ensemble_acc", "mean"),
            mean_baseline=("baseline_4seed_acc", "mean"),
            mean_delta=("delta_vs_baseline", "mean"),
            min_delta=("delta_vs_baseline", "min"),
            max_delta=("delta_vs_baseline", "max"),
            positive_splits=("delta_vs_baseline", lambda s: int((s > 0).sum())),
        )
        .reset_index()
        .sort_values(["mean_delta", "min_delta"], ascending=[False, False])
    )
    best = overall.iloc[0].to_dict()
    gate = {
        "mean_delta_required_for_swa_axis": 0.0015,
        "min_delta_required": 0.0,
        "splits_required": 3,
        "candidate_materialization_allowed": False,
    }
    gate_pass = bool(best["mean_delta"] >= gate["mean_delta_required_for_swa_axis"] and best["min_delta"] >= gate["min_delta_required"] and best["splits"] >= gate["splits_required"])
    payload = {
        "run_root": args.root,
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "external_metadata_used": False,
        "gate": gate,
        "gate_pass": gate_pass,
        "best_overall": best,
        "overall": overall.to_dict(orient="records"),
        "by_split": by_split,
        "score_files": [str(r["path"].relative_to(ROOT)) for r in file_rows],
    }
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    lines = [
        "# LightGCN checkpoint prediction averaging aggregate",
        "",
        f"- run root: `{args.root}`",
        "- validation only: true",
        "- Kaggle submit: false",
        "- candidate CSV written: false",
        "- external metadata: false",
        "",
        "## Overall",
        "",
        "| variant | splits | mean acc | baseline | mean Δ | min Δ | max Δ | +splits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overall.to_dict(orient="records"):
        lines.append(
            f"| `{r['variant']}` | {int(r['splits'])} | {float(r['mean_acc']):.6f} | {float(r['mean_baseline']):.6f} "
            f"| {float(r['mean_delta']):+.6f} | {float(r['min_delta']):+.6f} | {float(r['max_delta']):+.6f} | {int(r['positive_splits'])} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        f"- pass: `{gate_pass}`",
        f"- required: mean Δ >= {gate['mean_delta_required_for_swa_axis']:+.4f}, min Δ >= {gate['min_delta_required']:+.4f}, splits >= {gate['splits_required']}",
        "- full-test materialization remains blocked until explicit later approval.",
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
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "gate_pass": gate_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()
