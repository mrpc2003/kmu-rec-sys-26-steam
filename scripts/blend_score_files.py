#!/usr/bin/env python3
"""Merge candidate score files and evaluate normalized/rank/RRF blends.

No Kaggle submission is performed unless --write-predictions only writes local
candidate CSVs; it still does not call the Kaggle API.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_played_utils import ensure_dir, evaluate_tophalf, normalize_within_user, predict_tophalf, write_json, write_submission_like


def unique_score_columns(df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("z_", "rank_", "pct_rank_")
    return [c for c in df.columns if c.startswith("score_") and not c.startswith(excluded_prefixes)]


def read_and_prefix(path: Path, prefix: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    key_cols = [c for c in ["ID", "userID", "gameID", "Label", "pop_count"] if c in df.columns]
    score_cols = unique_score_columns(df)
    keep = key_cols + score_cols
    out = df[keep].copy()
    if prefix:
        rename = {c: f"score_{prefix}_{c[len('score_') :]}" for c in score_cols}
        out = out.rename(columns=rename)
    return out


def merge_score_files(files: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for spec in files:
        if "=" in spec:
            prefix, path_s = spec.split("=", 1)
        else:
            prefix, path_s = None, spec
        df = read_and_prefix(Path(path_s), prefix)
        if merged is None:
            merged = df
            continue
        key = ["ID", "userID", "gameID"]
        if "Label" in merged.columns and "Label" in df.columns:
            df = df.drop(columns=["Label"])
        if "pop_count" in merged.columns and "pop_count" in df.columns:
            df = df.drop(columns=["pop_count"])
        merged = merged.merge(df, on=key, how="inner", validate="one_to_one")
    if merged is None:
        raise ValueError("no score files provided")
    return merged


def select_columns(df: pd.DataFrame, include_regex: str | None, exclude_regex: str | None, explicit_cols: str | None) -> list[str]:
    cols = unique_score_columns(df)
    if explicit_cols:
        wanted = [c.strip() for c in explicit_cols.split(",") if c.strip()]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(f"explicit score columns missing: {missing}")
        cols = wanted
    if include_regex:
        rx = re.compile(include_regex)
        cols = [c for c in cols if rx.search(c)]
    if exclude_regex:
        rx = re.compile(exclude_regex)
        cols = [c for c in cols if not rx.search(c)]
    return cols


def add_blends(df: pd.DataFrame, base_cols: list[str], rrf_k: float) -> tuple[pd.DataFrame, list[str]]:
    out = normalize_within_user(df, base_cols)
    blend_cols: list[str] = []
    zcols = [f"z_{c}" for c in base_cols if f"z_{c}" in out.columns]
    rank_cols = [f"rank_{c}" for c in base_cols if f"rank_{c}" in out.columns]
    if len(zcols) >= 2:
        out["score_blend_mean_z"] = out[zcols].mean(axis=1)
        out["score_blend_median_z"] = out[zcols].median(axis=1)
        blend_cols += ["score_blend_mean_z", "score_blend_median_z"]
    if len(rank_cols) >= 2:
        out["score_blend_mean_rank_neg"] = -out[rank_cols].mean(axis=1)
        out["score_blend_rrf"] = sum(1.0 / (rrf_k + out[rc]) for rc in rank_cols)
        blend_cols += ["score_blend_mean_rank_neg", "score_blend_rrf"]
    return out, blend_cols


def evaluate(scores: pd.DataFrame, cols: list[str], out_dir: Path) -> list[dict[str, object]]:
    if "Label" not in scores.columns:
        return []
    summaries = []
    for col in cols:
        summary, pred_df = evaluate_tophalf(scores, col, tie_cols=[("pop_count", True), ("gameID", False)])
        summaries.append(summary)
        pred_df[["ID", "userID", "gameID", "Label", col, "Pred", "Correct", "rank_in_user"]].to_csv(out_dir / f"pred_eval_{col}.csv", index=False)
    return sorted(summaries, key=lambda x: x["row_accuracy"], reverse=True)


def write_md(path: Path, summaries: list[dict[str, object]], base_cols: list[str], blend_cols: list[str]) -> None:
    lines = ["# Blend evaluation", "", "## Input score columns", ""]
    for c in base_cols:
        lines.append(f"- `{c}`")
    lines += ["", "## Blend columns", ""]
    for c in blend_cols:
        lines.append(f"- `{c}`")
    lines += ["", "## Evaluation", "", "| rank | score_col | row_acc | user_acc |", "|---:|---|---:|---:|"]
    for i, s in enumerate(summaries, 1):
        lines.append(f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file", action="append", required=True, help="CSV path or prefix=CSV path; may repeat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-regex")
    ap.add_argument("--exclude-regex", default="(^score_blend_|^score_rrf_|mismatch_neg)")
    ap.add_argument("--cols")
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--evaluate-base", action="store_true")
    ap.add_argument("--write-predictions", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    scores = merge_score_files(args.score_file)
    base_cols = select_columns(scores, args.include_regex, args.exclude_regex, args.cols)
    if len(base_cols) < 2:
        raise ValueError(f"need at least two score columns to blend, got {base_cols}")
    scores, blend_cols = add_blends(scores, base_cols, args.rrf_k)
    scores.to_csv(out_dir / "merged_blend_scores.csv", index=False)

    eval_cols = blend_cols + (base_cols if args.evaluate_base else [])
    summaries = evaluate(scores, eval_cols, out_dir)
    write_json(out_dir / "blend_evaluation_summary.json", {"base_columns": base_cols, "blend_columns": blend_cols, "scores": summaries})
    write_md(out_dir / "blend_evaluation_summary.md", summaries, base_cols, blend_cols)

    if args.write_predictions:
        pred_dir = ensure_dir(out_dir / "prediction_csv")
        for col in blend_cols:
            pred_df = predict_tophalf(scores, col, label_col=None, tie_cols=[("pop_count", True), ("gameID", False)])
            write_submission_like(pred_df, pred_dir / f"candidate_{col}.csv")

    if summaries:
        print("[done] best blend scores:")
        for s in summaries[:20]:
            print(f"  {s['score_col']}: row_acc={s['row_accuracy']:.6f}, user_acc={s['per_user_mean_accuracy']:.6f}")
    else:
        print(f"[done] wrote blend scores to {out_dir}")


if __name__ == "__main__":
    main()
