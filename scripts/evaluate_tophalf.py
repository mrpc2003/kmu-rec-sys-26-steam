#!/usr/bin/env python3
"""Evaluate a candidate score file with the per-user top-half rule."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recsys_played_utils import ensure_dir, evaluate_tophalf, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV containing ID,userID,gameID,Label and score columns")
    ap.add_argument("--score-cols", required=True, help="Comma-separated score columns to evaluate")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    out_dir = ensure_dir(args.out_dir)
    summaries = []
    for col in [c.strip() for c in args.score_cols.split(",") if c.strip()]:
        summary, pred_df = evaluate_tophalf(scores, col, tie_cols=[("pop_count", True), ("gameID", False)])
        summaries.append(summary)
        pred_df.to_csv(out_dir / f"pred_{col}.csv", index=False)
    summaries = sorted(summaries, key=lambda x: x["row_accuracy"], reverse=True)
    write_json(out_dir / "tophalf_eval_summary.json", summaries)
    for s in summaries:
        print(f"{s['score_col']}: row_acc={s['row_accuracy']:.6f} user_acc={s['per_user_mean_accuracy']:.6f}")


if __name__ == "__main__":
    main()
