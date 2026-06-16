#!/usr/bin/env python3
"""Fast boundary v1 scored cross-fit gate (NO-SUBMIT by default).

This is the fast replacement for the earlier logistic/pairwise evaluator that was too
slow for an interactive decision. It uses the already-completed panel20 LightGCN score
coverage, evaluates a ridge row-utility model in leave-one-split-out fashion, and writes
diff-band precision curves.

It does not write a full-test candidate and does not submit to Kaggle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from boundary_v1_scored_crossfit_eval import (  # noqa: E402
    DIFF_BANDS,
    FEATURE_COLS,
    SCORE_ROOT_DEFAULT,
    TARGET_GATES,
    clean,
    discover_complete_splits,
    greedy_diffband_pred,
    load_split,
    metrics_for_band,
)
from recsys_played_utils import write_json  # noqa: E402

PANEL20_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
OUT_DIR_DEFAULT = ROOT / "reports"


def row_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(y, dtype=np.int8) == np.asarray(pred, dtype=np.int8)).mean())


def fit_ridge_utility(train_frames: list[pd.DataFrame], alpha: float) -> Any:
    train = pd.concat([f[f["boundary_band"]] for f in train_frames], ignore_index=True)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=0))
    model.fit(train[FEATURE_COLS].to_numpy(dtype=float), train["Label"].to_numpy(dtype=float))
    return model


def evaluate_fast(splits: list[Any], alpha: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for heldout_idx, heldout in enumerate(splits):
        train_frames = [s.frame for i, s in enumerate(splits) if i != heldout_idx]
        model = fit_ridge_utility(train_frames, alpha=alpha)
        utility = model.predict(heldout.frame[FEATURE_COLS].to_numpy(dtype=float))
        y = heldout.frame["Label"].to_numpy(dtype=np.int8)
        anchor = heldout.frame["anchor_pred"].to_numpy(dtype=np.int8)
        base_rows.append(
            {
                "split": heldout.split,
                "rows": int(len(heldout.frame)),
                "users": int(heldout.frame["userID"].nunique()),
                "anchor_accuracy": row_accuracy(y, anchor),
                "anchor_errors": int((anchor != y).sum()),
                "boundary_band_rows": int(heldout.frame["boundary_band"].sum()),
            }
        )
        for band in DIFF_BANDS:
            pred = greedy_diffband_pred(heldout.frame, utility, band)
            rows.append(metrics_for_band(heldout.frame, pred, "ridge_fast", band, heldout.split))
    curve = pd.DataFrame(rows)
    base = pd.DataFrame(base_rows)
    agg_rows: list[dict[str, Any]] = []
    for (model_name, band), g in curve.groupby(["model", "band_total_row_diff_target"], sort=True):
        gate_top2 = TARGET_GATES.get(("top2", int(band)))
        gate_top1 = TARGET_GATES.get(("top1", int(band)))
        mean_prec = float(g["flip_precision"].dropna().mean()) if g["flip_precision"].notna().any() else float("nan")
        pos_ratio = float((g["net_gain_rows"] > 0).mean())
        worst = int(g["net_gain_rows"].min())
        agg_rows.append(
            {
                "model": model_name,
                "band_total_row_diff_target": int(band),
                "splits": int(len(g)),
                "mean_changed_rows": float(g["changed_rows"].mean()),
                "mean_flip_precision": mean_prec,
                "mean_net_gain_rows": float(g["net_gain_rows"].mean()),
                "mean_delta_accuracy": float(g["delta_accuracy"].mean()),
                "positive_split_ratio": pos_ratio,
                "worst_split_net_gain_rows": worst,
                "best_split_net_gain_rows": int(g["net_gain_rows"].max()),
                "top2_gate_precision": gate_top2,
                "top2_gate_pass": bool(gate_top2 is not None and mean_prec >= gate_top2 and pos_ratio >= 0.70 and worst >= -5),
                "top1_gate_precision": gate_top1,
                "top1_gate_pass": bool(gate_top1 is not None and mean_prec >= gate_top1 and pos_ratio >= 0.80 and worst >= -5),
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values(
        ["top2_gate_pass", "mean_net_gain_rows", "mean_flip_precision"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return curve, base, agg


def write_md(path: Path, complete_splits: list[str], base: pd.DataFrame, agg: pd.DataFrame, payload: dict[str, Any]) -> None:
    lines = [
        "# boundary v1 scored split20 fast ridge eval",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- full_test_candidate_written: false",
        "",
        "## coverage",
        "",
        f"- complete scored splits: {len(complete_splits)}",
        f"- mean anchor accuracy: {base['anchor_accuracy'].mean():.6f}",
        f"- mean boundary band rows: {base['boundary_band_rows'].mean():.1f}",
        "",
        "## aggregate diff-band result",
        "",
        "| model | band | mean precision | mean net rows | positive split ratio | worst split | top2 pass | top1 pass |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['band_total_row_diff_target'])} | {float(r['mean_flip_precision']):.3f} | "
            f"{float(r['mean_net_gain_rows']):.2f} | {float(r['positive_split_ratio']):.2f} | {int(r['worst_split_net_gain_rows'])} | "
            f"{bool(r['top2_gate_pass'])} | {bool(r['top1_gate_pass'])} |"
        )
    lines += [
        "",
        "## submission readiness",
        "",
        f"`{payload['submission_readiness']}`",
        "",
        "top2/top1 pass가 없으면 full-test candidate를 만들지 않는다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-root", default=str(SCORE_ROOT_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_splits = sorted(p.name for p in PANEL20_ROOT.iterdir() if p.is_dir() and p.name.startswith("val_random_uniform_seed"))
    complete = discover_complete_splits(Path(args.score_root), all_splits)
    if len(complete) != 20:
        raise SystemExit(f"Need 20 complete splits, got {len(complete)}")
    splits = [load_split(Path(args.score_root), s) for s in complete]
    curve, base, agg = evaluate_fast(splits, alpha=args.alpha)
    curve_path = out_dir / "boundary_v1_diffband_precision_curve_scored_fast.csv"
    base_path = out_dir / "boundary_v1_scored_split_base_metrics_fast.csv"
    agg_path = out_dir / "boundary_v1_diffband_precision_curve_scored_fast_aggregate.csv"
    curve.to_csv(curve_path, index=False)
    base.to_csv(base_path, index=False)
    agg.to_csv(agg_path, index=False)
    payload = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_candidate_written": False,
        "model": "ridge_fast",
        "alpha": args.alpha,
        "complete_splits": complete,
        "curve_csv": str(curve_path),
        "base_metrics_csv": str(base_path),
        "aggregate_csv": str(agg_path),
        "top2_any_pass": bool(agg["top2_gate_pass"].any()),
        "top1_any_pass": bool(agg["top1_gate_pass"].any()),
        "submission_readiness": "PASS_REVIEW_REQUIRED" if bool(agg["top2_gate_pass"].any()) else "FAIL__NO_SCORED_GATE_PASS",
    }
    write_json(out_dir / "boundary_v1_scored_split20_fast_eval.json", clean(payload))
    write_md(out_dir / "boundary_v1_scored_split20_fast_eval.md", complete, base, agg, payload)
    print(json.dumps(clean(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
