#!/usr/bin/env python3
"""Aggressive user-confidence-gated z-blend probe for KMURecSys26 Steam.

Validation-only, no hidden/test read, no candidate/submission CSV. This probes whether
the weak but sign-stable z(emb128)+z(emb192) signal can be restricted to unsupervised
user subgroups where it is less noisy: low anchor margin, high capacity disagreement,
small/large candidate count, or combinations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggressive_rank_frontier_panel import (  # noqa: E402
    SPLITS,
    BASE_EXPECTED,
    MDE,
    build_split_frame,
    exact_two_sided_binom_p,
    predict_tophalf,
)

OUT_DIR_DEFAULT = ROOT / "artifacts/aggressive_user_gated_zblend"
REPORT_JSON_DEFAULT = ROOT / "reports/20260602_aggressive_user_gated_zblend.json"
REPORT_MD_DEFAULT = ROOT / "reports/20260602_aggressive_user_gated_zblend.md"


def user_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = df["e128"].to_numpy(float)
    z128 = df["z_e128"].to_numpy(float)
    z192 = df["z_e192"].to_numpy(float)
    z64 = df["z_e64"].to_numpy(float)
    zblend = z128 + z192
    for user, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        n = len(idx); k = n // 2
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        if k <= 0 or k >= n:
            margin = 0.0
        else:
            margin = float(base[order[k - 1]] - base[order[k]])
        base_top = set(order[:k].tolist())
        zorder = idx[np.argsort(zblend[idx], kind="mergesort")[::-1]]
        z_top = set(zorder[:k].tolist())
        rows.append({
            "userID": user,
            "n": n,
            "margin": margin,
            "absdiff192_mean": float(np.mean(np.abs(z192[idx] - z128[idx]))),
            "absdiff64_mean": float(np.mean(np.abs(z64[idx] - z128[idx]))),
            "zblend_base_changed": len(base_top.symmetric_difference(z_top)),
            "zblend_top_overlap": len(base_top & z_top) / max(k, 1),
        })
    return pd.DataFrame(rows)


def masks_from_stats(stats: pd.DataFrame) -> dict[str, set[Any]]:
    masks: dict[str, set[Any]] = {}
    users = stats["userID"].to_numpy()
    masks["all_users"] = set(users.tolist())
    for col, prefix in [
        ("margin", "margin"),
        ("absdiff192_mean", "diff192"),
        ("absdiff64_mean", "diff64"),
        ("zblend_base_changed", "changed"),
        ("n", "candn"),
    ]:
        vals = stats[col].to_numpy(float)
        for q in [0.20, 0.40, 0.60, 0.80]:
            thr = float(np.quantile(vals, q))
            masks[f"{prefix}_low_q{q:g}"] = set(stats.loc[stats[col] <= thr, "userID"].tolist())
            masks[f"{prefix}_high_q{q:g}"] = set(stats.loc[stats[col] >= thr, "userID"].tolist())
    # Predeclared combinations: uncertain boundary AND capacity disagreement, or changed by zblend.
    for q in [0.40, 0.60, 0.80]:
        margin_thr = float(np.quantile(stats["margin"].to_numpy(float), q))
        diff_thr = float(np.quantile(stats["absdiff192_mean"].to_numpy(float), 1.0 - q / 2.0))
        changed_thr = float(np.quantile(stats["zblend_base_changed"].to_numpy(float), 1.0 - q / 2.0))
        masks[f"margin_low_q{q:g}_and_diff192_high"] = set(stats.loc[(stats["margin"] <= margin_thr) & (stats["absdiff192_mean"] >= diff_thr), "userID"].tolist())
        masks[f"margin_low_q{q:g}_and_changed_high"] = set(stats.loc[(stats["margin"] <= margin_thr) & (stats["zblend_base_changed"] >= changed_thr), "userID"].tolist())
        masks[f"margin_low_q{q:g}_or_diff192_high"] = set(stats.loc[(stats["margin"] <= margin_thr) | (stats["absdiff192_mean"] >= diff_thr), "userID"].tolist())
    return masks


def boundary_score_for_selected_users(df: pd.DataFrame, selected_users: set[Any], band: int) -> np.ndarray:
    """Anchor on e128; within selected users only, reorder boundary band by z128+z192."""
    out = df["e128"].to_numpy(float).copy()
    base = df["e128"].to_numpy(float)
    zblend = (df["z_e128"].to_numpy(float) + df["z_e192"].to_numpy(float))
    for user, idx_raw in df.groupby("userID", sort=False).indices.items():
        if user not in selected_users:
            continue
        idx = np.asarray(idx_raw)
        n = len(idx); k = n // 2
        if k <= 0:
            continue
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        lo = max(0, k - band); hi = min(n, k + band)
        band_ids = order[lo:hi]
        if len(band_ids) == 0:
            continue
        # Keep outside boundary untouched; give boundary values enough range to reorder only inside.
        bvals = zblend[band_ids]
        if float(np.std(bvals)) > 1e-12:
            bvals = (bvals - float(np.mean(bvals))) / float(np.std(bvals))
        else:
            bvals = np.zeros(len(band_ids), dtype=float)
        out[band_ids] = float(np.mean(base[band_ids])) + bvals * 1e-3
    return out


def score_for_variant(df: pd.DataFrame, selected_users: set[Any], mode: str, band: int) -> np.ndarray:
    base = df["e128"].to_numpy(float)
    zblend = (df["z_e128"].to_numpy(float) + df["z_e192"].to_numpy(float))
    if mode == "full_user_zblend":
        out = base.copy()
        mask = df["userID"].isin(selected_users).to_numpy()
        out[mask] = zblend[mask]
        return out
    if mode == "boundary_user_zblend":
        return boundary_score_for_selected_users(df, selected_users, band=band)
    raise ValueError(mode)


def metric(df: pd.DataFrame, score: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    pred = predict_tophalf(df, score)
    y = df["Label"].to_numpy(np.int8)
    ok = pred == y
    base_ok = base_pred == y
    fixes = int((~base_ok & ok).sum())
    breaks = int((base_ok & ~ok).sum())
    return {
        "accuracy": float(ok.mean()),
        "delta": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": fixes + breaks,
        "changed": int((pred != base_pred).sum()),
    }


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v); return None if math.isnan(x) or math.isinf(x) else x
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--report-json", default=str(REPORT_JSON_DEFAULT))
    ap.add_argument("--report-md", default=str(REPORT_MD_DEFAULT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    base_rows = []
    for split in SPLITS:
        df = build_split_frame(split)
        base_pred = predict_tophalf(df, df["e128"].to_numpy(float))
        base_acc = float((base_pred == df["Label"].to_numpy(np.int8)).mean())
        if abs(base_acc - BASE_EXPECTED[split]) > 1e-9:
            raise RuntimeError(f"{split}: base mismatch {base_acc:.12f} != {BASE_EXPECTED[split]:.12f}")
        base_rows.append({"split": split, "base_accuracy": base_acc})
        stats = user_stats(df)
        masks = masks_from_stats(stats)
        for mask_name, users in masks.items():
            if not users:
                continue
            for mode in ["full_user_zblend", "boundary_user_zblend"]:
                bands = [4, 8, 16] if mode == "boundary_user_zblend" else [0]
                for band in bands:
                    variant = f"{mode}__{mask_name}" + (f"__B{band}" if band else "")
                    score = score_for_variant(df, users, mode=mode, band=band)
                    split_rows.append({"split": split, "variant": variant, "mode": mode, "mask": mask_name, "band": band, "selected_users": len(users), **metric(df, score, base_pred)})

    split_path = out_dir / "aggressive_user_gated_zblend_split_metrics.csv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "variant", "mode", "mask", "band", "selected_users", "accuracy", "delta", "fixes", "breaks", "discordant", "changed"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(split_rows)

    agg_rows = []
    by: dict[str, list[dict[str, Any]]] = {}
    for r in split_rows:
        by.setdefault(r["variant"], []).append(r)
    for variant, rows in by.items():
        fixes = int(sum(int(r["fixes"]) for r in rows)); breaks = int(sum(int(r["breaks"]) for r in rows))
        deltas = [float(r["delta"]) for r in rows]
        ratio = float("inf") if breaks == 0 and fixes > 0 else (fixes / breaks if breaks else 1.0)
        first = rows[0]
        agg_rows.append({
            "variant": variant,
            "mode": first["mode"],
            "mask": first["mask"],
            "band": first["band"],
            "mean_selected_users": float(np.mean([int(r["selected_users"]) for r in rows])),
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": fixes,
            "breaks": breaks,
            "fix_break_ratio": ratio,
            "discordant": fixes + breaks,
            "p_exact_two_sided": exact_two_sided_binom_p(fixes, fixes + breaks),
            "split_deltas": deltas,
            "changed_total": int(sum(int(r["changed"]) for r in rows)),
        })
    agg_rows.sort(key=lambda r: (float(r["mean_delta"]), int(r["fixes"])), reverse=True)
    agg_path = out_dir / "aggressive_user_gated_zblend_aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "mode", "mask", "band", "mean_selected_users", "mean_delta", "min_delta", "max_delta", "positive_splits", "fixes", "breaks", "fix_break_ratio", "discordant", "p_exact_two_sided", "split_deltas", "changed_total"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(agg_rows)

    strict = [r for r in agg_rows if float(r["mean_delta"]) >= MDE and int(r["positive_splits"]) == 3 and float(r["p_exact_two_sided"]) <= 0.01 and float(r["fix_break_ratio"]) >= 1.20 and float(r["min_delta"]) >= 0.001]
    best = agg_rows[0]
    verdict = "STRICT_PASS" if strict else ("MANUAL_RISK_SIGNAL" if float(best["mean_delta"]) > 0 and int(best["positive_splits"]) == 3 else "REJECT")
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "splits": list(SPLITS),
        "base": base_rows,
        "variant_count": len(agg_rows),
        "mde": MDE,
        "strict_pass_count": len(strict),
        "verdict": verdict,
        "top_variants": clean(agg_rows[:40]),
        "output_files": {"split_metrics": str(split_path), "aggregate": str(agg_path)},
        "note": "User-subgroup-gated z(emb128)+z(emb192) probe. Validation-only; no candidate/submission materialized.",
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Aggressive user-gated z-blend probe\n\n",
        f"- verdict: **{verdict}**\n",
        f"- variants: `{len(agg_rows)}`\n",
        f"- strict pass count: `{len(strict)}`\n",
        "- safety: validation-only; no hidden/test read; no candidate CSV; no Kaggle submit.\n\n",
        "| rank | variant | mean Δ | min~max Δ | splits+ | fixes | breaks | ratio | p | changed |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for i, r in enumerate(agg_rows[:25], 1):
        lines.append(f"| {i} | `{r['variant']}` | {float(r['mean_delta']):+.6f} | {float(r['min_delta']):+.6f}~{float(r['max_delta']):+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {float(r['fix_break_ratio']):.3f} | {float(r['p_exact_two_sided']):.4g} | {r['changed_total']} |\n")
    Path(args.report_md).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report_json": args.report_json, "best": clean(best)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
