#!/usr/bin/env python3
"""Aggressive boundary-only pairwise agreement probe for KMURecSys26 Steam.

Validation-only probe. It uses existing validation score files for emb64/emb128/emb192
4-seed ensembles and never reads hidden/test candidates or writes candidate/submission CSVs.

Idea: preserve emb128 as the anchor, only allow swaps in a symmetric boundary band around
the per-user top-half cutoff, and promote a bottom-band row only when cross-capacity
pairwise agreement says it should beat one or more currently selected rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123")
SEEDS = (42, 123, 2024, 7)
BASE_EXPECTED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
MDE = 0.00355
OUT_DIR_DEFAULT = ROOT / "artifacts/aggressive_boundary_pairwise"
REPORT_JSON_DEFAULT = ROOT / "reports/20260602_aggressive_boundary_pairwise.json"
REPORT_MD_DEFAULT = ROOT / "reports/20260602_aggressive_boundary_pairwise.md"


def exact_two_sided_binom_p(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(kk + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_layermix_uniform", "score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def path_for(axis: str, split: str, seed: int) -> Path:
    if axis == "e64":
        if split == "val_random_uniform_seed42":
            if seed == 42:
                return ROOT / "artifacts/lightgcn_ood_robustness" / split / "lightgcn_scores.csv"
            return ROOT / f"artifacts/lightgcn_uniform_eval/seed{seed}" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/split_panel_emb64/seed{seed}" / split / "lightgcn_scores.csv"
    if axis == "e128":
        if split == "val_random_uniform_seed42":
            if seed == 42:
                return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
            return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"
    if axis == "e192":
        if split == "val_random_uniform_seed42":
            base = "emb192_L4_r3" if seed == 42 else f"emb192_L4_r3_seed{seed}"
            return ROOT / "artifacts/capacity_uniform" / base / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/split_panel_emb192/{split}/seed{seed}/lightgcn_scores.csv"
    raise ValueError(axis)


def load_axis(axis: str, split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    cols: list[str] = []
    for seed in SEEDS:
        p = path_for(axis, split, seed)
        if not p.exists():
            raise FileNotFoundError(f"Missing score file axis={axis} split={split} seed={seed}: {p}")
        d = pd.read_csv(p)
        sc = score_col(d)
        need = {"ID", "userID", "gameID", "Label", sc}
        if not need.issubset(d.columns):
            raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
        col = f"{axis}_seed{seed}"
        part = d[["ID", "userID", "gameID", "Label", sc]].rename(columns={sc: col})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part[["ID", col]], on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed while merging {p}")
        cols.append(col)
    assert merged is not None
    merged[axis] = merged[cols].mean(axis=1)
    merged[f"std_{axis}"] = merged[cols].std(axis=1, ddof=0)
    return merged[["ID", "userID", "gameID", "Label", axis, f"std_{axis}"]]


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    mask = sd > 1e-12
    out[mask] = (x[mask] - mu[mask]) / sd[mask]
    out[~np.isfinite(out)] = 0.0
    return out


def predict_tophalf_from_score(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(score, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        pred[idx[np.argsort(v[idx], kind="mergesort")[::-1][:k]]] = 1
    return pred


def build_split_frame(split: str) -> pd.DataFrame:
    df = load_axis("e128", split)
    for axis in ("e64", "e192"):
        df = df.merge(load_axis(axis, split)[["ID", axis, f"std_{axis}"]], on="ID", how="inner", validate="one_to_one")
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True)
    for axis in ("e64", "e128", "e192"):
        df[f"z_{axis}"] = within_user_z(df, axis)
    return df


def rule_fires(rule: str, d64: np.ndarray, d128: np.ndarray, d192: np.ndarray, tau: float, guard: float) -> np.ndarray:
    if rule == "vote2":
        return ((d64 > tau).astype(int) + (d128 > tau).astype(int) + (d192 > tau).astype(int)) >= 2
    if rule == "vote192":
        return (d192 > tau) & (d128 > -guard)
    if rule == "vote64_192":
        return (d64 > tau) & (d192 > tau)
    if rule == "anti_anchor_192":
        # More aggressive: allow d128 to be slightly negative if both other capacities strongly prefer c.
        return (d192 > tau) & (d64 > tau) & (d128 > -guard)
    raise ValueError(rule)


def boundary_pairwise_pred(df: pd.DataFrame, *, band: int, rule: str, tau: float, guard: float, cap: int) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    z64 = df["z_e64"].to_numpy(dtype=float)
    z128 = df["z_e128"].to_numpy(dtype=float)
    z192 = df["z_e192"].to_numpy(dtype=float)
    base_score = df["e128"].to_numpy(dtype=float)

    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        n = len(idx)
        k = n // 2
        if k == 0:
            continue
        order = idx[np.argsort(base_score[idx], kind="mergesort")[::-1]]
        selected = set(order[:k].tolist())
        top_band = order[max(0, k - band):k]
        bottom_band = order[k:min(n, k + band)]
        if len(top_band) == 0 or len(bottom_band) == 0:
            pred[list(selected)] = 1
            continue

        # Matrix shape: bottom x top. A true cell means bottom candidate c should beat selected item a.
        d64 = z64[bottom_band][:, None] - z64[top_band][None, :]
        d128 = z128[bottom_band][:, None] - z128[top_band][None, :]
        d192 = z192[bottom_band][:, None] - z192[top_band][None, :]
        fire = rule_fires(rule, d64, d128, d192, tau=tau, guard=guard)
        bottom_wins = fire.sum(axis=1)
        top_losses = fire.sum(axis=0)

        promote_candidates = [i for i, wins in enumerate(bottom_wins) if int(wins) > 0]
        if not promote_candidates:
            pred[list(selected)] = 1
            continue
        promote_candidates.sort(key=lambda i: (int(bottom_wins[i]), z192[bottom_band[i]] + z64[bottom_band[i]], z128[bottom_band[i]], -int(bottom_band[i])), reverse=True)
        max_promote = min(cap, len(promote_candidates), len(top_band))
        promoted = [int(bottom_band[i]) for i in promote_candidates[:max_promote]]
        if not promoted:
            pred[list(selected)] = 1
            continue
        demote_candidates = list(range(len(top_band)))
        demote_candidates.sort(key=lambda j: (int(top_losses[j]), -(z192[top_band[j]] + z64[top_band[j]]), -z128[top_band[j]], int(top_band[j])), reverse=True)
        demoted = [int(top_band[j]) for j in demote_candidates[:len(promoted)]]
        for d in demoted:
            selected.discard(d)
        selected.update(promoted)
        if len(selected) != k:
            raise RuntimeError(f"Top-half count changed for user: {len(selected)} != {k}")
        pred[list(selected)] = 1
    return pred


def metric_from_pred(df: pd.DataFrame, pred: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    y = df["Label"].to_numpy(dtype=np.int8)
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
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    return v


def variant_grid() -> list[dict[str, Any]]:
    """Predeclared but bounded grid.

    The first unoptimized full grid was too slow because it re-ranked thousands of
    users for hundreds of variants. This reduced grid keeps the same structural
    levers (band width, threshold, rule, cap) while staying cheap enough for a
    last-slot no-submit probe.
    """
    rows: list[dict[str, Any]] = []
    bands = [4, 8, 16]
    taus = [0.0, 0.20]
    guards = [0.10]
    caps = [1, 2]
    for band in bands:
        for cap in caps:
            for tau in taus:
                rows.append({"rule": "vote2", "band": band, "tau": tau, "guard": 0.0, "cap": cap})
                rows.append({"rule": "vote64_192", "band": band, "tau": tau, "guard": 0.0, "cap": cap})
                for guard in guards:
                    rows.append({"rule": "vote192", "band": band, "tau": tau, "guard": guard, "cap": cap})
                    rows.append({"rule": "anti_anchor_192", "band": band, "tau": tau, "guard": guard, "cap": cap})
    return rows


def variant_name(v: dict[str, Any]) -> str:
    return f"{v['rule']}_B{v['band']}_tau{v['tau']:g}_guard{v['guard']:g}_cap{v['cap']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--report-json", default=str(REPORT_JSON_DEFAULT))
    ap.add_argument("--report-md", default=str(REPORT_MD_DEFAULT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = variant_grid()
    split_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        df = build_split_frame(split)
        base_pred = predict_tophalf_from_score(df, df["e128"].to_numpy(dtype=float))
        y = df["Label"].to_numpy(dtype=np.int8)
        base_acc = float((base_pred == y).mean())
        if abs(base_acc - BASE_EXPECTED[split]) > 1e-9:
            raise RuntimeError(f"{split}: base mismatch {base_acc:.12f} != {BASE_EXPECTED[split]:.12f}")
        base_rows.append({"split": split, "base_accuracy": base_acc})
        for v in variants:
            pred = boundary_pairwise_pred(df, band=int(v["band"]), rule=str(v["rule"]), tau=float(v["tau"]), guard=float(v["guard"]), cap=int(v["cap"]))
            # Hard per-user top-half preservation is enforced inside boundary_pairwise_pred
            # via the selected-size assert. Avoid a pandas groupby for every variant; it
            # dominated runtime in the first full-grid attempt.
            split_rows.append({"split": split, "variant": variant_name(v), **v, **metric_from_pred(df, pred, base_pred)})

    split_path = out_dir / "aggressive_boundary_pairwise_split_metrics.csv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "variant", "rule", "band", "tau", "guard", "cap", "accuracy", "delta", "fixes", "breaks", "discordant", "changed"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(split_rows)

    agg_rows: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in split_rows:
        by_variant.setdefault(str(r["variant"]), []).append(r)
    for name, rows in by_variant.items():
        fixes = int(sum(int(r["fixes"]) for r in rows))
        breaks = int(sum(int(r["breaks"]) for r in rows))
        deltas = [float(r["delta"]) for r in rows]
        first = rows[0]
        ratio = float("inf") if breaks == 0 and fixes > 0 else (fixes / breaks if breaks else 1.0)
        agg_rows.append({
            "variant": name,
            "rule": first["rule"],
            "band": first["band"],
            "tau": first["tau"],
            "guard": first["guard"],
            "cap": first["cap"],
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
    agg_path = out_dir / "aggressive_boundary_pairwise_aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant", "rule", "band", "tau", "guard", "cap", "mean_delta", "min_delta", "max_delta", "positive_splits", "fixes", "breaks", "fix_break_ratio", "discordant", "p_exact_two_sided", "split_deltas", "changed_total"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(agg_rows)

    strict_pass = [r for r in agg_rows if float(r["mean_delta"]) >= MDE and int(r["positive_splits"]) == 3 and float(r["p_exact_two_sided"]) <= 0.01 and float(r["fix_break_ratio"]) >= 1.20 and float(r["min_delta"]) >= 0.001]
    best = agg_rows[0]
    verdict = "STRICT_PASS" if strict_pass else ("MANUAL_RISK_SIGNAL" if float(best["mean_delta"]) > 0 and int(best["positive_splits"]) == 3 else "REJECT")
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "splits": list(SPLITS),
        "base": base_rows,
        "variant_count": len(agg_rows),
        "mde": MDE,
        "strict_gate": {
            "mean_delta_min": MDE,
            "positive_splits_required": "3/3",
            "p_exact_two_sided_max": 0.01,
            "fix_break_ratio_min": 1.20,
            "min_split_delta_min": 0.001,
            "strict_pass_count": len(strict_pass),
        },
        "verdict": verdict,
        "top_variants": clean(agg_rows[:40]),
        "output_files": {"split_metrics": str(split_path), "aggregate": str(agg_path)},
        "note": "Boundary-only pairwise agreement probe. Validation-only; no candidate/submission materialized.",
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Aggressive boundary pairwise agreement probe\n\n",
        f"- verdict: **{verdict}**\n",
        f"- variants: `{len(agg_rows)}`\n",
        f"- strict pass count: `{len(strict_pass)}`\n",
        "- safety: validation-only; no hidden/test read; no candidate CSV; no Kaggle submit.\n\n",
        "| rank | variant | mean Δ | min~max Δ | splits+ | fixes | breaks | ratio | p | changed |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for i, r in enumerate(agg_rows[:25], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {float(r['mean_delta']):+.6f} | {float(r['min_delta']):+.6f}~{float(r['max_delta']):+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {float(r['fix_break_ratio']):.3f} | {float(r['p_exact_two_sided']):.4g} | {r['changed_total']} |\n"
        )
    Path(args.report_md).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report_json": args.report_json, "best": clean(best)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
