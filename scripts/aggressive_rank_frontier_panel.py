#!/usr/bin/env python3
"""3-split aggressive rank/z frontier panel for KMURecSys26 Steam.

This is the multi-split follow-up to `aggressive_rank_frontier_scan.py`.
It evaluates the same fixed finite grid over emb64/emb128/emb192 4-seed ensembles on
three calibrated uniform validation splits. It is still validation-only and does not
read the real test pairs or write candidate/submission CSVs.
"""
from __future__ import annotations

import argparse
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
OUT_DIR_DEFAULT = ROOT / "artifacts/aggressive_rank_frontier_panel"
REPORT_JSON_DEFAULT = ROOT / "reports/20260602_aggressive_rank_frontier_panel.json"
REPORT_MD_DEFAULT = ROOT / "reports/20260602_aggressive_rank_frontier_panel.md"


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
            raise FileNotFoundError(f"Missing {axis} split={split} seed={seed}: {p}")
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
    return merged[["ID", "userID", "gameID", "Label", axis]]


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    m = sd > 1e-12
    out[m] = (x[m] - mu[m]) / sd[m]
    out[~np.isfinite(out)] = 0.0
    return out


def within_user_rank_high(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=float)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        ranks[idx[np.argsort(values[idx], kind="mergesort")]] = np.arange(len(idx), dtype=float)
    return ranks


def predict_tophalf(df: pd.DataFrame, score_values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(score_values, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        pred[idx[np.argsort(v[idx], kind="mergesort")[::-1][:k]]] = 1
    return pred


def metric(df: pd.DataFrame, score: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    pred = predict_tophalf(df, score)
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


def boundary_score(base: np.ndarray, variant: np.ndarray, df: pd.DataFrame, width_frac: float, cap: int) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    base = np.asarray(base, dtype=float)
    variant = np.asarray(variant, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        n = len(idx); h = n // 2
        w = min(cap, max(1, int(math.ceil(width_frac * h))))
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        ranks = np.empty(n, dtype=int)
        ranks[np.argsort(base[idx], kind="mergesort")[::-1]] = np.arange(n)
        local = -ranks.astype(float) * 1000.0
        lo = max(0, h - w); hi = min(n, h + w)
        band = order[lo:hi]
        if len(band):
            vv = variant[band]
            vz = (vv - vv.mean()) / vv.std(ddof=0) if vv.std(ddof=0) > 1e-12 else np.zeros(len(band))
            pos = {rid: j for j, rid in enumerate(idx)}
            for rid, val in zip(band, vz, strict=True):
                local[pos[rid]] = -h * 1000.0 + val
        out[idx] = local
    return out


def build_split_frame(split: str) -> pd.DataFrame:
    df = load_axis("e128", split)
    for axis in ("e64", "e192"):
        df = df.merge(load_axis(axis, split)[["ID", axis]], on="ID", how="inner", validate="one_to_one")
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True)
    for axis in ("e64", "e128", "e192"):
        df[f"z_{axis}"] = within_user_z(df, axis)
        df[f"r_{axis}"] = within_user_rank_high(df, axis)
    return df


def generate_variant_scores(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, str, dict[str, Any]]]:
    out: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}
    wg = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    for w192 in wg:
        for w64 in wg:
            if w192 == 0 and w64 == 0:
                continue
            out[f"rank_w128_1_w192_{w192:g}_w64_{w64:g}"] = (
                df["r_e128"].to_numpy() + w192 * df["r_e192"].to_numpy() + w64 * df["r_e64"].to_numpy(),
                "weighted_rank_sum",
                {"w128": 1.0, "w192": w192, "w64": w64},
            )
            out[f"z_w128_1_w192_{w192:g}_w64_{w64:g}"] = (
                df["z_e128"].to_numpy() + w192 * df["z_e192"].to_numpy() + w64 * df["z_e64"].to_numpy(),
                "weighted_z_sum",
                {"w128": 1.0, "w192": w192, "w64": w64},
            )
    n = df.groupby("userID", sort=False)["ID"].transform("size").to_numpy(dtype=float)
    low128 = (n - 1.0) - df["r_e128"].to_numpy()
    low192 = (n - 1.0) - df["r_e192"].to_numpy()
    low64 = (n - 1.0) - df["r_e64"].to_numpy()
    for k in (5, 10, 20, 50):
        out[f"rrf_128_192_k{k}"] = (1/(k+low128)+1/(k+low192), "rrf", {"axes": "128+192", "k": k})
        out[f"rrf_128_192_64_k{k}"] = (1/(k+low128)+1/(k+low192)+1/(k+low64), "rrf", {"axes": "128+192+64", "k": k})
    inner = {
        "rank_128_192": df["r_e128"].to_numpy() + df["r_e192"].to_numpy(),
        "rank_128_192_64": df["r_e128"].to_numpy() + df["r_e192"].to_numpy() + df["r_e64"].to_numpy(),
        "z_128_192": df["z_e128"].to_numpy() + df["z_e192"].to_numpy(),
        "z_128_192_64": df["z_e128"].to_numpy() + df["z_e192"].to_numpy() + df["z_e64"].to_numpy(),
    }
    for name, score in inner.items():
        for frac in (0.05, 0.10, 0.20):
            for cap in (5, 10, 20):
                out[f"boundary_{name}_frac{frac:g}_cap{cap}"] = (
                    boundary_score(df["e128"].to_numpy(dtype=float), score, df, frac, cap),
                    "boundary_only",
                    {"inner_score": name, "width_frac": frac, "cap": cap},
                )
    return out


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--report-json", default=str(REPORT_JSON_DEFAULT))
    ap.add_argument("--report-md", default=str(REPORT_MD_DEFAULT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    base_rows = []
    for split in SPLITS:
        df = build_split_frame(split)
        base_pred = predict_tophalf(df, df["e128"].to_numpy(dtype=float))
        y = df["Label"].to_numpy(dtype=np.int8)
        base_acc = float((base_pred == y).mean())
        if abs(base_acc - BASE_EXPECTED[split]) > 1e-9:
            raise RuntimeError(f"{split}: base acc mismatch {base_acc:.12f} != {BASE_EXPECTED[split]:.12f}")
        base_rows.append({"split": split, "base_acc": base_acc})
        for variant, (score, family, params) in generate_variant_scores(df).items():
            m = metric(df, score, base_pred)
            all_rows.append({"split": split, "variant": variant, "family": family, **params, **m})

    raw = pd.DataFrame(all_rows)
    raw.to_csv(out_dir / "aggressive_rank_frontier_panel_split_metrics.csv", index=False)
    agg_rows = []
    for variant, g in raw.groupby("variant", sort=False):
        fixes = int(g["fixes"].sum()); breaks = int(g["breaks"].sum())
        deltas = g["delta"].astype(float).to_numpy()
        first = g.iloc[0].to_dict()
        agg_rows.append({
            "variant": variant,
            "family": first.get("family"),
            "mean_delta": float(deltas.mean()),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "positive_splits": int((deltas > 0).sum()),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": fixes + breaks,
            "p_exact_two_sided": exact_two_sided_binom_p(fixes, fixes + breaks),
            "split_deltas": [float(x) for x in deltas],
            "changed_total": int(g["changed"].sum()),
        })
    agg = pd.DataFrame(agg_rows).sort_values(["mean_delta", "fixes"], ascending=[False, False], kind="mergesort")
    agg.to_csv(out_dir / "aggressive_rank_frontier_panel_aggregate.csv", index=False)

    pass_mask = (agg["mean_delta"] >= MDE) & (agg["fixes"] > agg["breaks"]) & (agg["p_exact_two_sided"] < 0.05) & (agg["positive_splits"] >= 2)
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "splits": list(SPLITS),
        "base": base_rows,
        "variant_count": int(agg.shape[0]),
        "mde": MDE,
        "strict_gate_pass_count": int(pass_mask.sum()),
        "top_variants": clean(agg.head(40).to_dict(orient="records")),
        "output_files": {
            "split_metrics": str(out_dir / "aggressive_rank_frontier_panel_split_metrics.csv"),
            "aggregate": str(out_dir / "aggressive_rank_frontier_panel_aggregate.csv"),
        },
        "verdict": "CANDIDATE_PANEL_PASS" if int(pass_mask.sum()) else "NO_STRICT_PASS_AGGRESSIVE_PANEL",
        "note": "Aggressive fixed-grid panel. If no strict pass, any later materialization is manual-risk only.",
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Aggressive rank/z frontier — 3-split panel\n\n",
        f"- verdict: **{payload['verdict']}**\n",
        f"- variants: `{payload['variant_count']}`\n",
        f"- strict gate pass count: `{payload['strict_gate_pass_count']}`\n",
        "- safety: validation-only; no hidden/test read; no candidate CSV; no Kaggle submit.\n\n",
        "| rank | variant | family | mean Δ | min~max Δ | splits+ | fixes | breaks | p |\n",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for i, r in enumerate(payload["top_variants"][:25], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['family']} | {r['mean_delta']:+.6f} | {r['min_delta']:+.6f}~{r['max_delta']:+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {r['p_exact_two_sided']:.4g} |\n"
        )
    Path(args.report_md).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "report_json": args.report_json, "best": payload["top_variants"][0]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
