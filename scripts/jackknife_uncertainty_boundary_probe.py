#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Validation-only jackknife uncertainty boundary probe.

Scans deterministic, label-free boundary rerank scores built from existing emb128/emb192
validation score artifacts.  It does not read hidden labels, does not touch real test pairs,
does not create a candidate/submission CSV, and does not call Kaggle.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from userknn_residual_probe import (  # noqa: E402
    SPLITS,
    SEEDS,
    emb128_paths,
    emb192_paths,
    eval_pred,
    score_col,
    top_half_pred,
    user_rank_high_is_good,
    z_within_user,
)


def load_members(split: str) -> pd.DataFrame:
    meta: pd.DataFrame | None = None
    score_parts: list[pd.DataFrame] = []
    cols: list[str] = []
    for family, paths in (("e128", emb128_paths(split)), ("e192", emb192_paths(split))):
        for seed, path in zip(SEEDS, paths):
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            sc = score_col(df)
            col = f"{family}_s{seed}"
            current_meta = df[["ID", "userID", "gameID", "Label"]].copy()
            if meta is None:
                meta = current_meta
            elif not meta[["ID", "userID", "gameID", "Label"]].equals(current_meta):
                raise RuntimeError(f"row mismatch: {path}")
            score_parts.append(df[["ID", sc]].rename(columns={sc: col}))
            cols.append(col)
    if meta is None:
        raise RuntimeError(f"no score members for split {split}")
    out = meta
    for part in score_parts:
        out = out.merge(part, on="ID", validate="one_to_one")
    out.attrs["member_cols"] = cols
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    e128 = [c for c in out.columns if c.startswith("e128_s")]
    e192 = [c for c in out.columns if c.startswith("e192_s")]
    all_cols = e128 + e192
    out["score_emb128"] = out[e128].mean(axis=1)
    out["score_emb192"] = out[e192].mean(axis=1)
    out["rank_emb128"] = user_rank_high_is_good(out, "score_emb128")
    out["rank_emb192"] = user_rank_high_is_good(out, "score_emb192")
    out["score_base_rankblend"] = out["rank_emb128"] + out["rank_emb192"]
    out["z_base"] = z_within_user(out, "score_base_rankblend")
    zcols = []
    for col in all_cols:
        zc = f"z_{col}"
        out[zc] = z_within_user(out, col)
        zcols.append(zc)
    z128 = [f"z_{c}" for c in e128]
    z192 = [f"z_{c}" for c in e192]
    out["z_mean_128"] = out[z128].mean(axis=1)
    out["z_mean_192"] = out[z192].mean(axis=1)
    out["z_mean_all"] = out[zcols].mean(axis=1)
    out["z_std_128"] = out[z128].std(axis=1).fillna(0.0)
    out["z_std_192"] = out[z192].std(axis=1).fillna(0.0)
    out["z_std_all"] = out[zcols].std(axis=1).fillna(0.0)
    out["capacity_gap_abs"] = (out["z_mean_128"] - out["z_mean_192"]).abs()
    for col in all_cols:
        pred = fast_tophalf(out, out[col].to_numpy(float))
        out[f"vote_{col}"] = pred.astype(float)
    vote_cols = [f"vote_{c}" for c in all_cols]
    out["vote_all_centered"] = out[vote_cols].sum(axis=1) - (len(vote_cols) / 2.0)
    out["vote_abs"] = out["vote_all_centered"].abs()
    return out


def boundary_mask(df: pd.DataFrame, band: int, gate: str) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    base = df["score_base_rankblend"].to_numpy(float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        n = len(idx)
        k = int(df.loc[idx, "Label"].sum()) if "Label" in df.columns else n // 2
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        lo = max(0, k - band)
        hi = min(n, k + band)
        mask[order[lo:hi]] = True
    if gate == "all_boundary":
        return mask
    if gate == "high_seed_std":
        thr = float(np.quantile(df["z_std_all"].to_numpy(float), 0.7))
        return mask & (df["z_std_all"].to_numpy(float) >= thr)
    if gate == "high_capacity_gap":
        thr = float(np.quantile(df["capacity_gap_abs"].to_numpy(float), 0.7))
        return mask & (df["capacity_gap_abs"].to_numpy(float) >= thr)
    if gate == "low_vote_abs":
        thr = float(np.quantile(df["vote_abs"].to_numpy(float), 0.4))
        return mask & (df["vote_abs"].to_numpy(float) <= thr)
    raise ValueError(gate)


def fast_tophalf(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    labels = df["Label"].to_numpy(dtype=np.int8) if "Label" in df.columns else None
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        k = int(labels[idx].sum()) if labels is not None else len(idx) // 2
        order = np.argsort(values[idx], kind="mergesort")[::-1]
        pred[idx[order[:k]]] = 1
    return pred


def candidate_score(df: pd.DataFrame, signal: str, weight: float, band: int, gate: str) -> np.ndarray:
    base = df["z_base"].to_numpy(float).copy()
    if signal == "lcb_all":
        residual = df["z_mean_all"].to_numpy(float) - df["z_std_all"].to_numpy(float)
    elif signal == "ucb_all":
        residual = df["z_mean_all"].to_numpy(float) + df["z_std_all"].to_numpy(float)
    elif signal == "std_demotion":
        residual = -df["z_std_all"].to_numpy(float)
    elif signal == "capacity_agree_128":
        residual = df["z_mean_128"].to_numpy(float) - df["capacity_gap_abs"].to_numpy(float)
    elif signal == "vote_consensus":
        residual = df["vote_all_centered"].to_numpy(float)
    else:
        raise ValueError(signal)
    mask = boundary_mask(df, band, gate)
    out = base.copy()
    out[mask] = base[mask] + weight * residual[mask]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json"))
    ap.add_argument("--md", default=str(ROOT / "reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.md"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary"))
    ap.add_argument("--weights", default="0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--bands", default="1,2,3,4")
    args = ap.parse_args()

    weights = [float(x) for x in args.weights.split(",") if x]
    bands = [int(x) for x in args.bands.split(",") if x]
    signals = ["lcb_all", "ucb_all", "std_demotion", "capacity_agree_128", "vote_consensus"]
    gates = ["all_boundary", "high_seed_std", "high_capacity_gap", "low_vote_abs"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    base_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        print(f"[split] {split}", flush=True)
        df = add_features(load_members(split))
        frames[split] = df
        y = df["Label"].astype(int).to_numpy()
        base_pred = fast_tophalf(df, df["score_base_rankblend"].to_numpy(float))
        base_ev = eval_pred(y, base_pred)
        base_rows.append({"split": split, "base_accuracy": base_ev["row_accuracy"], "base_correct": base_ev["correct"]})
        for signal in signals:
            for gate in gates:
                for band in bands:
                    for weight in weights:
                        variant = f"{signal}__{gate}__B{band}__w{weight:g}"
                        score = candidate_score(df, signal, weight, band, gate)
                        pred = fast_tophalf(df, score)
                        ev = eval_pred(y, pred, base_pred)
                        split_rows.append({
                            "split": split,
                            "variant": variant,
                            "signal": signal,
                            "gate": gate,
                            "band": band,
                            "weight": weight,
                            **ev,
                        })
        print(f"[split] {split} base={base_ev['row_accuracy']:.6f}; scanned={len(split_rows)}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in split_rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    agg: list[dict[str, Any]] = []
    for variant, rows in grouped.items():
        if len(rows) != len(SPLITS):
            continue
        fixes = sum(int(r["fixes"]) for r in rows)
        breaks = sum(int(r["breaks"]) for r in rows)
        disc = fixes + breaks
        p = float(binomtest(min(fixes, breaks), disc, 0.5, alternative="two-sided").pvalue) if disc else 1.0
        deltas = [float(r["delta_vs_base"]) for r in rows]
        first = rows[0]
        agg.append({
            "variant": variant,
            "mean_delta_vs_base": float(np.mean(deltas)),
            "min_delta_vs_base": float(np.min(deltas)),
            "max_delta_vs_base": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": disc,
            "pooled_p_exact": p,
            "signal": first["signal"],
            "gate": first["gate"],
            "band": first["band"],
            "weight": first["weight"],
            "split_deltas": {str(r["split"]): float(r["delta_vs_base"]) for r in rows},
        })
    agg.sort(key=lambda r: (r["mean_delta_vs_base"], r["positive_splits"], -r["pooled_p_exact"]), reverse=True)
    strict = [r for r in agg if r["mean_delta_vs_base"] >= 0.0015 and r["min_delta_vs_base"] >= 0 and r["positive_splits"] == 3 and r["fixes"] > r["breaks"] and r["pooled_p_exact"] < 0.05]
    verdict = "STRICT_PASS" if strict else ("WEAK_SIGNAL" if agg and agg[0]["mean_delta_vs_base"] > 0 else "REJECT")

    (out_dir / "split_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    with (out_dir / "split_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "variant", "signal", "gate", "band", "weight", "row_accuracy", "delta_vs_base", "fixes", "breaks", "discordant", "mcnemar_p_exact"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(split_rows)
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "mean_delta_vs_base", "min_delta_vs_base", "max_delta_vs_base", "positive_splits", "fixes", "breaks", "discordant", "pooled_p_exact", "signal", "gate", "band", "weight", "split_deltas"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(agg)

    top_variant = agg[0]["variant"] if agg else None
    score_artifacts: list[str] = []
    if top_variant:
        meta = next(r for r in agg if r["variant"] == top_variant)
        for split, df in frames.items():
            score = candidate_score(df, str(meta["signal"]), float(meta["weight"]), int(meta["band"]), str(meta["gate"]))
            out = df[["ID", "userID", "gameID", "Label", "score_base_rankblend"]].copy()
            out["score_jackknife_uncertainty"] = score
            out_path = out_dir / f"{split}__top_variant_scores.csv"
            out.to_csv(out_path, index=False)
            score_artifacts.append(str(out_path.resolve().relative_to(ROOT)))

    payload = {
        "safety": {
            "validation_only": True,
            "candidate_csv_written": False,
            "kaggle_submit_executed": False,
            "hidden_labels_used": False,
            "external_scraping_used": False,
        },
        "metadata": {"splits": SPLITS, "weights": weights, "bands": bands, "signals": signals, "gates": gates},
        "base": base_rows,
        "strict_pass_count": len(strict),
        "verdict": verdict,
        "top_variant_score_artifacts": score_artifacts,
        "top_variants": agg[:30],
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Jackknife uncertainty boundary probe",
        "",
        "Safety: validation-only; no Kaggle submit; no candidate/submission CSV; no hidden labels; no external scraping.",
        "",
        f"- verdict: **{verdict}**",
        f"- strict pass count: `{len(strict)}`",
        f"- top row-level score artifacts: `{len(score_artifacts)}` files",
        "",
        "| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(agg[:30], 1):
        lines.append(f"| {i} | `{r['variant']}` | {r['mean_delta_vs_base']:+.6f} | {r['min_delta_vs_base']:+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {r['pooled_p_exact']:.4g} |")
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": args.json, "md": args.md, "verdict": verdict, "strict_pass_count": len(strict), "top": agg[:5]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
