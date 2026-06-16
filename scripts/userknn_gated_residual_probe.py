#!/usr/bin/env python3
"""Validation-only user-gated UserKNN residual probe.

Follow-up to userknn_residual_probe.py.  The global UserKNN-max residual had a
small positive mean but failed one uniform split.  This probe checks whether the
same signal can be restricted to unsupervised user groups / decision-boundary
bands.  It never reads hidden labels, never writes a test candidate, and never
calls Kaggle.
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
    add_userknn_scores,
    emb128_paths,
    emb192_paths,
    eval_pred,
    load_ensemble,
    top_half_pred,
    user_rank_high_is_good,
    z_within_user,
)


def build_split_frame(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    a = load_ensemble(emb128_paths(split), "score_emb128")
    b = load_ensemble(emb192_paths(split), "score_emb192")[["ID", "score_emb192"]]
    df = a.merge(b, on="ID", validate="one_to_one")
    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_base_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    df = add_userknn_scores(split, df)
    y = df["Label"].astype(int).to_numpy()
    base_pred = top_half_pred(df, "score_base_rankblend")
    return df, y, base_pred


def user_features(split: str, df: pd.DataFrame) -> pd.DataFrame:
    train = pd.read_csv(ROOT / "artifacts/validation" / split / "train_interactions.csv")
    train_deg = train.groupby("userID").size().astype(float).to_dict()
    rows: list[dict[str, Any]] = []
    base = df["score_base_rankblend"].to_numpy(float)
    aux_cols = ["score_userknn_max", "score_userknn_top10", "score_userknn_popnorm", "score_userknn_sum"]
    for user, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        n = len(idx)
        k = n // 2
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        margin = 0.0 if k <= 0 or k >= n else float(base[order[k - 1]] - base[order[k]])
        rec: dict[str, Any] = {"userID": user, "n": n, "margin": margin, "train_deg": float(train_deg.get(user, 0.0))}
        for col in aux_cols:
            vals = df.loc[idx, col].to_numpy(float)
            rec[f"{col}_mean"] = float(np.mean(vals))
            rec[f"{col}_std"] = float(np.std(vals))
            # How much this aux would change top-half if used alone; unsupervised.
            aux_order = idx[np.argsort(vals, kind="mergesort")[::-1]]
            rec[f"{col}_base_changed"] = len(set(order[:k]).symmetric_difference(set(aux_order[:k])))
        rows.append(rec)
    return pd.DataFrame(rows)


def masks_from_features(feat: pd.DataFrame) -> dict[str, set[Any]]:
    masks: dict[str, set[Any]] = {"all_users": set(feat["userID"].tolist())}
    cols = [
        "n",
        "margin",
        "train_deg",
        "score_userknn_max_mean",
        "score_userknn_max_std",
        "score_userknn_max_base_changed",
        "score_userknn_top10_base_changed",
        "score_userknn_popnorm_base_changed",
    ]
    for col in cols:
        vals = feat[col].to_numpy(float)
        for q in [0.2, 0.4, 0.6, 0.8]:
            thr = float(np.quantile(vals, q))
            masks[f"{col}_low_q{q:g}"] = set(feat.loc[feat[col] <= thr, "userID"].tolist())
            masks[f"{col}_high_q{q:g}"] = set(feat.loc[feat[col] >= thr, "userID"].tolist())
    for q in [0.4, 0.6, 0.8]:
        margin_thr = float(np.quantile(feat["margin"].to_numpy(float), q))
        changed_thr = float(np.quantile(feat["score_userknn_max_base_changed"].to_numpy(float), 1.0 - q / 2.0))
        deg_thr = float(np.quantile(feat["train_deg"].to_numpy(float), q))
        masks[f"margin_low_q{q:g}_and_maxchanged_high"] = set(feat.loc[(feat["margin"] <= margin_thr) & (feat["score_userknn_max_base_changed"] >= changed_thr), "userID"].tolist())
        masks[f"margin_low_q{q:g}_and_traindeg_low"] = set(feat.loc[(feat["margin"] <= margin_thr) & (feat["train_deg"] <= deg_thr), "userID"].tolist())
        masks[f"maxchanged_high_or_traindeg_low_q{q:g}"] = set(feat.loc[(feat["score_userknn_max_base_changed"] >= changed_thr) | (feat["train_deg"] <= deg_thr), "userID"].tolist())
    return {k: v for k, v in masks.items() if v}


def score_for(df: pd.DataFrame, users: set[Any], aux_col: str, weight: float, band: int | None) -> np.ndarray:
    z_base = z_within_user(df, "score_base_rankblend")
    z_aux = z_within_user(df, aux_col)
    out = z_base.copy()
    selected = df["userID"].isin(users).to_numpy()
    if band is None:
        out[selected] = z_base[selected] + weight * z_aux[selected]
        return out
    # Apply only to base boundary positions within selected users.
    base = df["score_base_rankblend"].to_numpy(float)
    for user, idx_raw in df.groupby("userID", sort=False).indices.items():
        if user not in users:
            continue
        idx = np.asarray(idx_raw)
        n = len(idx); k = n // 2
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        lo = max(0, k - band); hi = min(n, k + band)
        bidx = order[lo:hi]
        out[bidx] = z_base[bidx] + weight * z_aux[bidx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports/userknn_gated_residual_probe.json"))
    ap.add_argument("--md", default=str(ROOT / "reports/userknn_gated_residual_probe.md"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/userknn_gated_residual_probe"))
    ap.add_argument("--weights", default="0.1,0.2,0.25,0.3,0.5")
    ap.add_argument("--bands", default="1,2,3")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x]
    bands = [int(x) for x in args.bands.split(",") if x]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        print(f"[split] {split}", flush=True)
        df, y, base_pred = build_split_frame(split)
        base_ev = eval_pred(y, base_pred)
        base_rows.append({"split": split, "base_accuracy": base_ev["row_accuracy"], "base_correct": base_ev["correct"]})
        feat = user_features(split, df)
        masks = masks_from_features(feat)
        aux_cols = ["score_userknn_max", "score_userknn_top10", "score_userknn_popnorm", "score_userknn_sum"]
        for mask_name, users in masks.items():
            for aux_col in aux_cols:
                for weight in weights:
                    for band in [None, *bands]:
                        variant = f"{aux_col}_w{weight:g}__{mask_name}" + ("" if band is None else f"__B{band}")
                        score = score_for(df, users, aux_col, weight, band)
                        pred = top_half_pred(df.assign(_score=score), "_score")
                        ev = eval_pred(y, pred, base_pred)
                        split_rows.append({
                            "split": split,
                            "variant": variant,
                            "aux_col": aux_col,
                            "weight": weight,
                            "band": band,
                            "mask": mask_name,
                            "selected_users": len(users),
                            **ev,
                        })
        print(f"[split] {split} base={base_ev['row_accuracy']:.6f}; rows={len(split_rows)}", flush=True)

    # Aggregate.
    by: dict[str, list[dict[str, Any]]] = {}
    for r in split_rows:
        by.setdefault(str(r["variant"]), []).append(r)
    agg: list[dict[str, Any]] = []
    for variant, rows in by.items():
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
            "aux_col": first["aux_col"],
            "weight": first["weight"],
            "band": first["band"],
            "mask": first["mask"],
            "mean_selected_users": float(np.mean([int(r["selected_users"]) for r in rows])),
            "split_deltas": {str(r["split"]): float(r["delta_vs_base"]) for r in rows},
        })
    agg.sort(key=lambda r: (r["mean_delta_vs_base"], r["positive_splits"], -r["pooled_p_exact"]), reverse=True)
    strict = [r for r in agg if r["mean_delta_vs_base"] >= 0.0015 and r["min_delta_vs_base"] >= 0 and r["positive_splits"] == 3 and r["fixes"] > r["breaks"] and r["pooled_p_exact"] < 0.05]
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "metadata": {"splits": SPLITS, "weights": weights, "bands": bands, "probe": "user-gated user-user KNN residual"},
        "base": base_rows,
        "strict_pass_count": len(strict),
        "verdict": "STRICT_PASS" if strict else ("WEAK_SIGNAL" if agg and agg[0]["mean_delta_vs_base"] > 0 else "REJECT"),
        "top_variants": agg[:50],
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # CSV artifacts for inspection.
    with (out_dir / "split_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "variant", "aux_col", "weight", "band", "mask", "selected_users", "row_accuracy", "delta_vs_base", "fixes", "breaks", "discordant", "mcnemar_p_exact"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(split_rows)
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "mean_delta_vs_base", "min_delta_vs_base", "max_delta_vs_base", "positive_splits", "fixes", "breaks", "discordant", "pooled_p_exact", "aux_col", "weight", "band", "mask", "mean_selected_users", "split_deltas"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(agg)
    lines = ["# User-gated UserKNN residual probe", "", "No Kaggle submission is performed.", "", f"- verdict: **{payload['verdict']}**", f"- strict pass count: `{len(strict)}`", "", "| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(agg[:30], 1):
        lines.append(f"| {i} | `{r['variant']}` | {r['mean_delta_vs_base']:+.6f} | {r['min_delta_vs_base']:+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {r['pooled_p_exact']:.4g} |")
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": args.json, "md": args.md, "verdict": payload["verdict"], "strict_pass_count": len(strict), "top": agg[:10]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
