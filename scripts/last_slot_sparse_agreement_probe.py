#!/usr/bin/env python3
"""Validation-only sparse agreement probe for the final KMURecSys26 Steam slot.

This is a no-submit last-slot diagnostic. It tests whether several weak-but
partly independent residual signals agree on the *same boundary swaps* against
current best rankblend (emb128+emb192). It never writes a Kaggle candidate and
never calls Kaggle.
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
    emb128_paths,
    emb192_paths,
    eval_pred,
    load_ensemble,
    top_half_pred,
    user_rank_high_is_good,
    z_within_user,
)

SEM_MODELS = [
    ("bge", ROOT / "artifacts/semantic_residual_probe/BAAI_bge-m3"),
    ("qwen", ROOT / "artifacts/semantic_residual_probe/Qwen_Qwen3-Embedding-0.6B"),
    ("nomic", ROOT / "artifacts/semantic_residual_probe/nomic-ai_modernbert-embed-base"),
]
README_SCORE_COLS = [
    "z_score_als_htr_f32_it30_alpha20_popa4",
    "z_score_als_f32_it30_alpha20_popa4",
    "z_score_bpr_htr_f32_it100_popa4",
    "z_score_bpr_f32_it100_popa4",
]


def build_split_frame_fast(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build current-best validation frame without recomputing UserKNN."""
    a = load_ensemble(emb128_paths(split), "score_emb128")
    b = load_ensemble(emb192_paths(split), "score_emb192")[["ID", "score_emb192"]]
    df = a.merge(b, on="ID", validate="one_to_one")
    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_base_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    y = df["Label"].astype(int).to_numpy()
    base_pred = top_half_pred(df, "score_base_rankblend")
    return df, y, base_pred


def safe_float_array(s: pd.Series) -> np.ndarray:
    out = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float).copy()
    out[~np.isfinite(out)] = 0.0
    return out


def base_rank_desc_and_boundary(df: pd.DataFrame, base_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank_desc = np.zeros(len(df), dtype=float)
    boundary_dist = np.zeros(len(df), dtype=float)
    k_by_row = np.zeros(len(df), dtype=float)
    vals = df[base_col].to_numpy(dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        k = int(df.loc[idx, "Label"].sum())
        order = idx[np.argsort(vals[idx], kind="mergesort")[::-1]]
        pos = np.arange(1, len(idx) + 1, dtype=float)
        rank_desc[order] = pos
        boundary_dist[order] = np.abs(pos - (k + 0.5))
        k_by_row[idx] = k
    return rank_desc, boundary_dist, k_by_row


def precompute_groups(df: pd.DataFrame) -> tuple[list[np.ndarray], list[int]]:
    groups: list[np.ndarray] = []
    ks: list[int] = []
    labels = df["Label"].to_numpy(dtype=int)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        groups.append(idx)
        ks.append(int(labels[idx].sum()))
    return groups, ks


def top_half_pred_fast(values: np.ndarray, groups: list[np.ndarray], ks: list[int]) -> np.ndarray:
    pred = np.zeros(len(values), dtype=np.int8)
    v = np.asarray(values, dtype=float)
    for idx, k in zip(groups, ks):
        if k <= 0:
            continue
        order = np.argsort(v[idx], kind="mergesort")[::-1]
        pred[idx[order[:k]]] = 1
    return pred


def add_semantic_scores(split: str, df: pd.DataFrame) -> list[str]:
    aux_cols: list[str] = []
    for short, root in SEM_MODELS:
        p = root / split / "validation_semantic_scores.csv"
        if not p.exists():
            continue
        sem = pd.read_csv(p)
        keep = ["ID"] + [c for c in ["z_sem_bin_resid_base_pop", "z_sem_htr_resid_base_pop", "z_sem_bin", "z_sem_htr"] if c in sem.columns]
        sem = sem[keep].copy()
        rename: dict[str, str] = {}
        for c in keep:
            if c == "ID":
                continue
            rename[c] = f"aux_sem_{short}_{c.replace('z_', '')}"
        sem = sem.rename(columns=rename)
        before = len(df)
        df2 = df.merge(sem, on="ID", how="left", validate="one_to_one")
        if len(df2) != before:
            raise RuntimeError(f"semantic merge changed rows for {split} {short}")
        for c in rename.values():
            df[c] = safe_float_array(df2[c])
            aux_cols.append(c)
    return aux_cols


def add_readme_scores(split: str, df: pd.DataFrame) -> list[str]:
    p = ROOT / "artifacts/scores" / f"{split}_readme_bprals" / "candidate_scores.csv"
    if not p.exists():
        return []
    sc = pd.read_csv(p, usecols=lambda c: c == "ID" or c in README_SCORE_COLS)
    rename = {c: f"aux_readme_{c[2:] if c.startswith('z_') else c}" for c in sc.columns if c != "ID"}
    sc = sc.rename(columns=rename)
    before = len(df)
    df2 = df.merge(sc, on="ID", how="left", validate="one_to_one")
    if len(df2) != before:
        raise RuntimeError(f"readme score merge changed rows for {split}")
    cols = []
    for c in rename.values():
        df[c] = safe_float_array(df2[c])
        cols.append(c)
    return cols


def add_auxiliary_scores(split: str, df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    sem_cols = add_semantic_scores(split, df)
    # Keep residualized semantic columns first; raw semantic is lower priority.
    groups["semantic_resid"] = [c for c in sem_cols if "resid_base_pop" in c]
    groups["semantic_raw"] = [c for c in sem_cols if c not in groups["semantic_resid"]]
    readme_cols = add_readme_scores(split, df)
    groups["readme_bprals"] = readme_cols
    return {k: v for k, v in groups.items() if v}


def build_user_masks(df: pd.DataFrame, split: str, boundary_dist: np.ndarray) -> dict[str, set[Any]]:
    """Build a small predeclared mask set from cheap unsupervised features."""
    train = pd.read_csv(ROOT / "artifacts/validation" / split / "train_interactions.csv")
    train_deg = train.groupby("userID").size().astype(float).to_dict()
    tmp = pd.DataFrame({
        "userID": df["userID"].to_numpy(),
        "bd": boundary_dist,
        "train_deg": df["userID"].map(train_deg).fillna(0.0).to_numpy(dtype=float),
    })
    feat = tmp.groupby("userID", sort=False).agg(n=("userID", "size"), bd=("bd", "mean"), train_deg=("train_deg", "mean")).reset_index()
    masks: dict[str, set[Any]] = {"all_users": set(feat["userID"].tolist())}
    for col, qs in {"n": [0.8], "train_deg": [0.2, 0.4], "bd": [0.4]}.items():
        vals = feat[col].to_numpy(float)
        for q in qs:
            thr = float(np.quantile(vals, q))
            masks[f"{col}_low_q{q:g}"] = set(feat.loc[feat[col] <= thr, "userID"].tolist())
            masks[f"{col}_high_q{q:g}"] = set(feat.loc[feat[col] >= thr, "userID"].tolist())
    keep = ["all_users", "n_high_q0.8", "train_deg_high_q0.2", "train_deg_high_q0.4", "bd_low_q0.4"]
    return {k: masks[k] for k in keep if k in masks and masks[k]}


def group_combos(groups: dict[str, list[str]], max_cols_per_group: int = 3) -> dict[str, list[str]]:
    # Bounded no-submit combinations of cached semantic and README/BPR-ALS signals.
    def first(name: str, n: int) -> list[str]:
        return groups.get(name, [])[:n]
    combos: dict[str, list[str]] = {}
    if first("semantic_resid", 3):
        combos["semantic_resid3"] = first("semantic_resid", 3)
    if first("semantic_resid", 3) and first("readme_bprals", 2):
        combos["semresid3_readme2"] = first("semantic_resid", 3) + first("readme_bprals", 2)
    if first("semantic_raw", 2) and first("readme_bprals", 2):
        combos["semraw2_readme2"] = first("semantic_raw", 2) + first("readme_bprals", 2)
    if first("semantic_resid", 3) and first("semantic_raw", 2):
        combos["semresid3_semraw2"] = first("semantic_resid", 3) + first("semantic_raw", 2)
    # Drop duplicate column lists.
    dedup: dict[tuple[str, ...], str] = {}
    out: dict[str, list[str]] = {}
    for name, cols in combos.items():
        cols = list(dict.fromkeys(cols))[:max_cols_per_group * 3]
        key = tuple(cols)
        if key and key not in dedup:
            dedup[key] = name
            out[name] = cols
    return out


def evaluate_split(split: str, weights: list[float], bands: list[int], min_votes_list: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df, y, base_pred = build_split_frame_fast(split)
    groups_idx, ks = precompute_groups(df)
    base_ev = eval_pred(y, base_pred)
    rank_desc, boundary_dist, _ = base_rank_desc_and_boundary(df, "score_base_rankblend")
    df["_base_z"] = z_within_user(df, "score_base_rankblend")
    df["_base_pred"] = base_pred
    groups = add_auxiliary_scores(split, df)
    combos = group_combos(groups)
    masks = build_user_masks(df, split, boundary_dist)
    rows: list[dict[str, Any]] = []
    base_z_values = df["_base_z"].to_numpy(float)
    for combo_name, cols in combos.items():
        aux_preds = np.vstack([top_half_pred_fast(df[c].to_numpy(float), groups_idx, ks) for c in cols])
        # Directional vote: + if aux wants to promote base-negative row, - if aux wants to demote base-positive row.
        vote_delta = ((aux_preds == 1) & (base_pred[None, :] == 0)).sum(axis=0) - ((aux_preds == 0) & (base_pred[None, :] == 1)).sum(axis=0)
        agreement = np.abs(vote_delta)
        mean_aux = np.mean([df[c].to_numpy(dtype=float) for c in cols], axis=0)
        mean_aux_z = mean_aux.copy()
        mean_aux_z[~np.isfinite(mean_aux_z)] = 0.0
        for min_votes in min_votes_list:
            if min_votes > len(cols):
                continue
            vote_gate = agreement >= min_votes
            for band in bands:
                band_gate = boundary_dist <= band
                for mask_name, users in masks.items():
                    user_gate = df["userID"].isin(users).to_numpy()
                    gate = vote_gate & band_gate & user_gate
                    if gate.sum() == 0:
                        continue
                    for weight in weights:
                        # Two variants: vote-only and vote plus small continuous tie-breaker.
                        for mode, score in [
                            ("vote", base_z_values + weight * vote_delta * gate),
                            ("vote_zmean", base_z_values + weight * (vote_delta + 0.25 * mean_aux_z) * gate),
                        ]:
                            pred = top_half_pred_fast(score, groups_idx, ks)
                            ev = eval_pred(y, pred, base_pred)
                            rows.append({
                                "split": split,
                                "variant": f"{combo_name}__{mode}__mv{min_votes}__B{band}__{mask_name}__w{weight:g}",
                                "combo": combo_name,
                                "mode": mode,
                                "cols": ";".join(cols),
                                "weight": weight,
                                "band": band,
                                "min_votes": min_votes,
                                "mask": mask_name,
                                "gate_rows": int(gate.sum()),
                                "gate_users": int(pd.Series(df.loc[gate, "userID"]).nunique()) if gate.any() else 0,
                                **ev,
                            })
    meta = {
        "split": split,
        "base": base_ev,
        "aux_groups": {k: v for k, v in groups.items()},
        "combo_count": len(combos),
        "mask_count": len(masks),
    }
    return meta, rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["variant"]), []).append(r)
    out: list[dict[str, Any]] = []
    for variant, rs in by.items():
        if len(rs) != len(SPLITS):
            continue
        fixes = sum(int(r["fixes"]) for r in rs)
        breaks = sum(int(r["breaks"]) for r in rs)
        disc = fixes + breaks
        p = float(binomtest(min(fixes, breaks), disc, 0.5, alternative="two-sided").pvalue) if disc else 1.0
        deltas = [float(r["delta_vs_base"]) for r in rs]
        gate_users = [int(r.get("gate_users", 0)) for r in rs]
        gate_rows = [int(r.get("gate_rows", 0)) for r in rs]
        first = rs[0]
        out.append({
            "variant": variant,
            "mean_delta_vs_base": float(np.mean(deltas)),
            "min_delta_vs_base": float(np.min(deltas)),
            "max_delta_vs_base": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": disc,
            "pooled_p_exact": p,
            "combo": first.get("combo"),
            "mode": first.get("mode"),
            "weight": first.get("weight"),
            "band": first.get("band"),
            "min_votes": first.get("min_votes"),
            "mask": first.get("mask"),
            "mean_gate_users": float(np.mean(gate_users)),
            "mean_gate_rows": float(np.mean(gate_rows)),
            "cols": first.get("cols"),
            "split_deltas": {str(r["split"]): float(r["delta_vs_base"]) for r in rs},
        })
    out.sort(key=lambda r: (r["mean_delta_vs_base"], r["min_delta_vs_base"], r["fixes"] - r["breaks"], -r["pooled_p_exact"]), reverse=True)
    return out


def write_outputs(args: argparse.Namespace, payload: dict[str, Any], split_rows: list[dict[str, Any]], agg: list[dict[str, Any]]) -> None:
    jp = Path(args.json); mp = Path(args.md); out_dir = Path(args.out_dir)
    jp.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    split_fields = ["split", "variant", "combo", "mode", "weight", "band", "min_votes", "mask", "gate_rows", "gate_users", "row_accuracy", "delta_vs_base", "fixes", "breaks", "discordant", "mcnemar_p_exact", "cols"]
    with (out_dir / "split_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=split_fields, extrasaction="ignore"); w.writeheader(); w.writerows(split_rows)
    agg_fields = ["variant", "mean_delta_vs_base", "min_delta_vs_base", "max_delta_vs_base", "positive_splits", "fixes", "breaks", "discordant", "pooled_p_exact", "combo", "mode", "weight", "band", "min_votes", "mask", "mean_gate_users", "mean_gate_rows", "split_deltas", "cols"]
    with (out_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore"); w.writeheader(); w.writerows(agg)
    lines = [
        "# Last-slot sparse agreement probe",
        "",
        "No Kaggle submission is performed. No candidate CSV is written.",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- strict pass count: `{payload['strict_pass_count']}`",
        f"- manual-risk count: `{payload['manual_risk_count']}`",
        "",
        "Strict threshold: mean Δ >= +0.0020, min Δ >= +0.0008, 3/3 positive, pooled p < 0.01, fixes-breaks >= 120.",
        "Manual-risk threshold: mean Δ >= +0.0015, min Δ >= +0.0005, 3/3 positive, pooled p < 0.03.",
        "",
        "| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p | gate users |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(agg[:30], 1):
        lines.append(f"| {i} | `{r['variant']}` | {r['mean_delta_vs_base']:+.6f} | {r['min_delta_vs_base']:+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {r['pooled_p_exact']:.4g} | {r['mean_gate_users']:.0f} |")
    mp.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports/last_slot_sparse_agreement_probe.json"))
    ap.add_argument("--md", default=str(ROOT / "reports/last_slot_sparse_agreement_probe.md"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/last_slot_sparse_agreement_probe"))
    ap.add_argument("--weights", default="0.05,0.1,0.2")
    ap.add_argument("--bands", default="1,2")
    ap.add_argument("--min-votes", default="2,3")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x]
    bands = [int(x) for x in args.bands.split(",") if x]
    min_votes = [int(x) for x in args.min_votes.split(",") if x]
    split_rows: list[dict[str, Any]] = []
    split_meta: list[dict[str, Any]] = []
    for split in SPLITS:
        print(f"[split] {split}", flush=True)
        meta, rows = evaluate_split(split, weights, bands, min_votes)
        split_meta.append(meta)
        split_rows.extend(rows)
        print(f"[split] {split} base={meta['base']['row_accuracy']:.6f} rows={len(rows)} combos={meta['combo_count']} masks={meta['mask_count']}", flush=True)
    agg = aggregate(split_rows)
    strict = [r for r in agg if r["mean_delta_vs_base"] >= 0.0020 and r["min_delta_vs_base"] >= 0.0008 and r["positive_splits"] == 3 and r["pooled_p_exact"] < 0.01 and (r["fixes"] - r["breaks"]) >= 120]
    manual = [r for r in agg if r["mean_delta_vs_base"] >= 0.0015 and r["min_delta_vs_base"] >= 0.0005 and r["positive_splits"] == 3 and r["pooled_p_exact"] < 0.03]
    verdict = "STRICT_PASS_NO_SUBMIT_REVIEW" if strict else ("MANUAL_RISK_WEAK" if manual else ("WEAK_SIGNAL" if agg and agg[0]["mean_delta_vs_base"] > 0 else "REJECT"))
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "metadata": {"splits": SPLITS, "weights": weights, "bands": bands, "min_votes": min_votes, "probe": "last-slot sparse agreement"},
        "base": split_meta,
        "strict_pass_count": len(strict),
        "manual_risk_count": len(manual),
        "verdict": verdict,
        "top_variants": agg[:100],
    }
    write_outputs(args, payload, split_rows, agg)
    print(json.dumps({"json": args.json, "md": args.md, "verdict": verdict, "strict_pass_count": len(strict), "manual_risk_count": len(manual), "top": agg[:10]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
