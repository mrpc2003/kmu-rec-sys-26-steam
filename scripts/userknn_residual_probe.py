#!/usr/bin/env python3
"""Validation-only user-neighborhood residual probe for KMURecSys26 Steam.

This script does NOT call Kaggle and does NOT materialize a submission.  It tests
whether a user-user KNN signal from official train interactions can add a small,
validated residual correction to the current rank-blend LightGCN backbone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123"]
SEEDS = [42, 123, 2024, 7]


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"no score column in {df.columns.tolist()}")


def emb128_paths(split: str) -> list[Path]:
    if split == "val_random_uniform_seed42":
        return [
            ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
        ]
    return [ROOT / f"artifacts/split_panel_emb128/{split}/seed{s}/lightgcn_scores.csv" for s in SEEDS]


def emb192_paths(split: str) -> list[Path]:
    if split == "val_random_uniform_seed42":
        return [
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed123/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed7/val_random_uniform_seed42/lightgcn_scores.csv",
        ]
    return [ROOT / f"artifacts/split_panel_emb192/{split}/seed{s}/lightgcn_scores.csv" for s in SEEDS]


def load_ensemble(paths: list[Path], name: str) -> pd.DataFrame:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {name}: {missing}")
    first = pd.read_csv(paths[0])
    sc = score_col(first)
    out = first[["ID", "userID", "gameID", "Label"]].copy()
    out[name] = first[sc].astype(float).to_numpy()
    for p in paths[1:]:
        d = pd.read_csv(p)
        sc2 = score_col(d)
        part = d[["ID", sc2]].rename(columns={sc2: "_score"})
        before = len(out)
        out = out.merge(part, on="ID", validate="one_to_one")
        if len(out) != before:
            raise RuntimeError(f"row mismatch merging {p}")
        out[name] += out.pop("_score").astype(float).to_numpy()
    out[name] /= len(paths)
    return out


def user_rank_high_is_good(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=np.float64)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        ranks[idx[np.argsort(values[idx])]] = np.arange(len(idx), dtype=np.float64)
    return ranks


def top_half_pred(df: pd.DataFrame, score_col_name: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    values = df[score_col_name].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = int(df.loc[idx, "Label"].sum()) if "Label" in df.columns else len(idx) // 2
        order = np.argsort(values[idx])[::-1]
        pred[idx[order[:k]]] = 1
    return pred


def eval_pred(y: np.ndarray, pred: np.ndarray, base_pred: np.ndarray | None = None) -> dict[str, object]:
    correct = pred == y
    out: dict[str, object] = {"row_accuracy": float(correct.mean()), "correct": int(correct.sum())}
    if base_pred is not None:
        base_correct = base_pred == y
        fixes = int((correct & ~base_correct).sum())
        breaks = int((~correct & base_correct).sum())
        discordant = fixes + breaks
        p = float(binomtest(min(fixes, breaks), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
        out.update({"fixes": fixes, "breaks": breaks, "discordant": discordant, "mcnemar_p_exact": p, "delta_vs_base": float(correct.mean() - base_correct.mean())})
    return out


def z_within_user(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mean = g.transform("mean").to_numpy(dtype=float)
    std = g.transform("std").fillna(0.0).to_numpy(dtype=float)
    val = df[col].to_numpy(dtype=float)
    return np.where(std > 1e-12, (val - mean) / std, 0.0)


def build_matrix(train: pd.DataFrame) -> tuple[sp.csr_matrix, dict[str, int], dict[str, int]]:
    users = sorted(train["userID"].unique())
    items = sorted(train["gameID"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    g2i = {g: i for i, g in enumerate(items)}
    rows = train["userID"].map(u2i).to_numpy(np.int32)
    cols = train["gameID"].map(g2i).to_numpy(np.int32)
    data = np.ones(len(train), dtype=np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)), dtype=np.float32)
    X.data[:] = 1.0
    X.eliminate_zeros()
    return X, u2i, g2i


def add_userknn_scores(split: str, df: pd.DataFrame) -> pd.DataFrame:
    split_dir = ROOT / "artifacts/validation" / split
    train = pd.read_csv(split_dir / "train_interactions.csv")
    X, u2i, g2i = build_matrix(train)
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=1)).ravel()).astype(np.float32)
    inv_norms = np.where(norms > 0, 1.0 / norms, 0.0).astype(np.float32)
    Xn = X.multiply(inv_norms[:, None]).tocsr()
    # 6710x6710 dense float32 is about 180MB and is faster than row-wise sparse loops here.
    S = (Xn @ Xn.T).astype(np.float32).toarray()
    np.fill_diagonal(S, 0.0)
    csc = X.tocsc()
    cand_u = df["userID"].map(u2i).fillna(-1).astype(int).to_numpy()
    cand_g = df["gameID"].map(g2i).fillna(-1).astype(int).to_numpy()
    score_sum = np.zeros(len(df), dtype=np.float32)
    score_top10 = np.zeros(len(df), dtype=np.float32)
    score_max = np.zeros(len(df), dtype=np.float32)
    score_popnorm = np.zeros(len(df), dtype=np.float32)
    for n, (ui, gi) in enumerate(zip(cand_u, cand_g)):
        if ui < 0 or gi < 0:
            continue
        start, end = csc.indptr[gi], csc.indptr[gi + 1]
        holders = csc.indices[start:end]
        if holders.size == 0:
            continue
        sims = S[ui, holders]
        if sims.size == 0:
            continue
        ss = float(sims.sum())
        score_sum[n] = ss
        score_max[n] = float(sims.max())
        k = min(10, sims.size)
        if k:
            score_top10[n] = float(np.partition(sims, -k)[-k:].mean())
        score_popnorm[n] = ss / math.sqrt(float(holders.size) + 1.0)
    out = df.copy()
    out["score_userknn_sum"] = score_sum
    out["score_userknn_top10"] = score_top10
    out["score_userknn_max"] = score_max
    out["score_userknn_popnorm"] = score_popnorm
    return out


def scan_split(split: str, weights: Iterable[float], bands: Iterable[int]) -> tuple[dict[str, object], list[dict[str, object]]]:
    a = load_ensemble(emb128_paths(split), "score_emb128")
    b = load_ensemble(emb192_paths(split), "score_emb192")[["ID", "score_emb192"]]
    df = a.merge(b, on="ID", validate="one_to_one")
    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_base_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    y = df["Label"].astype(int).to_numpy()
    base_pred = top_half_pred(df, "score_base_rankblend")
    base_eval = eval_pred(y, base_pred)
    df = add_userknn_scores(split, df)
    z_base = z_within_user(df, "score_base_rankblend")
    rows: list[dict[str, object]] = []
    aux_cols = ["score_userknn_sum", "score_userknn_top10", "score_userknn_max", "score_userknn_popnorm"]
    # Standalone scores.
    for col in aux_cols:
        pred = top_half_pred(df, col)
        ev = eval_pred(y, pred, base_pred)
        ev.update({"split": split, "variant": col, "score_col": col, "mode": "standalone"})
        rows.append(ev)
    # Z blends, optionally only around the base decision boundary.
    base_rank_desc = np.zeros(len(df), dtype=np.float64)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        vals = df.loc[idx, "score_base_rankblend"].to_numpy(float)
        order = np.argsort(vals)[::-1]
        base_rank_desc[idx[order]] = np.arange(1, len(idx) + 1)
    cutoff = df.groupby("userID")["Label"].transform("sum").to_numpy(float) + 0.5
    margin = np.abs(base_rank_desc - cutoff)
    for col in aux_cols:
        z_aux = z_within_user(df, col)
        for w in weights:
            score = z_base + float(w) * z_aux
            vname = f"zbase_plus_{col}_w{w:g}"
            df[vname] = score
            pred = top_half_pred(df, vname)
            ev = eval_pred(y, pred, base_pred)
            ev.update({"split": split, "variant": vname, "score_col": col, "mode": "zblend", "weight": float(w), "band": None})
            rows.append(ev)
            for band in bands:
                gated = z_base + float(w) * z_aux * (margin <= band)
                gv = f"zbase_plus_{col}_w{w:g}_band{band}"
                df[gv] = gated
                pred = top_half_pred(df, gv)
                ev = eval_pred(y, pred, base_pred)
                ev.update({"split": split, "variant": gv, "score_col": col, "mode": "gated_zblend", "weight": float(w), "band": int(band)})
                rows.append(ev)
    return {"split": split, "base": base_eval}, rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by.setdefault(str(r["variant"]), []).append(r)
    out = []
    for variant, rs in by.items():
        if len(rs) != len(SPLITS):
            continue
        fixes = sum(int(r.get("fixes", 0)) for r in rs)
        breaks = sum(int(r.get("breaks", 0)) for r in rs)
        disc = fixes + breaks
        p = float(binomtest(min(fixes, breaks), disc, 0.5, alternative="two-sided").pvalue) if disc else 1.0
        deltas = [float(r.get("delta_vs_base", 0.0)) for r in rs]
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
            "mode": rs[0].get("mode"),
            "score_col": rs[0].get("score_col"),
            "weight": rs[0].get("weight"),
            "band": rs[0].get("band"),
            "split_deltas": {str(r["split"]): float(r.get("delta_vs_base", 0.0)) for r in rs},
        })
    out.sort(key=lambda x: (x["mean_delta_vs_base"], x["positive_splits"], -x["pooled_p_exact"]), reverse=True)
    return out


def write_md(path: Path, payload: dict[str, object]) -> None:
    lines = ["# UserKNN residual probe", "", "No Kaggle submission is performed by this report.", "", "## Base rankblend", ""]
    for b in payload["base_summaries"]:  # type: ignore[index]
        lines.append(f"- {b['split']}: acc={b['base']['row_accuracy']:.6f}")
    lines.extend(["", "## Top aggregate variants", "", "| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for i, r in enumerate(payload["aggregate"][:30], 1):  # type: ignore[index]
        lines.append(f"| {i} | `{r['variant']}` | {r['mean_delta_vs_base']:+.6f} | {r['min_delta_vs_base']:+.6f} | {r['positive_splits']}/3 | {r['fixes']} | {r['breaks']} | {r['pooled_p_exact']:.4g} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "reports/userknn_residual_probe.json"))
    ap.add_argument("--md", default=str(ROOT / "reports/userknn_residual_probe.md"))
    ap.add_argument("--weights", default="-2,-1,-0.5,-0.25,0.25,0.5,1,2")
    ap.add_argument("--bands", default="1,2")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x]
    bands = [int(x) for x in args.bands.split(",") if x]
    base_summaries = []
    all_rows: list[dict[str, object]] = []
    for split in SPLITS:
        print(f"[split] {split}", flush=True)
        base, rows = scan_split(split, weights, bands)
        base_summaries.append(base)
        all_rows.extend(rows)
        print(f"[split] {split} base={base['base']['row_accuracy']:.6f} best_delta={max(float(r.get('delta_vs_base', 0.0)) for r in rows):+.6f}", flush=True)
    payload = {
        "metadata": {
            "splits": SPLITS,
            "weights": weights,
            "bands": bands,
            "no_kaggle_submission": True,
            "description": "user-user cosine KNN residual against rankblend emb128+emb192",
        },
        "base_summaries": base_summaries,
        "rows": all_rows,
        "aggregate": aggregate(all_rows),
    }
    jp = Path(args.json)
    mp = Path(args.md)
    jp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(mp, payload)
    print(json.dumps({"json": str(jp), "md": str(mp), "top": payload["aggregate"][:10]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
