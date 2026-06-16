#!/usr/bin/env python3
"""Validation-only TAG-CF-style test-time aggregation probe.

Goal: test a not-yet-tried operator axis: after training the canonical LightGCN
with the same BPR/uniform-negative objective, apply parameter-free neighbor
aggregation at inference time to the final embeddings. This probes whether a
TAG-CF-like aggregation operator can improve the per-user top-half uniform gate.

Safety: no Kaggle submission, no test candidate materialization by default.
Writes validation score diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lightgcn_train import train_lightgcn, score_candidates
from recsys_played_utils import (
    DEFAULT_DATA_DIR,
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
)


def _row_normalized(mat: sp.csr_matrix) -> sp.csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    deg = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
    inv = np.zeros_like(deg, dtype=np.float32)
    np.divide(1.0, deg, out=inv, where=deg > 0)
    return sp.diags(inv) @ mat


def _sym_parts(mat: sp.csr_matrix) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    mat = mat.tocsr().astype(np.float32)
    du = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
    di = np.asarray(mat.sum(axis=0)).ravel().astype(np.float32)
    du_inv = np.zeros_like(du, dtype=np.float32)
    di_inv = np.zeros_like(di, dtype=np.float32)
    np.divide(1.0 / np.sqrt(np.maximum(du, 1e-12)), 1.0, out=du_inv, where=du > 0)
    np.divide(1.0 / np.sqrt(np.maximum(di, 1e-12)), 1.0, out=di_inv, where=di > 0)
    # user-side aggregation: D_u^-1/2 R D_i^-1/2 item_emb
    user_op = sp.diags(du_inv) @ mat @ sp.diags(di_inv)
    # item-side aggregation: D_i^-1/2 R^T D_u^-1/2 user_emb
    item_op = sp.diags(di_inv) @ mat.T @ sp.diags(du_inv)
    return user_op.tocsr(), item_op.tocsr()


def _l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def make_tag_embeddings(
    mat: sp.csr_matrix,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    alpha: float,
    mode: str,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return TAG-style user/item embeddings for one variant."""
    if mode == "mean":
        user_agg = _row_normalized(mat) @ item_emb
        item_agg = _row_normalized(mat.T.tocsr()) @ user_emb
    elif mode == "sym":
        user_op, item_op = _sym_parts(mat)
        user_agg = user_op @ item_emb
        item_agg = item_op @ user_emb
    else:
        raise ValueError(mode)

    u = (1.0 - alpha) * user_emb + alpha * user_agg
    i = (1.0 - alpha) * item_emb + alpha * item_agg
    if normalize:
        u = _l2norm(u)
        i = _l2norm(i)
    return u.astype(np.float32), i.astype(np.float32)


def predict_tophalf(df: pd.DataFrame, score_col: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    for _, idx in df.groupby("userID", sort=False).groups.items():
        ids = np.asarray(list(idx))
        k = int(df.loc[ids, "Label"].sum())
        scores = df.loc[ids, score_col].to_numpy()
        row_ids = df.loc[ids, "ID"].to_numpy()
        order = np.lexsort((row_ids, -scores))
        pred[ids[order[:k]]] = 1
    return pred


def eval_vs_base(candidates: pd.DataFrame, score_col: str, base_pred: np.ndarray) -> Dict[str, float]:
    summary, _ = evaluate_tophalf(candidates, score_col, label_col="Label", user_col="userID", id_col="ID")
    y = candidates["Label"].to_numpy(dtype=np.int8)
    pred = predict_tophalf(candidates, score_col)
    base_ok = base_pred == y
    cand_ok = pred == y
    b = int(np.sum(base_ok & ~cand_ok))
    c = int(np.sum(~base_ok & cand_ok))
    if b + c == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (max(abs(b - c) - 1, 0) ** 2) / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "score": score_col,
        "row_accuracy": float(summary["row_accuracy"]),
        "delta_vs_base": float(summary["row_accuracy"] - (base_pred == y).mean()),
        "flips": int(np.sum(pred != base_pred)),
        "candidate_fixes": c,
        "base_breaks": b,
        "mcnemar_p": float(p),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    print(f"[TAG-CF probe] split={split_dir.name} users={len(users)} items={len(items)} interactions={mat.nnz}", flush=True)
    user_emb, item_emb, train_meta = train_lightgcn(
        mat,
        len(users),
        len(items),
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    candidates = candidates.copy()
    candidates["score_base"] = score_candidates(candidates, user_emb, item_emb, user_to_idx, item_to_idx)
    base_pred = predict_tophalf(candidates, "score_base")
    results = [eval_vs_base(candidates, "score_base", base_pred)]

    alphas = [float(x) for x in args.alphas.split(",") if x]
    modes = [x for x in args.modes.split(",") if x]
    for mode in modes:
        for alpha in alphas:
            for normalize in [False, True]:
                u_tag, i_tag = make_tag_embeddings(mat, user_emb, item_emb, alpha=alpha, mode=mode, normalize=normalize)
                tag = f"score_tag_{mode}_a{str(alpha).replace('.', 'p')}_{'l2' if normalize else 'raw'}"
                candidates[tag] = score_candidates(candidates, u_tag, i_tag, user_to_idx, item_to_idx)
                # also test a conservative 50/50 z blend with base.
                g = candidates.groupby("userID")
                z_base = (candidates["score_base"] - g["score_base"].transform("mean")) / g["score_base"].transform("std").replace(0, np.nan).fillna(1.0)
                z_tag = (candidates[tag] - g[tag].transform("mean")) / g[tag].transform("std").replace(0, np.nan).fillna(1.0)
                blend_tag = tag.replace("score_tag", "score_blend")
                candidates[blend_tag] = 0.5 * z_base + 0.5 * z_tag
                results.append(eval_vs_base(candidates, tag, base_pred))
                results.append(eval_vs_base(candidates, blend_tag, base_pred))

    results = sorted(results, key=lambda r: (r["row_accuracy"], r["delta_vs_base"]), reverse=True)
    score_cols = ["ID", "userID", "gameID", "Label", "score_base"] + [r["score"] for r in results if r["score"] != "score_base"]
    candidates[score_cols].to_csv(out_dir / "tagcf_validation_scores.csv", index=False)

    best_nonbase = next(r for r in results if r["score"] != "score_base")
    if best_nonbase["delta_vs_base"] >= args.mde and best_nonbase["mcnemar_p"] < 0.05:
        gate = "ESCALATE_3SPLIT"
    elif best_nonbase["delta_vs_base"] > 0 and best_nonbase["mcnemar_p"] < 0.05:
        gate = "WEAK_SIGNAL_PANEL_ONLY"
    else:
        gate = "REJECT"
    summary = {
        "safety": {"validation_only": True, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "split": split_dir.name,
        "args": vars(args),
        "train_meta": train_meta,
        "results": results,
        "best_nonbase": best_nonbase,
        "gate": gate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = out_dir / "tagcf_probe_report.md"
    lines = [
        f"# TAG-CF-style test-time aggregation probe — {split_dir.name}\n",
        "**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false\n\n",
        f"Baseline: emb_dim={args.emb_dim}, layers={args.n_layers}, reg={args.reg}, seed={args.seed}, epochs={args.epochs}.\n\n",
        "| score | acc | delta_vs_base | flips | fixes | breaks | McNemar p |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in results[:16]:
        lines.append(
            f"| {r['score']} | {r['row_accuracy']:.5f} | {r['delta_vs_base']:+.5f} | {r['flips']} | {r['candidate_fixes']} | {r['base_breaks']} | {r['mcnemar_p']:.4f} |\n"
        )
    lines.append(f"\n## Gate verdict\n**{gate}** — best non-baseline `{best_nonbase['score']}` delta={best_nonbase['delta_vs_base']:+.5f}, p={best_nonbase['mcnemar_p']:.4f}.\n")
    report.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "report": str(report), "best_nonbase": best_nonbase, "gate": gate}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default=str(ROOT / "artifacts/validation/val_random_uniform_seed42"))
    p.add_argument("--out-dir", default="artifacts/tagcf_probe/emb128_L4_r3_seed42")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--modes", default="mean,sym")
    p.add_argument("--mde", type=float, default=0.00355)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
