#!/usr/bin/env python3
"""Hours confidence-weighted LightGCN — UNIFORM gate (single seed first).

WHY (GPT-5.5 Pro direction #4, honest prior LOW)
-------------------------------------------------
The competition label is BINARY played: a 0.2-hour positive and a 200-hour positive
are both Label=1. So `hours` must NOT be used as preference magnitude (that mismatches
the target). The only defensible use is EDGE CONFIDENCE (Hu-Koren-Volinsky 2008): weight
the graph edge by how reliable the interaction is, leaving the binary topology intact.

This modifies the BACKBONE (the adjacency edge weights), unlike subset-loss (objective)
or temporal (post-hoc rerank). corr 0.98 saturation might bend if high-confidence edges
stabilize the user-taste embeddings differently. Honest prior is LOW because (a) the label
is existence not magnitude, and (b) it is a re-weighting of the SAME graph signal, so
orthogonality to LightGCN is expected to be small.

CONFIDENCE TRANSFORMS (all fixed, parameter-free, no LB fitting):
  user_quantile : c_ui = 0.5 + rankpct(hours_transformed | user)      (conservative)
  item_quantile : c_ui = 0.5 + rankpct(hours_transformed | item)
  balanced      : c_ui = 0.5 + 0.5*rankpct(.|user) + 0.5*rankpct(.|item) - 0.25  (->[0.5,1.0])
Edge value in symmetric-normalized adjacency = sqrt(c_ui). Absolute hours never used.

CONFOUND CONTROL: compare each weighted single-seed against the canonical BINARY
single-seed (emb128 L4 reg1e-3 seed42 uniform = 0.76205). The binary path is the proven
0.77745 backbone, so "weighting helped" only if it CLEARLY beats 0.76205 by > noise.

Gate: single-seed uniform Δ vs 0.76205:
  > +0.0007 -> build seed ensemble & re-gate (capacity-probe protocol)
  within ±0.0007 -> plateau / no gain
  < -0.0007 -> confidence weighting hurts this binary-label task

Validation-only. No Kaggle submission.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix, load_pairs_csv, load_train_interactions,
    evaluate_tophalf, ensure_dir, write_json,
)
from lightgcn_train import LightGCN, sample_bpr_batch, score_candidates  # noqa: E402

SPLIT = "val_random_uniform_seed42"
BINARY_SINGLE_SEED_REF = 0.76205   # emb128 L4 reg1e-3 seed42 uniform (binary backbone)
NOISE = 0.0007


def compute_confidence(ft: pd.DataFrame, mode: str) -> np.ndarray:
    """Return per-row edge confidence in [0.5, 1.0] from hours_transformed."""
    h = ft["hours_transformed"].to_numpy(dtype=np.float64)
    df = ft.copy()
    df["_h"] = h
    if mode == "user_quantile":
        pct = df.groupby("userID")["_h"].rank(pct=True, method="average").to_numpy()
        return 0.5 + 0.5 * pct
    if mode == "item_quantile":
        pct = df.groupby("gameID")["_h"].rank(pct=True, method="average").to_numpy()
        return 0.5 + 0.5 * pct
    if mode == "balanced":
        pu = df.groupby("userID")["_h"].rank(pct=True, method="average").to_numpy()
        pi = df.groupby("gameID")["_h"].rank(pct=True, method="average").to_numpy()
        return 0.5 + 0.5 * (0.5 * pu + 0.5 * pi)
    raise ValueError(mode)


def build_weighted_norm_adj(ft, u2i, i2i, n_users, n_items, conf) -> torch.Tensor:
    """Symmetric-normalized adjacency with edge value sqrt(confidence)."""
    rows = ft["userID"].map(u2i).to_numpy()
    cols = ft["gameID"].map(i2i).to_numpy()
    vals = np.sqrt(np.clip(conf, 1e-6, None)).astype(np.float32)
    n = n_users + n_items
    R = sp.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    R = R.tocoo()
    arow = np.concatenate([R.row, R.col + n_users])
    acol = np.concatenate([R.col + n_users, R.row])
    adata = np.concatenate([R.data, R.data])
    adj = sp.coo_matrix((adata, (arow, acol)), shape=(n, n))
    deg = np.array(adj.sum(axis=1)).flatten()
    dinv = np.where(deg > 0, np.power(deg, -0.5), 0.0).astype(np.float32)
    D = sp.diags(dinv)
    norm = (D @ adj @ D).tocoo()
    idx = torch.LongTensor(np.vstack([norm.row, norm.col]))
    v = torch.FloatTensor(norm.data)
    return torch.sparse_coo_tensor(idx, v, torch.Size([n, n]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confidence-mode", required=True,
                    choices=["user_quantile", "item_quantile", "balanced", "binary_control"])
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="artifacts/hours_confidence")
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir) / args.confidence_mode / SPLIT)
    sp_dir = ROOT / "artifacts/validation" / SPLIT
    ft = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(ft, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"[hours:{args.confidence_mode}] {SPLIT}: {n_users}u {n_items}i {mat.nnz}nnz", flush=True)

    if args.confidence_mode == "binary_control":
        conf = np.ones(len(ft), dtype=np.float64)
    else:
        conf = compute_confidence(ft, args.confidence_mode)
    print(f"[hours:{args.confidence_mode}] conf range [{conf.min():.3f},{conf.max():.3f}] mean {conf.mean():.3f}", flush=True)

    device = args.device
    adj = build_weighted_norm_adj(ft, u2i, i2i, n_users, n_items, conf).to(device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = LightGCN(n_users, n_items, args.emb_dim, args.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    n_batches = max(1, mat.nnz // args.batch_size)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for _ in range(n_batches):
            us, ps, ns = sample_bpr_batch(mat, args.batch_size, n_items, rng)
            ut = torch.from_numpy(us).long().to(device)
            pt = torch.from_numpy(ps).to(device)
            nt = torch.from_numpy(ns).to(device)
            ue, ie = model(adj)
            ps_ = (ue[ut] * ie[pt]).sum(1)
            ns_ = (ue[ut] * ie[nt]).sum(1)
            bpr = -F.logsigmoid(ps_ - ns_).mean()
            rl = args.reg * (model.user_emb(ut).norm(2).pow(2) + model.item_emb(pt).norm(2).pow(2)
                             + model.item_emb(nt).norm(2).pow(2)) / args.batch_size
            loss = bpr + rl
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  [{args.confidence_mode}] {epoch+1}/{args.epochs} loss={loss.item():.5f} {time.time()-t0:.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    ue, ie = ue.cpu().numpy(), ie.cpu().numpy()
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    acc = round(float(summ["row_accuracy"]), 5)
    d = round(acc - BINARY_SINGLE_SEED_REF, 5)
    if d > NOISE:
        tier = "CONF_GAIN_CHECK_ENSEMBLE"
    elif d >= -NOISE:
        tier = "CONF_PLATEAU_NO_GAIN"
    else:
        tier = "CONF_REGRESS"

    result = {"split": SPLIT, "confidence_mode": args.confidence_mode, "acc": acc,
              "binary_single_seed_ref": BINARY_SINGLE_SEED_REF, "delta_vs_binary": d,
              "noise_band": NOISE, "tier": tier, "seed": args.seed,
              "train_seconds": round(time.time() - t0, 1)}
    write_json(out / "summary.json", result)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out / "scores.csv", index=False)
    print(f"\n[hours:{args.confidence_mode}] uniform acc={acc} vs binary {BINARY_SINGLE_SEED_REF} = {d:+.5f} -> {tier}", flush=True)


if __name__ == "__main__":
    main()
