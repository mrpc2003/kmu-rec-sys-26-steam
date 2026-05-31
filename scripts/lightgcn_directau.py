#!/usr/bin/env python3
"""LightGCN with DirectAU loss (alignment + uniformity, Wang et al. KDD 2022).

Motivation (kmu-rec-sys-26-steam, 2026-05-31): the emb64 and emb128 BPR-LightGCN
ensembles are 0.97 correlated on the uniform surrogate and a 50/50 blend HURTS — the
BPR-LightGCN family has saturated. 21.4% of uniform rows are wrong for BOTH. To crack
those we need a DIFFERENT inductive bias, not a bigger backbone.

DirectAU replaces BPR with:
    align(u,i)   = || norm(e_u) - norm(e_i) ||^2                 (positive pairs only)
    uniform(x)   = log E_{x,y} exp(-2 || norm(e_x) - norm(e_y) ||^2)
    loss = alignment + gamma * (uniform(users) + uniform(items)) / 2

Why it fits THIS task:
- The uniformity term spreads embeddings evenly on the hypersphere, which matches the
  test's UNIFORM negative sampling (our verified public surrogate) far better than BPR,
  whose pairwise objective bakes in popularity bias.
- No negative sampling at all (positive pairs only) -> faster, and it learns NOTHING from
  validation labels, so it is immune to the stacker's negative-sampler overfit failure.
- Different geometry than BPR => candidate to decorrelate from the LightGCN-BPR family and
  reach the "neither correct" rows.

Reuses LightGCN / build_norm_adj / score_candidates from lightgcn_train.py so the graph
propagation is identical to the BPR models; only the training objective differs.

Validation-only by default; no Kaggle submission.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (
    DEFAULT_DATA_DIR, build_user_item_matrix, ensure_dir, evaluate_tophalf,
    load_pairs_csv, load_train_interactions, load_train_json, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, score_candidates

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")


def alignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).norm(p=2, dim=1).pow(2).mean()


def uniformity(x: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    # log E exp(-t * ||xi - xj||^2) over all pairs in the batch
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def sample_pos_pairs(mat, batch_size: int, rng: np.random.Generator):
    """Sample (user, pos_item) pairs proportional to interaction count."""
    n_users = mat.shape[0]
    users = rng.integers(0, n_users, size=batch_size)
    pos = np.zeros(batch_size, dtype=np.int64)
    for i, u in enumerate(users):
        s, e = mat.indptr[u], mat.indptr[u + 1]
        if s == e:
            # reroll to a user with interactions
            while s == e:
                u = rng.integers(0, n_users)
                s, e = mat.indptr[u], mat.indptr[u + 1]
            users[i] = u
        pos[i] = rng.choice(mat.indices[s:e])
    return users, pos


def train_directau(mat, n_users, n_items, emb_dim, n_layers, lr, gamma,
                   epochs, batch_size, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    nb = max(1, mat.nnz // batch_size)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = ep_al = ep_un = 0.0
        for _ in range(nb):
            u, p = sample_pos_pairs(mat, batch_size, rng)
            ut = torch.LongTensor(u).to(device)
            pt = torch.LongTensor(p).to(device)
            ue, ie = model(adj)
            eu = F.normalize(ue[ut], dim=1)
            ei = F.normalize(ie[pt], dim=1)
            al = alignment(eu, ei)
            un = (uniformity(eu) + uniformity(ei)) / 2
            loss = al + gamma * un
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()); ep_al += float(al.item()); ep_un += float(un.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep{ep+1}/{epochs} loss={ep_loss/nb:.5f} align={ep_al/nb:.5f} "
                  f"unif={ep_un/nb:.5f} t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    # score with normalized embeddings to match the trained (cosine) geometry
    ue = F.normalize(ue, dim=1).cpu().numpy()
    ie = F.normalize(ie, dim=1).cpu().numpy()
    meta = {"emb_dim": emb_dim, "n_layers": n_layers, "lr": lr, "gamma": gamma,
            "epochs": epochs, "batch_size": batch_size, "n_users": n_users,
            "n_items": n_items, "n_interactions": int(mat.nnz),
            "train_seconds": round(time.time() - t0, 1), "device": device, "seed": seed}
    return ue, ie, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"directau_emb{args.emb_dim}_L{args.n_layers}_g{args.gamma:g}_seed{args.seed}"

    if args.mode == "val":
        sp = ROOT / "artifacts/validation" / args.split
        tr = load_train_interactions(sp / "train_interactions.csv")
        cand = load_pairs_csv(sp / "candidates.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
        print(f"[{tag}] {args.split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
        ue, ie, meta = train_directau(mat, len(users), len(items), args.emb_dim,
                                       args.n_layers, args.lr, args.gamma, args.epochs,
                                       args.batch_size, args.device, args.seed)
        cand = cand.copy()
        cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
        summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
        outd = ensure_dir(out / args.split)
        cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(outd / "lightgcn_scores.csv", index=False)
        write_json(outd / "summary.json", {"tag": tag, "row_accuracy": float(summ["row_accuracy"]), "meta": meta})
        print(f"[{tag}] uniform row_acc={summ['row_accuracy']:.5f} ({meta['train_seconds']}s)", flush=True)
    else:
        tr = load_train_json(DEFAULT_DATA_DIR / "train.json")
        pairs = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
        print(f"[{tag}] FULL: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
        ue, ie, meta = train_directau(mat, len(users), len(items), args.emb_dim,
                                      args.n_layers, args.lr, args.gamma, args.epochs,
                                      args.batch_size, args.device, args.seed)
        pairs = pairs.copy()
        pairs["score_lightgcn"] = score_candidates(pairs, ue, ie, u2i, i2i)
        pairs[["ID", "userID", "gameID", "score_lightgcn"]].to_csv(out / "test.csv", index=False)
        write_json(out / "meta.json", {"tag": tag, "meta": meta})
        print(f"[{tag}] done -> {out/'test.csv'} ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
