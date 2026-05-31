#!/usr/bin/env python3
"""SGL: Self-supervised Graph Learning for LightGCN (Wu et al., SIGIR 2021).

WHY this is the last "strong + orthogonal" bet (kmu-rec-sys-26-steam, 2026-05-31):
The new-direction sweep showed every classic CF axis (ALS/EASE/ItemKNN) is REDUNDANT with
the BPR-LightGCN family (corr 0.73-0.83) and DirectAU is orthogonal (corr 0.16) but far too
weak alone (0.55). DirectAU failed because it REPLACED BPR and lost ranking strength.

SGL keeps BPR and ADDS a contrastive regularizer:
    loss = BPR  +  lambda_ssl * InfoNCE(view1, view2)  +  reg
The two views come from EDGE DROPOUT on the graph. The main (clean-graph) embeddings still
drive BPR/scoring, so ranking strength is preserved (~0.76), while the contrastive term
nudges the representation to be more uniform/robust -> a chance to be PARTIALLY orthogonal
to plain BPR-LightGCN and crack some of the 21.4% "neither correct" uniform rows.

lambda_ssl is the key knob: small -> ~pure BPR (strong, not orthogonal); large -> more
contrastive (orthogonal, risk of weakening). We sweep it.

Reuses LightGCN / build_norm_adj / score_candidates from lightgcn_train.py.
Validation-only by default; no Kaggle submission.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (
    DEFAULT_DATA_DIR, build_user_item_matrix, ensure_dir, evaluate_tophalf,
    load_pairs_csv, load_train_interactions, load_train_json, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")


def edge_dropout_adj(mat: sp.csr_matrix, n_users: int, n_items: int, rate: float,
                     rng: np.random.Generator, device: str) -> torch.Tensor:
    """Build a symmetric normalized adjacency after dropping `rate` of the interaction edges."""
    coo = mat.tocoo()
    keep = rng.random(coo.nnz) >= rate
    sub = sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=mat.shape)
    return build_norm_adj(sub.tocsr(), n_users, n_items).to(device)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE over a batch of node embeddings (positives = same node's two views)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    pos = (z1 * z2).sum(dim=1) / tau
    logits = (z1 @ z2.t()) / tau
    return (-pos + torch.logsumexp(logits, dim=1)).mean()


def train_sgl(mat, n_users, n_items, emb_dim, n_layers, lr, reg,
              lam_ssl, tau, drop, epochs, batch_size, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_norm_adj(mat, n_users, n_items).to(device)  # clean graph for BPR/scoring
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    nb = max(1, mat.nnz // batch_size)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        # two augmented views per epoch (standard SGL-ED)
        adj1 = edge_dropout_adj(mat, n_users, n_items, drop, rng, device)
        adj2 = edge_dropout_adj(mat, n_users, n_items, drop, rng, device)
        el = eb = es = 0.0
        for _ in range(nb):
            u, p, ng = sample_bpr_batch(mat, batch_size, n_items, rng)
            ut = torch.LongTensor(u).to(device)
            pt = torch.LongTensor(p).to(device)
            nt = torch.LongTensor(ng).to(device)

            ue, ie = model(adj)
            bpr = -F.logsigmoid((ue[ut] * ie[pt]).sum(1) - (ue[ut] * ie[nt]).sum(1)).mean()
            reg_l = reg * (model.user_emb(ut).norm(2).pow(2)
                           + model.item_emb(pt).norm(2).pow(2)
                           + model.item_emb(nt).norm(2).pow(2)) / batch_size

            # contrastive on the unique batch users + pos items, across two dropped views
            u1, i1 = model(adj1)
            u2, i2 = model(adj2)
            uu = torch.unique(ut)
            ii = torch.unique(pt)
            ssl = info_nce(u1[uu], u2[uu], tau) + info_nce(i1[ii], i2[ii], tau)

            loss = bpr + lam_ssl * ssl + reg_l
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += float(loss.item()); eb += float(bpr.item()); es += float(ssl.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep{ep+1}/{epochs} loss={el/nb:.4f} bpr={eb/nb:.4f} ssl={es/nb:.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    meta = {"emb_dim": emb_dim, "n_layers": n_layers, "lr": lr, "reg": reg,
            "lambda_ssl": lam_ssl, "tau": tau, "edge_drop": drop, "epochs": epochs,
            "batch_size": batch_size, "n_users": n_users, "n_items": n_items,
            "n_interactions": int(mat.nnz), "train_seconds": round(time.time() - t0, 1),
            "device": device, "seed": seed}
    return ue.cpu().numpy(), ie.cpu().numpy(), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["val", "test"], default="val")
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--lambda-ssl", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--edge-drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"sgl_emb{args.emb_dim}_L{args.n_layers}_lam{args.lambda_ssl:g}_t{args.tau:g}_d{args.edge_drop:g}_seed{args.seed}"

    if args.mode == "val":
        sp_dir = ROOT / "artifacts/validation" / args.split
        tr = load_train_interactions(sp_dir / "train_interactions.csv")
        cand = load_pairs_csv(sp_dir / "candidates.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
        print(f"[{tag}] {args.split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
        ue, ie, meta = train_sgl(mat, len(users), len(items), args.emb_dim, args.n_layers,
                                 args.lr, args.reg, args.lambda_ssl, args.tau, args.edge_drop,
                                 args.epochs, args.batch_size, args.device, args.seed)
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
        ue, ie, meta = train_sgl(mat, len(users), len(items), args.emb_dim, args.n_layers,
                                 args.lr, args.reg, args.lambda_ssl, args.tau, args.edge_drop,
                                 args.epochs, args.batch_size, args.device, args.seed)
        pairs = pairs.copy()
        pairs["score_lightgcn"] = score_candidates(pairs, ue, ie, u2i, i2i)
        pairs[["ID", "userID", "gameID", "score_lightgcn"]].to_csv(out / "test.csv", index=False)
        write_json(out / "meta.json", {"tag": tag, "meta": meta})
        print(f"[{tag}] done -> {out/'test.csv'} ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
