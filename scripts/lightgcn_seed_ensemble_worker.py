"""LightGCN seed-ensemble worker — one seed, all splits + full-test (CPU-light, GPU co-located).

Robust single-family variance reduction (NOT a cross-family blend), chosen because the
logreg stacker FAILED on public (overfit the validation negative sampler). Seed
ensembling averages raw scores of the SAME verified config (emb64 L3 reg1e-4), so it
cannot exploit split-specific artifacts.

For the assigned --seed, this trains LightGCN and saves raw candidate scores on:
  - 3 validation splits (sqrtpop, recent, popbin)  -> artifacts/lightgcn_seed_ensemble/seed{S}/val_{split}.csv
  - full train -> test pairs                       -> artifacts/lightgcn_seed_ensemble/seed{S}/test.csv
Seed 42 already exists elsewhere; this adds new seeds for averaging. No Kaggle submission.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (
    DEFAULT_DATA_DIR, build_user_item_matrix, load_train_json,
    load_pairs_csv, load_train_interactions, ensure_dir,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

SPLITS = ["val_random_sqrtpop_seed42", "val_recent_sqrtpop_seed42", "val_random_popbin_seed42"]
EMB_DIM, N_LAYERS, LR, REG, EPOCHS, BATCH = 64, 3, 1e-3, 1e-4, 200, 4096


def train(mat, n_users, n_items, seed, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    nb = max(1, mat.nnz // BATCH)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for _ in range(nb):
            u, p, ng = sample_bpr_batch(mat, BATCH, n_items, rng)
            ut, pt, nt = (torch.LongTensor(u).to(device), torch.LongTensor(p).to(device),
                          torch.LongTensor(ng).to(device))
            ue, ie = model(adj)
            loss = -F.logsigmoid((ue[ut]*ie[pt]).sum(1) - (ue[ut]*ie[nt]).sum(1)).mean()
            loss = loss + REG*(model.user_emb(ut).norm(2).pow(2)+model.item_emb(pt).norm(2).pow(2)
                               +model.item_emb(nt).norm(2).pow(2))/BATCH
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss.item())
        if (ep+1) % 50 == 0 or ep == 0:
            print(f"    ep{ep+1}/{EPOCHS} loss={epoch_loss/nb:.5f} t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    return ue.cpu().numpy(), ie.cpu().numpy(), round(time.time()-t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = ensure_dir(ROOT / f"artifacts/lightgcn_seed_ensemble/seed{args.seed}")

    # validation splits
    for split in SPLITS:
        sp = ROOT / "artifacts/validation" / split
        tr = load_train_interactions(sp / "train_interactions.csv")
        cand = load_pairs_csv(sp / "candidates.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
        print(f"[seed{args.seed}] {split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
        ue, ie, el = train(mat, len(users), len(items), args.seed, args.device)
        cand = cand.copy()
        cand["score"] = score_candidates(cand, ue, ie, u2i, i2i)
        cand[["ID", "userID", "gameID", "Label", "score"]].to_csv(out / f"val_{split}.csv", index=False)
        print(f"[seed{args.seed}] {split} done ({el}s) -> val_{split}.csv", flush=True)

    # full train -> test
    tr = load_train_json(DEFAULT_DATA_DIR / "train.json")
    pairs = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[seed{args.seed}] FULL: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
    ue, ie, el = train(mat, len(users), len(items), args.seed, args.device)
    pairs = pairs.copy()
    pairs["score"] = score_candidates(pairs, ue, ie, u2i, i2i)
    pairs[["ID", "userID", "gameID", "score"]].to_csv(out / "test.csv", index=False)
    print(f"[seed{args.seed}] FULL done ({el}s) -> test.csv", flush=True)
    print(f"[seed{args.seed}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
