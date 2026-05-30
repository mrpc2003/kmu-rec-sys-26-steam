"""LightGCN sweep worker — runs a subset of configs on one GPU.

Usage:
  python scripts/lightgcn_sweep_worker.py --device cuda:0 --gpu-id 0
"""
from __future__ import annotations

import argparse
import json
import time
import itertools
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
    load_pairs_csv, load_train_interactions, evaluate_tophalf,
    ensure_dir, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

SPLITS = [
    "val_random_sqrtpop_seed42",
    "val_recent_sqrtpop_seed42",
    "val_random_popbin_seed42",
]
BASELINE = {
    "val_random_sqrtpop_seed42": 0.6748,
    "val_recent_sqrtpop_seed42": 0.6396,
    "val_random_popbin_seed42":  0.6020,
}
EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
SEED = 42

ALL_CONFIGS = [
    (e, l, r)
    for e, l, r in itertools.product([64, 128, 256], [2, 3, 4], [1e-4, 1e-3])
    if not (e == 64 and l == 3 and r == 1e-4)  # skip baseline
]


def eval_split(split_name, user_np, item_np, u2i, i2i) -> float:
    """Canonical eval: candidates.csv + evaluate_tophalf row_accuracy (matches baseline)."""
    split_path = ROOT / "artifacts/validation" / split_name
    candidates = load_pairs_csv(split_path / "candidates.csv")
    scores = score_candidates(candidates, user_np, item_np, u2i, i2i)
    candidates = candidates.copy()
    candidates["score_lightgcn"] = scores
    summary, _ = evaluate_tophalf(
        candidates, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID"
    )
    return float(summary["row_accuracy"])


def train_lgcn(mat, n_users, n_items, emb_dim, n_layers, reg, device):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    n_batches = max(1, mat.nnz // BATCH_SIZE)
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(mat, BATCH_SIZE, n_items, rng)
            u_t = torch.LongTensor(u).to(device)
            p_t = torch.LongTensor(p).to(device)
            n_t = torch.LongTensor(n).to(device)
            ue, ie = model(adj)
            pos_s = (ue[u_t] * ie[p_t]).sum(1)
            neg_s = (ue[u_t] * ie[n_t]).sum(1)
            bpr = -F.logsigmoid(pos_s - neg_s).mean()
            reg_l = reg * (
                model.user_emb(u_t).norm(2).pow(2)
                + model.item_emb(p_t).norm(2).pow(2)
                + model.item_emb(n_t).norm(2).pow(2)
            ) / BATCH_SIZE
            loss = bpr + reg_l
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"    ep{epoch+1}/{EPOCHS} loss={epoch_loss/n_batches:.6f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    return ue.cpu().numpy(), ie.cpu().numpy(), round(time.time() - t0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    gpu_id = args.gpu_id
    my_configs = ALL_CONFIGS[gpu_id::4]  # round-robin assignment

    out_dir = ensure_dir(ROOT / "artifacts/lightgcn_hparam_sweep")
    out_json = out_dir / f"gpu{gpu_id}_results.json"

    print(f"[GPU{gpu_id}] device={device} configs={len(my_configs)}", flush=True)
    for e, l, r in my_configs:
        print(f"  emb{e} L{l} reg{r:.0e}", flush=True)

    # Build split matrices once
    split_data = {}
    for sname in SPLITS:
        split_path = ROOT / "artifacts/validation" / sname
        train_split = pd.read_csv(split_path / "train_interactions.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(train_split, binary=True)
        split_data[sname] = dict(mat=mat, u2i=u2i, i2i=i2i,
                                  n_users=len(users), n_items=len(items))
        print(f"  [GPU{gpu_id}] {sname}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    results = []
    for ci, (emb_dim, n_layers, reg) in enumerate(my_configs):
        tag = f"emb{emb_dim}_L{n_layers}_reg{reg:.0e}"
        print(f"\n[GPU{gpu_id}] [{ci+1}/{len(my_configs)}] {tag}", flush=True)
        row = {"tag": tag, "emb_dim": emb_dim, "n_layers": n_layers, "reg": reg, "gpu_id": gpu_id}
        split_scores = []
        for sname in SPLITS:
            sd = split_data[sname]
            print(f"  [GPU{gpu_id}] split={sname}", flush=True)
            ue, ie, elapsed = train_lgcn(
                sd["mat"], sd["n_users"], sd["n_items"],
                emb_dim, n_layers, reg, device
            )
            acc = eval_split(sname, ue, ie, sd["u2i"], sd["i2i"])
            row[sname] = round(acc, 5)
            split_scores.append(acc)
            delta = acc - BASELINE[sname]
            print(f"  [GPU{gpu_id}] acc={acc:.5f} baseline={BASELINE[sname]:.5f} "
                  f"Δ={delta:+.5f} elapsed={elapsed}s", flush=True)
        row["mean_val"] = round(sum(split_scores) / len(split_scores), 5)
        row["mean_delta"] = round(row["mean_val"] - sum(BASELINE.values()) / len(BASELINE), 5)
        results.append(row)
        write_json(out_json, results)
        print(f"  [GPU{gpu_id}] mean_val={row['mean_val']:.5f} Δ={row['mean_delta']:+.5f}", flush=True)

    print(f"\n[GPU{gpu_id}] DONE — saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
