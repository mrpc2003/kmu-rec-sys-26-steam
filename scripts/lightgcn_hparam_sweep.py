"""LightGCN hyperparameter sweep — validation-only, no submission.

Sweeps:
  emb_dim: 64 (baseline), 128, 256
  n_layers: 2, 3 (baseline), 4
  reg: 1e-4 (baseline), 1e-3

Evaluates on 3 validation splits (random_sqrtpop, recent_sqrtpop, random_popbin).
Reports JSON + Markdown. No Kaggle submission.
"""
from __future__ import annotations

import json
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (
    DEFAULT_DATA_DIR, build_user_item_matrix, load_train_json,
    load_pairs_csv, ensure_dir, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

# ── validation splits ──────────────────────────────────────────────────────────
SPLIT_DIR = ROOT / "artifacts/validation"
SPLITS = [
    "val_random_sqrtpop_seed42",
    "val_recent_sqrtpop_seed42",
    "val_random_popbin_seed42",
]

# ── sweep grid ─────────────────────────────────────────────────────────────────
SWEEP = list(itertools.product(
    [64, 128, 256],   # emb_dim
    [2, 3, 4],        # n_layers
    [1e-4, 1e-3],     # reg
))
EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
SEED = 42
DEVICE = "cuda:0"

OUT_DIR = ensure_dir(ROOT / "artifacts/lightgcn_hparam_sweep")
OUT_JSON = ROOT / "reports/20260530_lightgcn_hparam_sweep.json"
OUT_MD   = ROOT / "reports/20260530_lightgcn_hparam_sweep.md"

# ── helpers ────────────────────────────────────────────────────────────────────

def eval_split(split_name: str, user_np, item_np, user_to_idx, item_to_idx) -> float:
    split_path = SPLIT_DIR / split_name
    val_pairs = pd.read_csv(split_path / "val_pairs.csv")
    scores = score_candidates(val_pairs, user_np, item_np, user_to_idx, item_to_idx)
    val_pairs = val_pairs.copy()
    val_pairs["score"] = scores
    # per-user top-half
    correct = 0
    total = 0
    for uid, grp in val_pairs.groupby("userID"):
        n = len(grp)
        k = n // 2
        if k == 0:
            continue
        top_ids = grp.nlargest(k, "score")["ID"].values
        labels = grp.set_index("ID")["Label"]
        correct += int(labels.loc[top_ids].sum())
        total += k
    return correct / total if total > 0 else 0.0


def train_lgcn(mat, n_users, n_items, emb_dim, n_layers, reg, epochs, device):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    n_batches = max(1, mat.nnz // BATCH_SIZE)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(mat, BATCH_SIZE, n_items, rng)
            u_t = torch.LongTensor(u).to(device)
            p_t = torch.LongTensor(p).to(device)
            n_t = torch.LongTensor(n).to(device)
            user_emb, item_emb = model(adj)
            pos_s = (user_emb[u_t] * item_emb[p_t]).sum(1)
            neg_s = (user_emb[u_t] * item_emb[n_t]).sum(1)
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
            print(f"  epoch {epoch+1}/{epochs} loss={epoch_loss/n_batches:.6f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    return ue.cpu().numpy(), ie.cpu().numpy(), round(time.time() - t0, 1)


def main():
    print("Loading train data...", flush=True)
    train_df = load_train_json(DEFAULT_DATA_DIR / "train.json")

    # Build per-split matrices
    split_data = {}
    for sname in SPLITS:
        split_path = SPLIT_DIR / sname
        train_split = pd.read_csv(split_path / "train_interactions.csv")
        mat, u2i, i2i, users, items = build_user_item_matrix(train_split, binary=True)
        split_data[sname] = dict(mat=mat, u2i=u2i, i2i=i2i,
                                  n_users=len(users), n_items=len(items))
        print(f"  {sname}: {len(users)} users, {len(items)} items, {mat.nnz} nnz", flush=True)

    # Baseline: emb64 L3 reg1e-4 (already known from prior run)
    BASELINE = {
        "val_random_sqrtpop_seed42": 0.6748,
        "val_recent_sqrtpop_seed42": 0.6396,
        "val_random_popbin_seed42":  0.6020,
    }

    results = []
    total_configs = len(SWEEP)
    for ci, (emb_dim, n_layers, reg) in enumerate(SWEEP):
        tag = f"emb{emb_dim}_L{n_layers}_reg{str(reg).replace('-','m').replace('.','p')}"
        # Skip baseline (already done)
        if emb_dim == 64 and n_layers == 3 and reg == 1e-4:
            row = {"tag": tag, "emb_dim": emb_dim, "n_layers": n_layers, "reg": reg,
                   "skipped": True, "note": "baseline — already evaluated"}
            for s in SPLITS:
                row[s] = BASELINE[s]
            row["mean_val"] = round(sum(BASELINE[s] for s in SPLITS) / len(SPLITS), 5)
            results.append(row)
            print(f"\n[{ci+1}/{total_configs}] {tag} — SKIPPED (baseline)", flush=True)
            continue

        print(f"\n[{ci+1}/{total_configs}] {tag}", flush=True)
        row = {"tag": tag, "emb_dim": emb_dim, "n_layers": n_layers, "reg": reg}
        split_scores = []
        for sname in SPLITS:
            sd = split_data[sname]
            print(f"  split={sname}", flush=True)
            ue, ie, elapsed = train_lgcn(
                sd["mat"], sd["n_users"], sd["n_items"],
                emb_dim, n_layers, reg, EPOCHS, DEVICE
            )
            acc = eval_split(sname, ue, ie, sd["u2i"], sd["i2i"])
            row[sname] = round(acc, 5)
            split_scores.append(acc)
            print(f"  → acc={acc:.5f} (baseline={BASELINE[sname]:.5f} "
                  f"Δ={acc-BASELINE[sname]:+.5f}) elapsed={elapsed}s", flush=True)
        row["mean_val"] = round(sum(split_scores) / len(split_scores), 5)
        row["mean_delta"] = round(row["mean_val"] - sum(BASELINE.values()) / len(BASELINE), 5)
        results.append(row)

        # Save intermediate
        write_json(OUT_JSON, results)
        print(f"  mean_val={row['mean_val']:.5f} mean_delta={row['mean_delta']:+.5f}", flush=True)

    # Final report
    write_json(OUT_JSON, results)

    # Markdown
    md = ["# LightGCN Hyperparameter Sweep\n"]
    md.append("| tag | emb_dim | n_layers | reg | sqrtpop | recent | popbin | mean | Δmean |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(results, key=lambda x: -x.get("mean_val", 0)):
        skipped = "✓baseline" if r.get("skipped") else ""
        md.append(
            f"| {r['tag']} {skipped} | {r['emb_dim']} | {r['n_layers']} | {r['reg']:.0e} "
            f"| {r.get('val_random_sqrtpop_seed42', 0):.5f} "
            f"| {r.get('val_recent_sqrtpop_seed42', 0):.5f} "
            f"| {r.get('val_random_popbin_seed42', 0):.5f} "
            f"| {r.get('mean_val', 0):.5f} "
            f"| {r.get('mean_delta', 0):+.5f} |"
        )
    OUT_MD.write_text("\n".join(md))
    print(f"\n== SWEEP DONE ==")
    print(f"saved: {OUT_JSON}")
    print(f"saved: {OUT_MD}")
    best = max(results, key=lambda x: x.get("mean_val", 0))
    print(f"best config: {best['tag']} mean_val={best['mean_val']:.5f}")


if __name__ == "__main__":
    main()
