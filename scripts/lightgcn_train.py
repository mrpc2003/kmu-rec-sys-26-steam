#!/usr/bin/env python3
"""LightGCN (GPU) for KMU RecSys 26 Steam played prediction.

Trains a LightGCN model on the user-item interaction graph using PyTorch + CUDA,
then scores validation candidates. Validation-only; no Kaggle submission.

References:
- He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation", SIGIR 2020.
- LightGCL (ICLR 2023) uses SVD-based contrastive augmentation on top of LightGCN.

This script implements vanilla LightGCN with BPR loss as the first GPU-powered
score axis for this project.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (
    DEFAULT_DATA_DIR,
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    write_json,
)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, n_layers: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def get_ego_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ego = self.get_ego_embeddings()
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego)
            all_embs.append(ego)
        stacked = torch.stack(all_embs, dim=1)
        final = stacked.mean(dim=1)
        user_final = final[: self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final


def build_norm_adj(interaction_matrix: sp.csr_matrix, n_users: int, n_items: int) -> torch.Tensor:
    """Build symmetric normalized adjacency for LightGCN."""
    R = interaction_matrix.tocoo().astype(np.float32)
    n = n_users + n_items
    # Build adjacency: [[0, R], [R^T, 0]]
    row = np.concatenate([R.row, R.col + n_users])
    col = np.concatenate([R.col + n_users, R.row])
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(n, n))

    # D^{-1/2} A D^{-1/2}
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()

    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size([n, n]))


def sample_bpr_batch(
    interaction_matrix: sp.csr_matrix,
    batch_size: int,
    n_items: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample BPR triplets (user, pos_item, neg_item)."""
    users = rng.integers(0, interaction_matrix.shape[0], size=batch_size)
    pos_items = np.zeros(batch_size, dtype=np.int64)
    neg_items = np.zeros(batch_size, dtype=np.int64)
    for i, u in enumerate(users):
        start, end = interaction_matrix.indptr[u], interaction_matrix.indptr[u + 1]
        if start == end:
            pos_items[i] = rng.integers(0, n_items)
            neg_items[i] = rng.integers(0, n_items)
            continue
        pos_pool = interaction_matrix.indices[start:end]
        pos_items[i] = rng.choice(pos_pool)
        neg = rng.integers(0, n_items)
        while neg in pos_pool:
            neg = rng.integers(0, n_items)
        neg_items[i] = neg
    return users, pos_items, neg_items


def train_lightgcn(
    interaction_matrix: sp.csr_matrix,
    n_users: int,
    n_items: int,
    emb_dim: int = 64,
    n_layers: int = 3,
    lr: float = 1e-3,
    reg: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 4096,
    device: str = "cuda:0",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Train LightGCN and return user/item embeddings."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    adj = build_norm_adj(interaction_matrix, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    n_interactions = interaction_matrix.nnz
    n_batches = max(1, n_interactions // batch_size)
    losses = []

    started = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_bpr_batch(interaction_matrix, batch_size, n_items, rng)
            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            user_emb, item_emb = model(adj)
            u_emb = user_emb[users_t]
            p_emb = item_emb[pos_t]
            n_emb = item_emb[neg_t]

            pos_scores = (u_emb * p_emb).sum(dim=1)
            neg_scores = (u_emb * n_emb).sum(dim=1)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            # L2 regularization on ego embeddings
            reg_loss = reg * (
                model.user_emb(users_t).norm(2).pow(2)
                + model.item_emb(pos_t).norm(2).pow(2)
                + model.item_emb(neg_t).norm(2).pow(2)
            ) / batch_size

            loss = bpr_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [LightGCN] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} elapsed={time.time()-started:.1f}s", flush=True)

    # Final embeddings
    model.eval()
    with torch.no_grad():
        user_final, item_final = model(adj)
    user_np = user_final.cpu().numpy()
    item_np = item_final.cpu().numpy()

    meta = {
        "emb_dim": emb_dim,
        "n_layers": n_layers,
        "lr": lr,
        "reg": reg,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "final_loss": float(losses[-1]),
        "train_seconds": round(time.time() - started, 1),
        "device": device,
    }
    return user_np, item_np, meta


def score_candidates(
    candidates: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> np.ndarray:
    scores = np.full(len(candidates), -1e9, dtype=np.float32)
    for n, (uid, gid) in enumerate(candidates[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(str(uid))
        gi = item_to_idx.get(str(gid))
        if ui is None or gi is None:
            continue
        scores[n] = float(np.dot(user_emb[ui], item_emb[gi]))
    return scores


def run_split(
    split_dir: Path,
    out_dir: Path,
    emb_dim: int,
    n_layers: int,
    lr: float,
    reg: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")

    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    n_users, n_items = len(users), len(items)

    print(f"[LightGCN] {split_dir.name}: {n_users} users, {n_items} items, {mat.nnz} interactions", flush=True)

    user_emb, item_emb, train_meta = train_lightgcn(
        mat, n_users, n_items,
        emb_dim=emb_dim, n_layers=n_layers, lr=lr, reg=reg,
        epochs=epochs, batch_size=batch_size, device=device, seed=seed,
    )

    scores = score_candidates(candidates, user_emb, item_emb, user_to_idx, item_to_idx)
    candidates = candidates.copy()
    candidates["score_lightgcn"] = scores

    summary, _ = evaluate_tophalf(candidates, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")

    split_out = ensure_dir(out_dir / split_dir.name)
    candidates[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(split_out / "lightgcn_scores.csv", index=False)

    result = {
        "split": split_dir.name,
        "summary": summary,
        "train_meta": train_meta,
    }
    write_json(split_out / "summary.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-root", default="artifacts/validation")
    ap.add_argument("--splits", nargs="*", default=[
        "val_random_sqrtpop_seed42",
        "val_recent_sqrtpop_seed42",
        "val_random_popbin_seed42",
    ])
    ap.add_argument("--out-dir", default="artifacts/lightgcn_20260530")
    ap.add_argument("--report-json", default="reports/20260530_lightgcn.json")
    ap.add_argument("--report-md", default="reports/20260530_lightgcn.md")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    results = []
    for split in args.splits:
        split_dir = Path(args.validation_root) / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        results.append(run_split(
            split_dir, out_dir,
            emb_dim=args.emb_dim, n_layers=args.n_layers,
            lr=args.lr, reg=args.reg, epochs=args.epochs,
            batch_size=args.batch_size, device=args.device, seed=args.seed,
        ))

    payload = {"note": "LightGCN GPU validation-only. No Kaggle submission.", "results": results}
    write_json(args.report_json, payload)

    lines = [
        "# KMU RecSys 26 Steam — LightGCN (GPU) validation",
        "",
        "PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.",
        "",
        "## Results",
        "",
        "| split | row acc | per-user mean acc | epochs | loss | train sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s = r["summary"]
        m = r["train_meta"]
        lines.append(f"| `{r['split']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} | {m['epochs']} | {m['final_loss']:.6f} | {m['train_seconds']} |")

    lines.extend(["", "## Config", "", f"- emb_dim={args.emb_dim}, n_layers={args.n_layers}, lr={args.lr}, reg={args.reg}", f"- batch_size={args.batch_size}, device={args.device}, seed={args.seed}", ""])
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_md": args.report_md, "report_json": args.report_json, "splits": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
