#!/usr/bin/env python3
"""Validation-only LightGCN++-style layer-mixture rescoring probe.

Goal: test whether the uniform LightGCN layer average h0..hK is a small but
avoidable constraint for the per-user top-half validation gate. This is a cheap
propagation-weight / inference-operator probe, not a new encoder campaign.

Safety: validation-only. The script never reads the hidden test pairs, never
writes a Kaggle candidate/submission CSV, and never calls Kaggle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates
from recsys_played_utils import (
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
)


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


def mcnemar_vs_base(candidates: pd.DataFrame, score_col: str, base_pred: np.ndarray) -> dict[str, object]:
    summary, _ = evaluate_tophalf(candidates, score_col, label_col="Label", user_col="userID", id_col="ID")
    y = candidates["Label"].to_numpy(dtype=np.int8)
    pred = predict_tophalf(candidates, score_col)
    base_ok = base_pred == y
    cand_ok = pred == y
    b = int(np.sum(base_ok & ~cand_ok))
    c = int(np.sum(~base_ok & cand_ok))
    if b + c == 0:
        p = 1.0
    else:
        chi2 = (max(abs(b - c) - 1, 0) ** 2) / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2.0))
    row_accuracy = float(summary["row_accuracy"])
    base_accuracy = float((base_pred == y).mean())
    return {
        "score": score_col,
        "row_accuracy": row_accuracy,
        "delta_vs_base": row_accuracy - base_accuracy,
        "flips": int(np.sum(pred != base_pred)),
        "candidate_fixes": c,
        "base_breaks": b,
        "mcnemar_p": float(p),
    }


def train_lightgcn_layer_embeddings(
    interaction_matrix,
    n_users: int,
    n_items: int,
    emb_dim: int,
    n_layers: int,
    lr: float,
    reg: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, object]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    adj = build_norm_adj(interaction_matrix, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    n_interactions = interaction_matrix.nnz
    n_batches = max(1, n_interactions // batch_size)
    losses: list[float] = []
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
            bpr_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
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
            print(f"  [LayerMix] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} elapsed={time.time()-started:.1f}s", flush=True)

    model.eval()
    with torch.no_grad():
        ego = model.get_ego_embeddings()
        all_embs = [ego]
        cur = ego
        for _ in range(n_layers):
            cur = torch.sparse.mm(adj, cur)
            all_embs.append(cur)
        user_layers = [x[:n_users].detach().cpu().numpy().astype(np.float32) for x in all_embs]
        item_layers = [x[n_users:].detach().cpu().numpy().astype(np.float32) for x in all_embs]

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
    return user_layers, item_layers, meta


def normalize_weights(weights: list[float], n: int) -> np.ndarray:
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    w = np.asarray(weights, dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("negative layer weights are not allowed in this probe")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("layer weights sum to zero")
    return w / s


def default_mixtures(n_layers: int) -> Dict[str, np.ndarray]:
    n = n_layers + 1
    if n_layers != 4:
        # Generic fallback: simple diagnostics around uniform.
        uniform = np.full(n, 1.0 / n, dtype=np.float32)
        decay = np.asarray([0.5 ** k for k in range(n)], dtype=np.float32)
        decay = decay / decay.sum()
        late = decay[::-1] / decay[::-1].sum()
        no_ego = np.r_[0.0, np.full(n - 1, 1.0 / (n - 1), dtype=np.float32)].astype(np.float32)
        return {"uniform": uniform, "decay_half": decay, "late_half": late, "no_ego": no_ego}
    raw = {
        "uniform": [0.20, 0.20, 0.20, 0.20, 0.20],
        "ego_heavy": [0.40, 0.25, 0.15, 0.10, 0.10],
        "shallow_heavy": [0.10, 0.35, 0.25, 0.20, 0.10],
        "mid_heavy": [0.10, 0.20, 0.35, 0.25, 0.10],
        "late_heavy": [0.05, 0.10, 0.20, 0.30, 0.35],
        "no_ego": [0.00, 0.25, 0.25, 0.25, 0.25],
        "no_deep": [0.30, 0.30, 0.20, 0.15, 0.05],
        "decay_half": [0.516, 0.258, 0.129, 0.065, 0.032],
        "tail_cut": [0.20, 0.25, 0.25, 0.20, 0.10],
    }
    return {name: normalize_weights(w, n) for name, w in raw.items()}


def mix_layers(layers: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(layers[0], dtype=np.float32)
    for w, layer in zip(weights, layers):
        out += float(w) * layer
    return out.astype(np.float32)


def add_z_blend(df: pd.DataFrame, base_col: str, alt_col: str, out_col: str, lam: float) -> None:
    g = df.groupby("userID")
    z_base = (df[base_col] - g[base_col].transform("mean")) / g[base_col].transform("std").replace(0, np.nan).fillna(1.0)
    z_alt = (df[alt_col] - g[alt_col].transform("mean")) / g[alt_col].transform("std").replace(0, np.nan).fillna(1.0)
    df[out_col] = (1.0 - lam) * z_base + lam * z_alt


def run(args: argparse.Namespace) -> Dict[str, object]:
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    print(f"[LayerMix probe] split={split_dir.name} users={len(users)} items={len(items)} interactions={mat.nnz}", flush=True)

    user_layers, item_layers, train_meta = train_lightgcn_layer_embeddings(
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

    mixtures = default_mixtures(args.n_layers)
    candidates = candidates.copy()
    results: list[dict[str, object]] = []

    base_col = "score_layermix_uniform"
    uniform_weights = mixtures["uniform"]
    candidates[base_col] = score_candidates(
        candidates,
        mix_layers(user_layers, uniform_weights),
        mix_layers(item_layers, uniform_weights),
        user_to_idx,
        item_to_idx,
    )
    base_pred = predict_tophalf(candidates, base_col)
    results.append(mcnemar_vs_base(candidates, base_col, base_pred))

    for name, weights in mixtures.items():
        if name == "uniform":
            continue
        u = mix_layers(user_layers, weights)
        i = mix_layers(item_layers, weights)
        col = f"score_layermix_{name}"
        candidates[col] = score_candidates(candidates, u, i, user_to_idx, item_to_idx)
        results.append(mcnemar_vs_base(candidates, col, base_pred))

    # Conservative fixed residual z-blends against the uniform baseline.
    blend_lams = [float(x) for x in args.blend_lambdas.split(",") if x]
    for name in mixtures:
        if name == "uniform":
            continue
        alt_col = f"score_layermix_{name}"
        for lam in blend_lams:
            out_col = f"score_blend_{name}_l{str(lam).replace('.', 'p')}"
            add_z_blend(candidates, base_col, alt_col, out_col, lam)
            results.append(mcnemar_vs_base(candidates, out_col, base_pred))

    results = sorted(
        results,
        key=lambda r: (float(cast(float, r["row_accuracy"])), float(cast(float, r["delta_vs_base"]))),
        reverse=True,
    )
    best_nonbase = next(r for r in results if r["score"] != base_col)
    best_delta = float(cast(float, best_nonbase["delta_vs_base"]))
    best_p = float(cast(float, best_nonbase["mcnemar_p"]))
    if best_delta >= args.mde and best_p < 0.05:
        gate = "ESCALATE_3SPLIT"
    elif best_delta > 0 and best_p < 0.05:
        gate = "WEAK_SIGNAL_PANEL_ONLY"
    else:
        gate = "REJECT"

    score_cols = ["ID", "userID", "gameID", "Label", base_col] + [r["score"] for r in results if r["score"] != base_col]
    # Preserve only validation score diagnostics. This is not a candidate/test CSV.
    candidates[score_cols].to_csv(out_dir / "layermix_validation_scores.csv", index=False)

    summary = {
        "safety": {"validation_only": True, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "split": split_dir.name,
        "args": vars(args),
        "train_meta": train_meta,
        "base_score": base_col,
        "mixtures": {k: [float(x) for x in v] for k, v in mixtures.items()},
        "results": results,
        "best_nonbase": best_nonbase,
        "gate": gate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# LightGCN++-style layer-mixture probe — {split_dir.name}\n",
        "**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false\n\n",
        f"Baseline: `{base_col}` = uniform h0..hK average. emb_dim={args.emb_dim}, layers={args.n_layers}, reg={args.reg}, seed={args.seed}, epochs={args.epochs}.\n\n",
        "| score | acc | delta_vs_base | flips | fixes | breaks | McNemar p |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in results[:18]:
        lines.append(
            f"| {r['score']} | {r['row_accuracy']:.5f} | {r['delta_vs_base']:+.5f} | {r['flips']} | {r['candidate_fixes']} | {r['base_breaks']} | {r['mcnemar_p']:.4f} |\n"
        )
    lines.append(f"\n## Gate verdict\n**{gate}** — best non-baseline `{best_nonbase['score']}` delta={best_nonbase['delta_vs_base']:+.5f}, p={best_nonbase['mcnemar_p']:.4f}.\n")
    (out_dir / "layermix_probe_report.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "best_nonbase": best_nonbase, "gate": gate}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default=str(ROOT / "artifacts/validation/val_random_uniform_seed42"))
    p.add_argument("--out-dir", default="artifacts/layermix_probe/emb128_L4_r3_seed42")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blend-lambdas", default="0.25")
    p.add_argument("--mde", type=float, default=0.00355)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
