#!/usr/bin/env python3
"""Validation-only SL@K-lite continuation probe for KMU Steam played prediction.

This script tests a structurally different objective lever after CF/LightGCN
saturation: continue one shared LightGCN checkpoint with either the old BPR loss
(control) or a hybrid BPR + soft top-negative/listwise SL@K-lite loss (variant).

Safety contract:
- validation-only: reads only split train_interactions.csv + candidates.csv with labels
- never reads hidden test pairs
- never writes Kaggle candidate/submission CSVs
- never calls Kaggle

The comparison is paired at every level: same split, same initial checkpoint,
same sampled users/positives/negatives, same continuation budget, same evaluator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lightgcn_train import LightGCN, build_norm_adj, score_candidates  # noqa: E402
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
)


def predict_tophalf_array(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Return validation predictions as an array in df row order."""
    pred = np.zeros(len(df), dtype=np.int8)
    for _, idx in df.groupby("userID", sort=False).groups.items():
        ids = np.asarray(list(idx))
        k = int(df.loc[ids, "Label"].sum())
        scores = df.loc[ids, score_col].to_numpy()
        row_ids = df.loc[ids, "ID"].to_numpy()
        order = np.lexsort((row_ids, -scores))
        pred[ids[order[:k]]] = 1
    return pred


def mcnemar_variant_vs_control(candidates: pd.DataFrame, variant_col: str, control_col: str) -> dict[str, object]:
    """Evaluate variant against same-budget control with paired McNemar diagnostics."""
    variant_summary, _ = evaluate_tophalf(candidates, variant_col, label_col="Label", user_col="userID", id_col="ID")
    control_summary, _ = evaluate_tophalf(candidates, control_col, label_col="Label", user_col="userID", id_col="ID")
    y = candidates["Label"].to_numpy(dtype=np.int8)
    variant_pred = predict_tophalf_array(candidates, variant_col)
    control_pred = predict_tophalf_array(candidates, control_col)
    control_ok = control_pred == y
    variant_ok = variant_pred == y
    b = int(np.sum(control_ok & ~variant_ok))
    c = int(np.sum(~control_ok & variant_ok))
    if b + c == 0:
        p = 1.0
    else:
        chi2 = (max(abs(b - c) - 1, 0) ** 2) / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2.0))
    variant_acc = float(cast(float, variant_summary["row_accuracy"]))
    control_acc = float(cast(float, control_summary["row_accuracy"]))
    return {
        "variant_score": variant_col,
        "control_score": control_col,
        "variant_accuracy": variant_acc,
        "control_accuracy": control_acc,
        "delta_vs_control": variant_acc - control_acc,
        "flips": int(np.sum(variant_pred != control_pred)),
        "variant_fixes": c,
        "control_breaks": b,
        "mcnemar_p": float(p),
    }


def build_user_positive_lists(interaction_matrix) -> tuple[list[np.ndarray], list[set[int]]]:
    lists: list[np.ndarray] = []
    sets: list[set[int]] = []
    for u in range(interaction_matrix.shape[0]):
        start, end = interaction_matrix.indptr[u], interaction_matrix.indptr[u + 1]
        arr = interaction_matrix.indices[start:end].astype(np.int64, copy=True)
        lists.append(arr)
        sets.append(set(int(x) for x in arr.tolist()))
    return lists, sets


def draw_negative(rng: np.random.Generator, n_items: int, positive_set: set[int]) -> int:
    neg = int(rng.integers(0, n_items))
    # Steam split users are sparse; rejection is cheap. Guard pathological cases.
    guard = 0
    while neg in positive_set and guard < 10_000:
        neg = int(rng.integers(0, n_items))
        guard += 1
    if guard >= 10_000:
        raise RuntimeError("failed to draw an unobserved negative item")
    return neg


def sample_shared_batch(
    user_positive_lists: list[np.ndarray],
    user_positive_sets: list[set[int]],
    batch_size: int,
    n_items: int,
    n_slk_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one paired continuation batch.

    Returns users, positives, and a [batch, n_slk_negatives] negative matrix.
    The first negative column is also used as the BPR negative for both arms.
    """
    n_users = len(user_positive_lists)
    users = rng.integers(0, n_users, size=batch_size, dtype=np.int64)
    pos_items = np.empty(batch_size, dtype=np.int64)
    neg_items = np.empty((batch_size, n_slk_negatives), dtype=np.int64)
    for row, u_raw in enumerate(users):
        u = int(u_raw)
        pos_pool = user_positive_lists[u]
        if pos_pool.size == 0:
            pos_items[row] = int(rng.integers(0, n_items))
            positive_set: set[int] = set()
        else:
            pos_items[row] = int(rng.choice(pos_pool))
            positive_set = user_positive_sets[u]
        for j in range(n_slk_negatives):
            neg_items[row, j] = draw_negative(rng, n_items, positive_set)
    return users, pos_items, neg_items


def bpr_loss_for_batch(
    model: LightGCN,
    adj: torch.Tensor,
    users_t: torch.Tensor,
    pos_t: torch.Tensor,
    neg_t: torch.Tensor,
    reg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    user_emb, item_emb = model(adj)
    u_emb = user_emb[users_t]
    p_emb = item_emb[pos_t]
    n_emb = item_emb[neg_t]
    pos_scores = (u_emb * p_emb).sum(dim=1)
    neg_scores = (u_emb * n_emb).sum(dim=1)
    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    batch_size = users_t.shape[0]
    reg_loss = reg * (
        model.user_emb(users_t).norm(2).pow(2)
        + model.item_emb(pos_t).norm(2).pow(2)
        + model.item_emb(neg_t).norm(2).pow(2)
    ) / batch_size
    loss = bpr_loss + reg_loss
    return loss, {"bpr": float(bpr_loss.detach().cpu()), "reg": float(reg_loss.detach().cpu())}


def hybrid_slk_loss_for_batch(
    model: LightGCN,
    adj: torch.Tensor,
    users_t: torch.Tensor,
    pos_t: torch.Tensor,
    neg_matrix_t: torch.Tensor,
    reg: float,
    alpha: float,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    user_emb, item_emb = model(adj)
    u_emb = user_emb[users_t]
    p_emb = item_emb[pos_t]
    neg_emb = item_emb[neg_matrix_t]  # [B, M, D]
    pos_scores = (u_emb * p_emb).sum(dim=1)
    neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)
    bpr_loss = -F.logsigmoid(pos_scores - neg_scores[:, 0]).mean()

    # Soft top-negative/listwise term. Detaching weights makes this a stable
    # top-negative expectation rather than a temperature-sensitive second-order
    # softmax optimizer.
    neg_weights = torch.softmax(neg_scores / tau, dim=1).detach()
    hard_neg_score = (neg_weights * neg_scores).sum(dim=1)
    slk_loss = F.softplus(hard_neg_score - pos_scores).mean()

    batch_size = users_t.shape[0]
    reg_loss = reg * (
        model.user_emb(users_t).norm(2).pow(2)
        + model.item_emb(pos_t).norm(2).pow(2)
        + model.item_emb(neg_matrix_t[:, 0]).norm(2).pow(2)
    ) / batch_size
    loss = alpha * bpr_loss + (1.0 - alpha) * slk_loss + reg_loss
    return loss, {
        "bpr": float(bpr_loss.detach().cpu()),
        "slk": float(slk_loss.detach().cpu()),
        "reg": float(reg_loss.detach().cpu()),
    }


def train_bpr_pretrain(
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
) -> tuple[LightGCN, torch.Tensor, dict[str, object]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adj = build_norm_adj(interaction_matrix, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    user_pos_lists, user_pos_sets = build_user_positive_lists(interaction_matrix)
    n_batches = max(1, interaction_matrix.nnz // batch_size)
    started = time.time()
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_matrix = sample_shared_batch(
                user_pos_lists, user_pos_sets, batch_size, n_items, 1, rng
            )
            users_t = torch.as_tensor(users, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos_items, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg_matrix[:, 0], dtype=torch.long, device=device)
            loss, _ = bpr_loss_for_batch(model, adj, users_t, pos_t, neg_t, reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [SLK pretrain BPR] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} elapsed={time.time()-started:.1f}s", flush=True)
    meta = {
        "emb_dim": emb_dim,
        "n_layers": n_layers,
        "lr": lr,
        "reg": reg,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(interaction_matrix.nnz),
        "final_loss": float(losses[-1]),
        "train_seconds": round(time.time() - started, 1),
        "device": device,
    }
    return model, adj, meta


def score_model(model: LightGCN, adj: torch.Tensor, candidates: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj)
    return score_candidates(
        candidates,
        user_emb.detach().cpu().numpy().astype(np.float32),
        item_emb.detach().cpu().numpy().astype(np.float32),
        user_to_idx,
        item_to_idx,
    )


def parse_eval_epochs(raw: str, max_epoch: int) -> list[int]:
    epochs = sorted({int(x) for x in raw.split(",") if x.strip()})
    if not epochs:
        raise ValueError("--eval-epochs cannot be empty")
    if epochs[-1] > max_epoch:
        raise ValueError("eval epoch exceeds --cont-epochs")
    if epochs[0] <= 0:
        raise ValueError("eval epochs must be positive")
    return epochs


def run(args: argparse.Namespace) -> dict[str, object]:
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv").copy()
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    print(
        f"[SLK-lite probe] split={split_dir.name} users={len(users)} items={len(items)} interactions={mat.nnz}",
        flush=True,
    )

    pre_model, adj, pretrain_meta = train_bpr_pretrain(
        mat,
        len(users),
        len(items),
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    eval_epochs = parse_eval_epochs(args.eval_epochs, args.cont_epochs)
    start_state = copy.deepcopy(pre_model.state_dict())

    control_model = LightGCN(len(users), len(items), args.emb_dim, args.n_layers).to(args.device)
    variant_model = LightGCN(len(users), len(items), args.emb_dim, args.n_layers).to(args.device)
    control_model.load_state_dict(start_state)
    variant_model.load_state_dict(start_state)
    control_optimizer = torch.optim.Adam(control_model.parameters(), lr=args.cont_lr, weight_decay=0)
    variant_optimizer = torch.optim.Adam(variant_model.parameters(), lr=args.cont_lr, weight_decay=0)

    user_pos_lists, user_pos_sets = build_user_positive_lists(mat)
    cont_rng = np.random.default_rng(args.seed + args.cont_seed_offset)
    n_batches = max(1, mat.nnz // args.batch_size)

    candidates["score_frozen_checkpoint"] = score_model(pre_model, adj, candidates, user_to_idx, item_to_idx)
    frozen_summary, _ = evaluate_tophalf(candidates, "score_frozen_checkpoint", label_col="Label", user_col="userID", id_col="ID")

    records: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, args.cont_epochs + 1):
        control_model.train()
        variant_model.train()
        control_loss_sum = 0.0
        variant_loss_sum = 0.0
        variant_bpr_sum = 0.0
        variant_slk_sum = 0.0
        for _ in range(n_batches):
            users_b, pos_b, neg_matrix = sample_shared_batch(
                user_pos_lists,
                user_pos_sets,
                args.batch_size,
                len(items),
                args.slk_negatives,
                cont_rng,
            )
            users_t = torch.as_tensor(users_b, dtype=torch.long, device=args.device)
            pos_t = torch.as_tensor(pos_b, dtype=torch.long, device=args.device)
            neg_matrix_t = torch.as_tensor(neg_matrix, dtype=torch.long, device=args.device)
            neg_t = neg_matrix_t[:, 0]

            control_loss, _ = bpr_loss_for_batch(control_model, adj, users_t, pos_t, neg_t, args.reg)
            control_optimizer.zero_grad()
            control_loss.backward()
            control_optimizer.step()
            control_loss_sum += float(control_loss.detach().cpu())

            variant_loss, variant_parts = hybrid_slk_loss_for_batch(
                variant_model,
                adj,
                users_t,
                pos_t,
                neg_matrix_t,
                reg=args.reg,
                alpha=args.alpha,
                tau=args.tau,
            )
            variant_optimizer.zero_grad()
            variant_loss.backward()
            variant_optimizer.step()
            variant_loss_sum += float(variant_loss.detach().cpu())
            variant_bpr_sum += variant_parts["bpr"]
            variant_slk_sum += variant_parts["slk"]

        print(
            "  [SLK continuation] "
            f"epoch {epoch}/{args.cont_epochs} "
            f"control_loss={control_loss_sum/n_batches:.6f} "
            f"variant_loss={variant_loss_sum/n_batches:.6f} "
            f"variant_bpr={variant_bpr_sum/n_batches:.6f} "
            f"variant_slk={variant_slk_sum/n_batches:.6f} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )

        if epoch in eval_epochs:
            control_col = f"score_control_bpr_e{epoch}"
            variant_col = f"score_slk_hybrid_e{epoch}"
            candidates[control_col] = score_model(control_model, adj, candidates, user_to_idx, item_to_idx)
            candidates[variant_col] = score_model(variant_model, adj, candidates, user_to_idx, item_to_idx)
            rec = mcnemar_variant_vs_control(candidates, variant_col, control_col)
            frozen_acc = float(cast(float, frozen_summary["row_accuracy"]))
            control_acc = float(cast(float, rec["control_accuracy"]))
            variant_acc = float(cast(float, rec["variant_accuracy"]))
            delta_vs_control = float(cast(float, rec["delta_vs_control"]))
            rec.update(
                {
                    "epoch": epoch,
                    "frozen_accuracy": frozen_acc,
                    "control_delta_vs_frozen": control_acc - frozen_acc,
                    "variant_delta_vs_frozen": variant_acc - frozen_acc,
                    "alpha": args.alpha,
                    "tau": args.tau,
                    "slk_negatives": args.slk_negatives,
                }
            )
            records.append(rec)
            print(
                "    [eval] "
                f"epoch={epoch} frozen={frozen_acc:.6f} "
                f"control={control_acc:.6f} "
                f"variant={variant_acc:.6f} "
                f"delta_vs_control={delta_vs_control:+.6f} "
                f"fixes/breaks={rec['variant_fixes']}/{rec['control_breaks']} "
                f"p={rec['mcnemar_p']:.4f}",
                flush=True,
            )

    score_cols = ["ID", "userID", "gameID", "Label", "score_frozen_checkpoint"]
    for rec in records:
        score_cols.append(str(rec["control_score"]))
        score_cols.append(str(rec["variant_score"]))
    # Validation score diagnostics only. This is not a Kaggle candidate/test CSV.
    candidates[score_cols].to_csv(out_dir / "slk_lite_validation_scores.csv", index=False)

    best = max(records, key=lambda r: float(cast(float, r["delta_vs_control"]))) if records else None
    if best and float(cast(float, best["delta_vs_control"])) >= args.single_split_weak_delta and float(cast(float, best["mcnemar_p"])) < 0.05:
        gate = "WEAK_SIGNAL_PANEL_ONLY"
    elif best and float(cast(float, best["delta_vs_control"])) > 0:
        gate = "TINY_POSITIVE_PANEL_ONLY"
    else:
        gate = "REJECT"

    summary = {
        "safety": {"validation_only": True, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "split": split_dir.name,
        "args": vars(args),
        "pretrain_meta": pretrain_meta,
        "frozen_summary": frozen_summary,
        "records": records,
        "best_record": best,
        "gate": gate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# SL@K-lite continuation probe — {split_dir.name}\n\n",
        "**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false\n\n",
        f"Pretrain: LightGCN emb={args.emb_dim}, layers={args.n_layers}, reg={args.reg}, epochs={args.pretrain_epochs}, seed={args.seed}.\n\n",
        f"Continuation: old BPR control vs alpha={args.alpha:.2f} BPR + {(1.0-args.alpha):.2f} SL@K-lite, tau={args.tau}, negatives/user={args.slk_negatives}.\n\n",
        "| epoch | frozen acc | control acc | variant acc | Δ variant-control | flips | fixes | breaks | McNemar p |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for rec in records:
        lines.append(
            f"| {rec['epoch']} | {rec['frozen_accuracy']:.6f} | {rec['control_accuracy']:.6f} | {rec['variant_accuracy']:.6f} | "
            f"{rec['delta_vs_control']:+.6f} | {rec['flips']} | {rec['variant_fixes']} | {rec['control_breaks']} | {rec['mcnemar_p']:.4f} |\n"
        )
    if best:
        lines.append(
            f"\n## Gate verdict\n**{gate}** — best epoch {best['epoch']} delta_vs_control={best['delta_vs_control']:+.6f}, p={best['mcnemar_p']:.4f}.\n"
        )
    (out_dir / "slk_lite_probe_report.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "best_record": best, "gate": gate}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", default=str(ROOT / "artifacts/validation/val_random_uniform_seed42"))
    p.add_argument("--out-dir", default="artifacts/slk_lite_probe/emb128_L4_r3_seed42")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cont-lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--cont-epochs", type=int, default=2)
    p.add_argument("--eval-epochs", default="1,2")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--slk-negatives", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cont-seed-offset", type=int, default=777)
    p.add_argument("--single-split-weak-delta", type=float, default=0.0015)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
