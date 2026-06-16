#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from lightgcn_train import sample_bpr_batch, score_candidates  # noqa: E402
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    write_json,
)

SPLIT = "val_random_uniform_seed42"
EMB128_SINGLE_REF = 0.76205
EMB128_ENSEMBLE_REF = 0.76505
NOISE = 0.0007


def emb128_uni_path(seed: int, split: str) -> Path:
    if seed == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"


def load_emb128_ensemble(split: str) -> pd.DataFrame | None:
    seeds = [42, 123, 2024, 7]
    paths = [emb128_uni_path(seed, split) for seed in seeds]
    if not all(path.exists() for path in paths):
        return None
    base = pd.read_csv(paths[0])[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "score_emb128_seed42"}
    )
    for seed, path in zip(seeds[1:], paths[1:]):
        scores = pd.read_csv(path)[["ID", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"score_emb128_seed{seed}"}
        )
        base = base.merge(scores, on="ID", validate="one_to_one")
    cols = [f"score_emb128_seed{seed}" for seed in seeds]
    base["score_emb128_ens"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "score_emb128_ens"]]


def z_within_user(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("userID")[col]
    std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
    return (df[col] - grouped.transform("mean")) / std


def metric_float(summary: dict[str, object], key: str) -> float:
    value = summary[key]
    if not isinstance(value, Real):
        raise TypeError(f"metric {key} is not numeric: {value!r}")
    return float(value)


def build_item_neighbors(mat: sp.csr_matrix, topk: int) -> tuple[np.ndarray, np.ndarray]:
    item_degree = np.asarray(mat.sum(axis=0)).ravel().astype(np.float64)
    cooc = (mat.T @ mat).astype(np.float64).tocsr()
    n_items = mat.shape[1]
    neighbors = np.zeros((n_items, topk), dtype=np.int64)
    weights = np.zeros((n_items, topk), dtype=np.float32)
    for item in range(n_items):
        start, end = cooc.indptr[item], cooc.indptr[item + 1]
        idx = cooc.indices[start:end]
        val = cooc.data[start:end]
        mask = idx != item
        idx = idx[mask]
        val = val[mask]
        if idx.size == 0:
            neighbors[item, :] = item
            continue
        denom = np.sqrt(max(item_degree[item], 1.0) * np.maximum(item_degree[idx], 1.0))
        sim = val / denom
        order = np.argsort(sim, kind="mergesort")[::-1][:topk]
        chosen = idx[order]
        chosen_w = sim[order]
        fill = len(chosen)
        neighbors[item, :fill] = chosen
        weights[item, :fill] = chosen_w.astype(np.float32)
        if fill < topk:
            neighbors[item, fill:] = chosen[-1]
            weights[item, fill:] = float(chosen_w[-1])
    max_w = float(weights.max())
    if max_w > 0:
        weights = weights / max_w
    return neighbors, weights


class ConstraintMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)


def train_constraint_mf(
    mat: sp.csr_matrix,
    n_users: int,
    n_items: int,
    neighbor_idx: np.ndarray,
    neighbor_w: np.ndarray,
    *,
    emb_dim: int,
    lr: float,
    reg: float,
    epochs: int,
    batch_size: int,
    bpr_weight: float,
    pointwise_weight: float,
    item_constraint_weight: float,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = ConstraintMF(n_users, n_items, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    n_batches = max(1, mat.nnz // batch_size)
    item_neighbor_idx = torch.as_tensor(neighbor_idx, dtype=torch.long, device=device)
    item_neighbor_w = torch.as_tensor(neighbor_w, dtype=torch.float32, device=device)
    losses: list[float] = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_bpr_batch(mat, batch_size, n_items, rng)
            users_t = torch.as_tensor(users, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos_items, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg_items, dtype=torch.long, device=device)

            pos_score = model.score(users_t, pos_t)
            neg_score = model.score(users_t, neg_t)
            bpr = -F.logsigmoid(pos_score - neg_score).mean()
            pointwise = -0.5 * (F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())

            pick = torch.as_tensor(
                rng.integers(0, neighbor_idx.shape[1], size=len(pos_items)),
                dtype=torch.long,
                device=device,
            )
            neigh_t = item_neighbor_idx[pos_t, pick]
            neigh_weight = item_neighbor_w[pos_t, pick]
            item_score = (model.item_emb(pos_t) * model.item_emb(neigh_t)).sum(dim=1)
            item_constraint = (neigh_weight * -F.logsigmoid(item_score)).mean()

            l2 = reg * (
                model.user_emb(users_t).norm(2).pow(2)
                + model.item_emb(pos_t).norm(2).pow(2)
                + model.item_emb(neg_t).norm(2).pow(2)
                + model.item_emb(neigh_t).norm(2).pow(2)
            ) / batch_size
            loss = bpr_weight * bpr + pointwise_weight * pointwise + item_constraint_weight * item_constraint + l2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.item())
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [UltraGCN-smoke] epoch {epoch + 1}/{epochs} loss={avg:.6f} elapsed={time.time() - started:.1f}s", flush=True)
        if not np.isfinite(avg):
            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}: {avg}")
    model.eval()
    with torch.no_grad():
        user_np = model.user_emb.weight.detach().cpu().numpy()
        item_np = model.item_emb.weight.detach().cpu().numpy()
    meta = {
        "emb_dim": emb_dim,
        "lr": lr,
        "reg": reg,
        "epochs": epochs,
        "batch_size": batch_size,
        "bpr_weight": bpr_weight,
        "pointwise_weight": pointwise_weight,
        "item_constraint_weight": item_constraint_weight,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(mat.nnz),
        "n_batches": n_batches,
        "final_loss": float(losses[-1]),
        "train_seconds": round(time.time() - started, 1),
        "device": device,
        "seed": seed,
    }
    return user_np, item_np, meta


def classify(acc: float, eq_blend: float | None, corr_z: float | None) -> tuple[str, str]:
    if acc <= 0.76000:
        return "KILL_WEAK_SOLO", f"solo {acc:.5f} <= 0.76000 kill gate"
    if eq_blend is None or corr_z is None:
        return "INCOMPLETE_REF", "emb128 ensemble reference missing; cannot judge blend gate"
    delta_eq = eq_blend - EMB128_ENSEMBLE_REF
    if delta_eq > NOISE:
        return "ESCALATE_3SPLIT_PANEL", f"eq-blend {eq_blend:.5f} beats emb128 ensemble by {delta_eq:+.5f} > noise"
    if acc > EMB128_SINGLE_REF + NOISE:
        return "BACKBONE_SIGNAL_BUT_NO_BLEND", f"solo beats single-seed ref but eq-blend delta {delta_eq:+.5f} does not clear noise"
    return "KILL_NO_GAIN", f"solo {acc:.5f} and eq-blend delta {delta_eq:+.5f} do not justify panel"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--validation-root", default="artifacts/validation")
    ap.add_argument("--out-dir", default="artifacts/ultragcn_constraint_smoke")
    ap.add_argument("--report-json", default="reports/20260616T_ultragcn_constraint_smoke.json")
    ap.add_argument("--report-md", default="reports/20260616T_ultragcn_constraint_smoke.md")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--item-topk", type=int, default=10)
    ap.add_argument("--bpr-weight", type=float, default=1.0)
    ap.add_argument("--pointwise-weight", type=float, default=0.2)
    ap.add_argument("--item-constraint-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    split_dir = ROOT / args.validation_root / args.split
    train = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"[UltraGCN-smoke] {args.split}: {n_users} users, {n_items} items, {mat.nnz} interactions, device={device}", flush=True)

    neighbor_idx, neighbor_w = build_item_neighbors(mat, args.item_topk)
    user_emb, item_emb, train_meta = train_constraint_mf(
        mat,
        n_users,
        n_items,
        neighbor_idx,
        neighbor_w,
        emb_dim=args.emb_dim,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bpr_weight=args.bpr_weight,
        pointwise_weight=args.pointwise_weight,
        item_constraint_weight=args.item_constraint_weight,
        device=device,
        seed=args.seed,
    )

    scored = candidates.copy()
    scored["score_ultragcn"] = score_candidates(scored, user_emb, item_emb, user_to_idx, item_to_idx)
    summary, _ = evaluate_tophalf(scored, "score_ultragcn", label_col="Label", user_col="userID", id_col="ID")
    acc = round(metric_float(summary, "row_accuracy"), 5)

    corr_z = None
    eq_blend = None
    ref = load_emb128_ensemble(args.split)
    if ref is not None:
        merged = scored[["ID", "userID", "Label", "score_ultragcn"]].merge(ref[["ID", "score_emb128_ens"]], on="ID", validate="one_to_one")
        merged["z_ultra"] = z_within_user(merged, "score_ultragcn")
        merged["z_emb128"] = z_within_user(merged, "score_emb128_ens")
        corr_z = round(float(merged["z_ultra"].corr(merged["z_emb128"])), 4)
        merged["score_eq_blend"] = 0.5 * merged["z_ultra"] + 0.5 * merged["z_emb128"]
        blend_summary, _ = evaluate_tophalf(merged, "score_eq_blend", label_col="Label", user_col="userID", id_col="ID")
        eq_blend = round(metric_float(blend_summary, "row_accuracy"), 5)

    tier, tier_reason = classify(acc, eq_blend, corr_z)
    run_dir = ensure_dir(ROOT / args.out_dir / f"d{args.emb_dim}_seed{args.seed}" / args.split)
    scored[["ID", "userID", "gameID", "Label", "score_ultragcn"]].to_csv(run_dir / "ultragcn_validation_scores.csv", index=False)
    payload = {
        "note": "UltraGCN-style constraint-loss smoke. Validation-only; no Kaggle submission; no full-test candidate.",
        "safety": {
            "validation_only": True,
            "candidate_csv_written": False,
            "full_test_scored": False,
            "submission_csv_written": False,
            "kaggle_submit_executed": False,
            "hidden_labels_used": False,
            "external_metadata_used": False,
        },
        "split": args.split,
        "artifact_dir": str(run_dir.relative_to(ROOT)),
        "config": vars(args) | {"effective_device": device},
        "references": {
            "emb128_single_seed_uniform": EMB128_SINGLE_REF,
            "emb128_4seed_uniform": EMB128_ENSEMBLE_REF,
            "noise": NOISE,
        },
        "metrics": {
            "solo_acc": acc,
            "delta_vs_emb128_single": round(acc - EMB128_SINGLE_REF, 5),
            "delta_vs_emb128_4seed": round(acc - EMB128_ENSEMBLE_REF, 5),
            "corr_z_vs_emb128_4seed": corr_z,
            "eq_blend_acc": eq_blend,
            "eq_blend_delta_vs_emb128_4seed": (round(eq_blend - EMB128_ENSEMBLE_REF, 5) if eq_blend is not None else None),
            "solo_summary": summary,
        },
        "train_meta": train_meta,
        "tier": tier,
        "tier_reason": tier_reason,
    }
    write_json(ROOT / args.report_json, payload)

    lines = [
        "# UltraGCN-style constraint-loss smoke",
        "",
        "Safety: validation-only; no Kaggle submit; no `submissions/` write; no full-test candidate.",
        "",
        "## Result",
        "",
        f"- split: `{args.split}`",
        f"- artifact dir: `{run_dir.relative_to(ROOT)}`",
        f"- solo accuracy: **{acc:.5f}**",
        f"- delta vs emb128 single-seed 0.76205: **{acc - EMB128_SINGLE_REF:+.5f}**",
        f"- delta vs emb128 4-seed 0.76505: **{acc - EMB128_ENSEMBLE_REF:+.5f}**",
        f"- corr_z vs emb128 4-seed: `{corr_z}`",
        f"- 50/50 z-blend accuracy: `{eq_blend}`",
        f"- tier: **{tier}** — {tier_reason}",
        "",
        "## Config",
        "",
        f"- emb_dim={args.emb_dim}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, reg={args.reg}",
        f"- bpr_weight={args.bpr_weight}, pointwise_weight={args.pointwise_weight}, item_constraint_weight={args.item_constraint_weight}",
        f"- item_topk={args.item_topk}, seed={args.seed}, device={device}",
        "",
        "## Gate",
        "",
        "Escalate only if the 50/50 z-blend beats emb128 4-seed by more than +0.0007 on this smoke, then rerun as a 3-split panel. Otherwise close the axis.",
        "",
        "ULTRAGCN_CONSTRAINT_SMOKE_DONE",
    ]
    Path(ROOT / args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": args.report_json, "report_md": args.report_md, "tier": tier, "solo_acc": acc, "eq_blend": eq_blend}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
