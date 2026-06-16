#!/usr/bin/env python3
"""Validation-only LightGCN checkpoint prediction averaging probe.

This tests the second remaining internal improvement axis after pseudo-label top-1 did
not pass the multi-split gate: late-epoch prediction averaging / SWA-like stabilization.

For one validation split and one model seed, train the canonical emb128 L4 reg1e-3
LightGCN and save candidate scores at selected late epochs. The script evaluates
single-checkpoint scores and deterministic averages of late checkpoints.

Safety contract:
- validation_only: true
- no Kaggle submission
- no submissions/*.csv writes
- no full-test candidate writes
- no external Steam metadata or scraping
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.io.formats.csvs  # noqa: F401  # eager import avoids parallel to_csv lazy-import race
import torch
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))

from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates  # noqa: E402
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    write_json,
)

BASELINE_4SEED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}


def parse_checkpoints(text: str) -> list[int]:
    checkpoints = sorted({int(x) for x in text.split(",") if x.strip()})
    if not checkpoints:
        raise ValueError("at least one checkpoint epoch is required")
    return checkpoints


def avg_columns(score_df: pd.DataFrame, epochs: list[int]) -> dict[str, list[str]]:
    cols = {e: f"score_ep{e}" for e in epochs}
    variants: dict[str, list[str]] = {f"score_ep{e}": [cols[e]] for e in epochs}
    if len(epochs) >= 2:
        variants[f"score_avg_last2_{epochs[-2]}_{epochs[-1]}"] = [cols[e] for e in epochs[-2:]]
    if len(epochs) >= 3:
        variants[f"score_avg_last3_{epochs[-3]}_{epochs[-1]}"] = [cols[e] for e in epochs[-3:]]
    if len(epochs) >= 5:
        variants[f"score_avg_all_{epochs[0]}_{epochs[-1]}"] = [cols[e] for e in epochs]
    return variants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(BASELINE_4SEED))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--checkpoint-epochs", default="120,140,160,180,200")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-root", default="artifacts/lightgcn_checkpoint_avg_20260613T0106KST")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoint_epochs)
    if checkpoints[-1] > args.epochs:
        raise ValueError(f"checkpoint {checkpoints[-1]} > epochs {args.epochs}")

    split_dir = ROOT / "artifacts/validation" / args.split
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(train_df, binary=True)
    tag = f"split={args.split} seed={args.seed} checkpoints={checkpoints}"
    print(f"[ckptavg] {tag}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    adj = build_norm_adj(mat, len(users), len(items)).to(args.device)
    model = LightGCN(len(users), len(items), args.emb_dim, args.n_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    n_batches = max(1, mat.nnz // args.batch_size)
    started = time.time()
    scored = candidates[["ID", "userID", "gameID", "Label"]].copy()
    per_epoch_summary: dict[str, object] = {}

    checkpoint_set = set(checkpoints)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(mat, args.batch_size, len(items), rng)
            ut = torch.LongTensor(u).to(args.device)
            pt = torch.LongTensor(p).to(args.device)
            nt = torch.LongTensor(n).to(args.device)
            ue, ie = model(adj)
            pos = (ue[ut] * ie[pt]).sum(dim=1)
            neg = (ue[ut] * ie[nt]).sum(dim=1)
            loss = -F.logsigmoid(pos - neg).mean()
            loss = loss + args.reg * (
                model.user_emb(ut).norm(2).pow(2)
                + model.item_emb(pt).norm(2).pow(2)
                + model.item_emb(nt).norm(2).pow(2)
            ) / args.batch_size
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        if epoch == 1 or epoch % 20 == 0 or epoch in checkpoint_set:
            print(f"  [ckptavg] epoch {epoch}/{args.epochs} loss={epoch_loss/n_batches:.6f} elapsed={time.time()-started:.1f}s", flush=True)
        if epoch in checkpoint_set:
            model.eval()
            with torch.no_grad():
                ue, ie = model(adj)
            col = f"score_ep{epoch}"
            scored[col] = score_candidates(candidates, ue.cpu().numpy(), ie.cpu().numpy(), u2i, i2i)
            summ, _ = evaluate_tophalf(scored, col, label_col="Label", user_col="userID", id_col="ID")
            per_epoch_summary[col] = summ
            print(f"  [ckptavg] {col} acc={summ['row_accuracy']:.6f}", flush=True)

    variants = avg_columns(scored, checkpoints)
    variant_summary: dict[str, object] = {}
    for name, cols in variants.items():
        if len(cols) == 1 and name in per_epoch_summary:
            variant_summary[name] = per_epoch_summary[name]
        else:
            scored[name] = scored[cols].mean(axis=1)
            summ, _ = evaluate_tophalf(scored, name, label_col="Label", user_col="userID", id_col="ID")
            variant_summary[name] = summ

    out = ensure_dir(Path(args.out_root) / args.split / f"seed{args.seed}")
    score_cols = [c for c in scored.columns if c.startswith("score_")]
    scored[["ID", "userID", "gameID", "Label", *score_cols]].to_csv(out / "checkpoint_scores.csv", index=False)
    summary = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_scored": False,
        "external_metadata_used": False,
        "split": args.split,
        "seed": args.seed,
        "config": vars(args) | {"checkpoints": checkpoints},
        "baseline_4seed_acc_recorded": BASELINE_4SEED[args.split],
        "variant_summary": variant_summary,
        "train_seconds": round(time.time() - started, 1),
        "out_dir": str(out),
    }
    write_json(out / "summary.json", summary)
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
