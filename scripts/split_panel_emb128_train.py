#!/usr/bin/env python3
"""Train emb128 LightGCN (single model-seed) on a PARAMETERIZED validation split.

Purpose: empirically validate the gate-floor BLUNT verdict with REAL independent splits.
The bootstrap in gate_floor_bootstrap_analysis.py resamples ONE split (within-split variance).
The decision-relevant quantity is BETWEEN-split variance: train the same emb128 4-seed ensemble
on 3 independent uniform splits (seeds 42/7/123) and measure the real accuracy spread.
  spread ~0.007 -> BLUNT verdict empirically confirmed
  spread ~0.002 -> verdict overturned (bootstrap overestimated; gate actually sharp)

This trainer does ONE (split, model-seed) cell. The aggregator combines 4 model-seeds per split
into the canonical raw-mean ensemble and compares across splits.

Canonical train_lightgcn code path (same as every gated LightGCN result). Validation-only. No submission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix, load_pairs_csv, load_train_interactions,
    evaluate_tophalf, ensure_dir, write_json,
)
from lightgcn_train import train_lightgcn, score_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, help="e.g. val_random_uniform_seed7")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, required=True, help="model seed (42/123/2024/7)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="artifacts/split_panel_emb128")
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir) / args.split / f"seed{args.seed}")
    sp_dir = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[panel {args.split} seed{args.seed}] {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    ue, ie, meta = train_lightgcn(
        mat, len(users), len(items), emb_dim=args.emb_dim, n_layers=args.n_layers,
        lr=args.lr, reg=args.reg, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    acc = round(float(summ["row_accuracy"]), 5)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out / "lightgcn_scores.csv", index=False)
    write_json(out / "summary.json", {"split": args.split, "seed": args.seed, "emb_dim": args.emb_dim,
               "row_accuracy": acc, "meta": meta})
    print(f"[panel {args.split} seed{args.seed}] uniform row_acc={acc} ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
