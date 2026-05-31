"""emb128_L4_reg1e-3 ensemble EXPANSION worker — one new seed, uniform-eval + full-test.

Context
-------
SGL and DirectAU both FAILED the strong+orthogonal bet (uniform 0.51-0.64, below the
0.684 popularity baseline). The honest remaining move from the synthesis report is
ensemble expansion (4 -> 8 seeds), pure variance reduction on the already-validated
emb128_L4_reg1e-3 backbone. No new validation-label learning, so no stacker-trap risk.

For the assigned --seed, this trains the CANONICAL emb128_L4_reg1e-3 LightGCN twice:
  1. uniform validation split -> score candidates -> uniform row_acc gate input
     out: artifacts/lightgcn_emb128L4r3_ens/seed{S}/val_random_uniform_seed42/lightgcn_scores.csv
  2. full train.json -> score test pairs -> 8-seed candidate material
     out: artifacts/lightgcn_emb128L4r3_fulltest/seed{S}/test.csv

Same code path (lightgcn_train.train_lightgcn) as the existing 4 seeds, so the new seeds
are drop-in for both the uniform gate and the full-test aggregator. No Kaggle submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    DEFAULT_DATA_DIR, build_user_item_matrix, load_train_json,
    load_pairs_csv, load_train_interactions, ensure_dir, write_json, evaluate_tophalf,
)
from lightgcn_train import train_lightgcn, score_candidates  # noqa: E402

EMB_DIM, N_LAYERS, LR, REG, EPOCHS, BATCH = 128, 4, 1e-3, 1e-3, 200, 4096
SPLIT = "val_random_uniform_seed42"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    tag = f"emb128_L4_reg1e-03_seed{args.seed}"

    # ---- 1. uniform validation split ----
    ens_out = ensure_dir(ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{args.seed}" / SPLIT)
    sp = ROOT / "artifacts/validation" / SPLIT
    tr = load_train_interactions(sp / "train_interactions.csv")
    cand = load_pairs_csv(sp / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] UNIFORM-VAL: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
    ue, ie, meta = train_lightgcn(
        mat, len(users), len(items), emb_dim=EMB_DIM, n_layers=N_LAYERS, lr=LR, reg=REG,
        epochs=EPOCHS, batch_size=BATCH, device=args.device, seed=args.seed,
    )
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(
        ens_out / "lightgcn_scores.csv", index=False)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label",
                               user_col="userID", id_col="ID")
    uni_acc = round(float(summ["row_accuracy"]), 5)
    write_json(ens_out / "summary.json", {"split": SPLIT, "summary": summ, "train_meta": meta})
    print(f"[{tag}] uniform row_acc={uni_acc}", flush=True)

    # ---- 2. full train -> test ----
    full_out = ensure_dir(ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{args.seed}")
    tr_full = load_train_json(DEFAULT_DATA_DIR / "train.json")
    pairs = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
    matf, u2if, i2if, usersf, itemsf = build_user_item_matrix(tr_full, binary=True)
    print(f"[{tag}] FULL: {len(usersf)}u {len(itemsf)}i {matf.nnz}nnz", flush=True)
    uef, ief, metaf = train_lightgcn(
        matf, len(usersf), len(itemsf), emb_dim=EMB_DIM, n_layers=N_LAYERS, lr=LR, reg=REG,
        epochs=EPOCHS, batch_size=BATCH, device=args.device, seed=args.seed,
    )
    pairs = pairs.copy()
    pairs["score_lightgcn"] = score_candidates(pairs, uef, ief, u2if, i2if)
    pairs[["ID", "userID", "gameID", "score_lightgcn"]].to_csv(full_out / "test.csv", index=False)
    write_json(full_out / "meta.json", {"tag": tag, "uniform_acc": uni_acc, "train_meta": metaf})
    print(f"[{tag}] FULL done -> test.csv | uniform={uni_acc} | ALL DONE", flush=True)


if __name__ == "__main__":
    main()
