"""Full-train LightGCN and SAVE raw test-pair scores (for stacker materialization).

The original full-train only saved decoded 0/1 labels. The stacker needs the
continuous score_lightgcn per test ID. This reruns the SAME config (default
emb64 L3 reg1e-4 seed42, exactly reproducing the 0.76245 candidate) and saves:
  - artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores.csv  (ID, score_lightgcn)
  - re-verifies the decoded candidate hashes the same as the submitted file.

No Kaggle submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    load_pairs_csv, ensure_dir, write_json, predict_tophalf,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

SUBMITTED_SHA = "a3dbe043f0f8b781d8c35aea88b7a1f561fa7b705b34edf6c7b7d0451eceb2a6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="emb64_L3_reg1e-04")
    args = ap.parse_args()

    train_df = load_train_json(DEFAULT_DATA_DIR / "train.json")
    pairs_df = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(train_df, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"[FULL-SAVE] {n_users}u {n_items}i {mat.nnz}nnz config={args.tag}", flush=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    adj = build_norm_adj(mat, n_users, n_items).to(args.device)
    model = LightGCN(n_users, n_items, args.emb_dim, args.n_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    n_batches = max(1, mat.nnz // args.batch_size)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        eloss = 0.0
        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(mat, args.batch_size, n_items, rng)
            u_t = torch.LongTensor(u).to(args.device)
            p_t = torch.LongTensor(p).to(args.device)
            n_t = torch.LongTensor(n).to(args.device)
            ue, ie = model(adj)
            pos = (ue[u_t] * ie[p_t]).sum(1)
            neg = (ue[u_t] * ie[n_t]).sum(1)
            bpr = -F.logsigmoid(pos - neg).mean()
            reg = args.reg * (model.user_emb(u_t).norm(2).pow(2)
                              + model.item_emb(p_t).norm(2).pow(2)
                              + model.item_emb(n_t).norm(2).pow(2)) / args.batch_size
            loss = bpr + reg
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  ep{epoch+1}/{args.epochs} loss={eloss/n_batches:.6f} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    user_np, item_np = ue.cpu().numpy(), ie.cpu().numpy()
    elapsed = round(time.time() - t0, 1)
    print(f"[FULL-SAVE] trained in {elapsed}s", flush=True)

    scores = score_candidates(pairs_df, user_np, item_np, u2i, i2i)
    pairs_df = pairs_df.copy()
    pairs_df["score_lightgcn"] = scores

    out_dir = ensure_dir(ROOT / "artifacts/lightgcn_20260530/test_full_train")
    raw_path = out_dir / f"lightgcn_test_raw_scores_{args.tag}.csv"
    pairs_df[["ID", "userID", "gameID", "score_lightgcn"]].to_csv(raw_path, index=False)
    print(f"[FULL-SAVE] raw scores → {raw_path}", flush=True)

    # Re-decode and verify hash matches the submitted candidate (only for baseline tag)
    pred = predict_tophalf(pairs_df, "score_lightgcn", label_col=None,
                           user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")
    verify_path = out_dir / f"candidate_verify_{args.tag}.csv"
    sub.to_csv(verify_path, index=False)
    sha = hashlib.sha256(verify_path.read_bytes()).hexdigest()
    matches = (args.tag == "emb64_L3_reg1e-04" and sha == SUBMITTED_SHA)
    meta = {
        "config": vars(args),
        "raw_scores_path": str(raw_path),
        "verify_candidate_path": str(verify_path),
        "verify_sha256": sha,
        "submitted_sha256": SUBMITTED_SHA,
        "reproduces_submitted": bool(matches),
        "train_seconds": elapsed,
    }
    write_json(out_dir / f"raw_save_meta_{args.tag}.json", meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if args.tag == "emb64_L3_reg1e-04":
        print(f"[FULL-SAVE] reproduces submitted 0.76245 candidate: {matches}")


if __name__ == "__main__":
    main()
