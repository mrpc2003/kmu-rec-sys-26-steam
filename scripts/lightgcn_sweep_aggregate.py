"""Aggregate LightGCN sweep results from 4 GPU workers and generate best-config full-train candidate.

Steps:
  1. Load gpu{0..3}_results.json from artifacts/lightgcn_hparam_sweep/
  2. Merge with baseline (emb64 L3 reg1e-4 = 0.6748/0.6396/0.6020)
  3. Print ranked table, save reports/20260530_lightgcn_hparam_sweep_final.{json,md}
  4. If --full-train: train best config on full train.json and save submission candidate
"""
from __future__ import annotations

import argparse
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
BASELINE_MEAN = sum(BASELINE.values()) / len(BASELINE)

EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
SEED = 42


def load_all_results() -> list[dict]:
    sweep_dir = ROOT / "artifacts/lightgcn_hparam_sweep"
    all_rows = []

    # Baseline
    all_rows.append({
        "tag": "emb64_L3_reg1e-04",
        "emb_dim": 64, "n_layers": 3, "reg": 1e-4,
        "val_random_sqrtpop_seed42": 0.6748,
        "val_recent_sqrtpop_seed42": 0.6396,
        "val_random_popbin_seed42":  0.6020,
        "mean_val": round(BASELINE_MEAN, 5),
        "mean_delta": 0.0,
        "note": "baseline (already submitted, public=0.76245)",
    })

    for gpu_id in range(4):
        p = sweep_dir / f"gpu{gpu_id}_results.json"
        if not p.exists():
            print(f"  [WARN] {p} not found — GPU{gpu_id} may still be running")
            continue
        rows = json.loads(p.read_text())
        for r in rows:
            if "mean_delta" not in r:
                scores = [r.get(s, 0) for s in SPLITS]
                r["mean_val"] = round(sum(scores) / len(scores), 5)
                r["mean_delta"] = round(r["mean_val"] - BASELINE_MEAN, 5)
            all_rows.append(r)

    return all_rows


def print_table(rows: list[dict]) -> None:
    sorted_rows = sorted(rows, key=lambda x: -x.get("mean_val", 0))
    header = f"{'tag':<30} {'emb':>5} {'L':>3} {'reg':>7} {'sqrtpop':>9} {'recent':>9} {'popbin':>9} {'mean':>9} {'Δmean':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted_rows:
        note = " ← BEST" if r == sorted_rows[0] else ""
        print(
            f"{r['tag']:<30} {r['emb_dim']:>5} {r['n_layers']:>3} {r['reg']:>7.0e} "
            f"{r.get('val_random_sqrtpop_seed42', 0):>9.5f} "
            f"{r.get('val_recent_sqrtpop_seed42', 0):>9.5f} "
            f"{r.get('val_random_popbin_seed42', 0):>9.5f} "
            f"{r.get('mean_val', 0):>9.5f} "
            f"{r.get('mean_delta', 0):>+8.5f}{note}"
        )


def train_full(emb_dim: int, n_layers: int, reg: float, device: str):
    print(f"\n[FULL-TRAIN] emb_dim={emb_dim} n_layers={n_layers} reg={reg:.0e} device={device}", flush=True)
    train_df = load_train_json(DEFAULT_DATA_DIR / "train.json")
    pairs_df = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(train_df, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"  {n_users} users, {n_items} items, {mat.nnz} interactions", flush=True)

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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  ep{epoch+1}/{EPOCHS} loss={epoch_loss/n_batches:.6f} t={time.time()-t0:.1f}s", flush=True)

    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    user_np = ue.cpu().numpy()
    item_np = ie.cpu().numpy()
    elapsed = round(time.time() - t0, 1)
    print(f"  training done in {elapsed}s", flush=True)

    # Score test pairs
    scores = score_candidates(pairs_df, user_np, item_np, u2i, i2i)
    pairs_df = pairs_df.copy()
    pairs_df["score_lightgcn"] = scores
    pred_df = predict_tophalf(pairs_df, "score_lightgcn", label_col=None,
                               user_col="userID", id_col="ID")
    submission = pred_df[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    tag = f"emb{emb_dim}_L{n_layers}_reg{reg:.0e}"
    out_dir = ensure_dir(ROOT / f"artifacts/lightgcn_best_config/{tag}")
    csv_path = out_dir / f"candidate_lightgcn_{tag}_full_train.csv"
    submission.to_csv(csv_path, index=False)

    import hashlib
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = {
        "file": str(csv_path),
        "sha256": sha,
        "rows": int(len(submission)),
        "label_1_count": int(submission["Label"].sum()),
        "label_0_count": int((1 - submission["Label"]).sum()),
        "train_seconds": elapsed,
        "config": {"emb_dim": emb_dim, "n_layers": n_layers, "lr": LR,
                   "reg": reg, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "seed": SEED, "device": device},
    }
    write_json(out_dir / "meta.json", meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return csv_path, sha, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-train", action="store_true",
                        help="Train best config on full data and generate submission candidate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-config", default=None,
                        help="Override best config as 'emb,layers,reg' e.g. '128,3,1e-4'")
    args = parser.parse_args()

    rows = load_all_results()
    if not rows:
        print("No results found yet.")
        return

    print(f"\n== LightGCN Sweep Results ({len(rows)} configs) ==\n")
    print_table(rows)

    sorted_rows = sorted(rows, key=lambda x: -x.get("mean_val", 0))
    best = sorted_rows[0]
    print(f"\nBest: {best['tag']} mean_val={best['mean_val']:.5f} Δ={best.get('mean_delta', 0):+.5f}")

    # Save final report
    out_json = ROOT / "reports/20260530_lightgcn_hparam_sweep_final.json"
    out_md = ROOT / "reports/20260530_lightgcn_hparam_sweep_final.md"
    write_json(out_json, sorted_rows)

    md = ["# LightGCN Hyperparameter Sweep — Final Results\n"]
    md.append("| tag | emb | L | reg | sqrtpop | recent | popbin | mean | Δmean |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted_rows:
        note = " **BEST**" if r == sorted_rows[0] else ""
        md.append(
            f"| {r['tag']}{note} | {r['emb_dim']} | {r['n_layers']} | {r['reg']:.0e} "
            f"| {r.get('val_random_sqrtpop_seed42', 0):.5f} "
            f"| {r.get('val_recent_sqrtpop_seed42', 0):.5f} "
            f"| {r.get('val_random_popbin_seed42', 0):.5f} "
            f"| {r.get('mean_val', 0):.5f} "
            f"| {r.get('mean_delta', 0):+.5f} |"
        )
    out_md.write_text("\n".join(md))
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")

    if args.full_train:
        if args.force_config:
            e, l, r = args.force_config.split(",")
            emb_dim, n_layers, reg = int(e), int(l), float(r)
        else:
            emb_dim = best["emb_dim"]
            n_layers = best["n_layers"]
            reg = best["reg"]
        csv_path, sha, meta = train_full(emb_dim, n_layers, reg, args.device)
        print(f"\n== Full-train candidate ready ==")
        print(f"  file: {csv_path}")
        print(f"  sha256: {sha}")
        print(f"  rows: {meta['rows']} label1={meta['label_1_count']} label0={meta['label_0_count']}")
        print(f"\n[STOP] Submission requires explicit user approval.")


if __name__ == "__main__":
    main()
