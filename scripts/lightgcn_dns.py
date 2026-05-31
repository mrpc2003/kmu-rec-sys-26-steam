#!/usr/bin/env python3
"""DNS (Dynamic Negative Sampling) LightGCN — UNIFORM gate.

WHY this is structurally different from every prior failed bet (2026-05-31)
---------------------------------------------------------------------------
All prior new-direction attempts (SGL/SimGCL/XSimGCL/DirectAU/MiniLM/Turbo-CF) tried to ADD
a weak orthogonal axis to the strong emb128 backbone, and all failed because the orthogonal
signals are too weak (below the 0.684 popularity floor) on this small balanced-reranking set.

DNS attacks the problem from the opposite side: it does NOT add a side signal. It STRENGTHENS
the backbone itself by replacing uniform-random BPR negatives with HARD negatives — for each
(user, pos), sample a pool of M candidate negatives, score them with the current embeddings,
and train against the hardest (highest-scoring) one. This sharpens the decision boundary and
is documented (DNS, Zhang'13; MixGCF KDD'21; DINS'23) to push single-model ranking above
plain BPR. It keeps full BPR ranking strength (unlike contrastive uniformity), so it can
clear the floor.

Honest prior: MODERATE. Risk = hard negatives skew toward popular-item discrimination, and
this competition's public LB tracks the UNIFORM negative distribution, so a hard-trained model
could help on hard samplers but not on uniform. That's exactly why the gate is the uniform
public-surrogate split: only a uniform gain (vs single-seed backbone ~0.762 and 4-seed
ensemble 0.76505) counts. No validation-label learning -> not a stacker-trap.

Sweeps DNS pool M in {1(=plain control), 8, 16, 32} on emb128 L4 reg1e-3. Validation-only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (
    build_user_item_matrix, ensure_dir, evaluate_tophalf,
    load_pairs_csv, load_train_interactions, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, score_candidates

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_ENS_REF = 0.76505
SINGLE_SEED_REF = 0.762  # plain emb128 single-seed uniform (control should land near here)
POP_FLOOR = 0.684
NOISE = 0.0007


def sample_pos(mat, batch_size, n_items, rng):
    users = rng.integers(0, mat.shape[0], size=batch_size)
    pos = np.zeros(batch_size, dtype=np.int64)
    for i, u in enumerate(users):
        s, e = mat.indptr[u], mat.indptr[u + 1]
        pos[i] = rng.choice(mat.indices[s:e]) if e > s else rng.integers(0, n_items)
    return users, pos


def train_dns(mat, n_users, n_items, emb_dim, n_layers, lr, reg, dns_pool,
              epochs, batch_size, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    nb = max(1, mat.nnz // batch_size)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        eloss = 0.0
        for _ in range(nb):
            u, p = sample_pos(mat, batch_size, n_items, rng)
            ut = torch.LongTensor(u).to(device)
            pt = torch.LongTensor(p).to(device)
            # candidate negative pool: M uniform negatives per row
            negc = rng.integers(0, n_items, size=(batch_size, dns_pool))
            negc_t = torch.LongTensor(negc).to(device)

            ue, ie = model(adj)
            u_e = ue[ut]                                  # (B, d)
            if dns_pool > 1:
                with torch.no_grad():
                    cand_e = ie[negc_t]                   # (B, M, d)
                    cand_s = torch.einsum("bd,bmd->bm", u_e, cand_e)  # (B, M)
                    hard = cand_s.argmax(dim=1)           # (B,) hardest negative idx
                nt = negc_t[torch.arange(batch_size, device=device), hard]
            else:
                nt = negc_t[:, 0]                         # plain random (control)

            p_e = ie[pt]; n_e = ie[nt]
            bpr = -F.logsigmoid((u_e * p_e).sum(1) - (u_e * n_e).sum(1)).mean()
            reg_l = reg * (model.user_emb(ut).norm(2).pow(2)
                           + model.item_emb(pt).norm(2).pow(2)
                           + model.item_emb(nt).norm(2).pow(2)) / batch_size
            loss = bpr + reg_l
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += float(loss.item())
        if (ep + 1) % 40 == 0 or ep == 0:
            print(f"  ep{ep+1}/{epochs} loss={eloss/nb:.4f} t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    meta = {"emb_dim": emb_dim, "n_layers": n_layers, "lr": lr, "reg": reg,
            "dns_pool": dns_pool, "epochs": epochs, "batch_size": batch_size,
            "n_users": n_users, "n_items": n_items, "n_interactions": int(mat.nnz),
            "train_seconds": round(time.time() - t0, 1), "device": device, "seed": seed}
    return ue.cpu().numpy(), ie.cpu().numpy(), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--dns-pool", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"dns{args.dns_pool}_emb{args.emb_dim}_L{args.n_layers}_reg{args.reg:.0e}_seed{args.seed}"

    sp_dir = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] {args.split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
    ue, ie, meta = train_dns(mat, len(users), len(items), args.emb_dim, args.n_layers,
                             args.lr, args.reg, args.dns_pool, args.epochs,
                             args.batch_size, args.device, args.seed)
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label",
                               user_col="userID", id_col="ID")
    acc = round(float(summ["row_accuracy"]), 5)
    outd = ensure_dir(out / args.split)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(
        outd / "lightgcn_scores.csv", index=False)
    vs_ens = round(acc - EMB128_ENS_REF, 5)
    write_json(outd / "summary.json", {"tag": tag, "row_accuracy": acc,
               "emb128_ens_ref": EMB128_ENS_REF, "single_seed_ref": SINGLE_SEED_REF,
               "vs_ens_ref": vs_ens, "meta": meta})
    print(f"[{tag}] uniform row_acc={acc} (ens ref {EMB128_ENS_REF}, vs {vs_ens:+.5f}; "
          f"single-seed ref ~{SINGLE_SEED_REF}) ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
