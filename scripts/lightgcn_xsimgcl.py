#!/usr/bin/env python3
"""XSimGCL: eXtremely Simple Graph Contrastive Learning (Yu et al., TKDE 2023).

WHY this is a distinct bet from the failed SGL (kmu-rec-sys-26-steam, 2026-05-31)
--------------------------------------------------------------------------------
SGL (edge-dropout contrastive) failed the uniform gate badly (0.515-0.641, below the
0.684 popularity floor), as did DirectAU (0.547-0.597). Both use a uniformity-promoting
contrastive/align objective. My SGL already used the CLEAN graph for BPR/scoring, so the
failure was NOT edge-dropout corrupting the scored embeddings — it was InfoNCE dragging the
shared embedding table toward uniformity, which destroys the fine per-user ranking on this
dataset's tiny candidate sets (median 4 candidates/user).

XSimGCL isolates the OTHER variable: it DISCARDS edge dropout entirely and instead injects
small sign-preserving uniform noise at each propagation layer in a SINGLE forward pass, then
contrasts the final readout against one intermediate layer (l*). The TKDE paper shows this
beats edge-dropout CL "by a large margin" on large sparse graphs.

Honest expectation: XSimGCL shares the InfoNCE-uniformity mechanism that killed SGL/DirectAU,
so the prior is LOW. But running it cleanly CLOSES the contrastive-CF axis with evidence
rather than speculation, and it is cheap (single pass). Gate = beat emb128 4-seed 0.76505 on
the uniform public-surrogate split by > noise 0.0007 (parameter-free bar). Validation-only.

Reuses LightGCN graph + utils. No Kaggle submission.
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
    DEFAULT_DATA_DIR, build_user_item_matrix, ensure_dir, evaluate_tophalf,
    load_pairs_csv, load_train_interactions, load_train_json, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, sample_bpr_batch, score_candidates

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_ENS_REF = 0.76505
POP_FLOOR = 0.684
NOISE = 0.0007


def xsimgcl_forward(model: LightGCN, adj: torch.Tensor, eps: float, layer_cl: int,
                    perturbed: bool):
    """Single forward pass. Returns (user_final, item_final, emb_cl_full).

    XSimGCL: at each layer add Δ = sign(e) ⊙ normalize(U(0,1)) * eps (sign-preserving noise
    with controlled L2 magnitude). Readout = mean over [ego, L1..Ln] (matches the strong
    LightGCN baseline). emb_cl_full = the perturbed embedding at layer l* (full node table),
    used as one contrastive view; the final readout is the other view.
    """
    ego = model.get_ego_embeddings()
    all_embs = [ego]
    emb_cl = None
    cur = ego
    for layer in range(model.n_layers):
        cur = torch.sparse.mm(adj, cur)
        if perturbed:
            noise = F.normalize(torch.rand_like(cur), dim=1) * torch.sign(cur) * eps
            cur = cur + noise
        all_embs.append(cur)
        if (layer + 1) == layer_cl:
            emb_cl = cur
    final = torch.stack(all_embs, dim=1).mean(dim=1)
    if emb_cl is None:
        emb_cl = all_embs[-1]
    user_final, item_final = final[: model.n_users], final[model.n_users:]
    return user_final, item_final, emb_cl


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    pos = (z1 * z2).sum(dim=1) / tau
    logits = (z1 @ z2.t()) / tau
    return (-pos + torch.logsumexp(logits, dim=1)).mean()


def train_xsimgcl(mat, n_users, n_items, emb_dim, n_layers, lr, reg,
                  lam_ssl, tau, eps, layer_cl, epochs, batch_size, device, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    nb = max(1, mat.nnz // batch_size)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        el = eb = es = 0.0
        for _ in range(nb):
            u, p, ng = sample_bpr_batch(mat, batch_size, n_items, rng)
            ut = torch.LongTensor(u).to(device)
            pt = torch.LongTensor(p).to(device)
            nt = torch.LongTensor(ng).to(device)

            # perturbed forward used for BOTH BPR and CL (XSimGCL design)
            ue, ie, emb_cl = xsimgcl_forward(model, adj, eps, layer_cl, perturbed=True)
            bpr = -F.logsigmoid((ue[ut] * ie[pt]).sum(1) - (ue[ut] * ie[nt]).sum(1)).mean()
            reg_l = reg * (model.user_emb(ut).norm(2).pow(2)
                           + model.item_emb(pt).norm(2).pow(2)
                           + model.item_emb(nt).norm(2).pow(2)) / batch_size

            # CL views: final readout (perturbed) vs layer-l* (perturbed), on batch nodes
            final_full = torch.cat([ue, ie], dim=0)
            uu = torch.unique(ut)
            ii = torch.unique(pt) + n_users
            ssl = (info_nce(final_full[uu], emb_cl[uu], tau)
                   + info_nce(final_full[ii], emb_cl[ii], tau))

            loss = bpr + lam_ssl * ssl + reg_l
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += float(loss.item()); eb += float(bpr.item()); es += float(ssl.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep{ep+1}/{epochs} loss={el/nb:.4f} bpr={eb/nb:.4f} ssl={es/nb:.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    # eval with CLEAN (unperturbed) forward for deterministic scoring
    model.eval()
    with torch.no_grad():
        ue, ie, _ = xsimgcl_forward(model, adj, eps, layer_cl, perturbed=False)
    meta = {"emb_dim": emb_dim, "n_layers": n_layers, "lr": lr, "reg": reg,
            "lambda_ssl": lam_ssl, "tau": tau, "eps": eps, "layer_cl": layer_cl,
            "epochs": epochs, "batch_size": batch_size, "n_users": n_users,
            "n_items": n_items, "n_interactions": int(mat.nnz),
            "train_seconds": round(time.time() - t0, 1), "device": device, "seed": seed}
    return ue.cpu().numpy(), ie.cpu().numpy(), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--lambda-ssl", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--layer-cl", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = (f"xsimgcl_emb{args.emb_dim}_L{args.n_layers}_lam{args.lambda_ssl:g}"
           f"_eps{args.eps:g}_lcl{args.layer_cl}_t{args.tau:g}_seed{args.seed}")

    sp_dir = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] {args.split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)
    ue, ie, meta = train_xsimgcl(mat, len(users), len(items), args.emb_dim, args.n_layers,
                                 args.lr, args.reg, args.lambda_ssl, args.tau, args.eps,
                                 args.layer_cl, args.epochs, args.batch_size,
                                 args.device, args.seed)
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label",
                               user_col="userID", id_col="ID")
    acc = round(float(summ["row_accuracy"]), 5)
    outd = ensure_dir(out / args.split)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(
        outd / "lightgcn_scores.csv", index=False)
    verdict = ("ABOVE_FLOOR_CHECK_BLEND" if acc >= POP_FLOOR else "BELOW_POP_FLOOR_REJECT")
    write_json(outd / "summary.json", {"tag": tag, "row_accuracy": acc,
               "emb128_ref": EMB128_ENS_REF, "pop_floor": POP_FLOOR,
               "verdict": verdict, "meta": meta})
    print(f"[{tag}] uniform row_acc={acc} (ref {EMB128_ENS_REF}, floor {POP_FLOOR}) "
          f"-> {verdict} ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
