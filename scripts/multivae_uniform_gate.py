#!/usr/bin/env python3
"""MultiVAE (Liang et al., WWW'18) — latent VAE reconstruction CF — UNIFORM gate.

WHY this is the genuinely-last untried MAJOR paradigm (2026-05-31)
------------------------------------------------------------------
Every prior bet fell into three already-closed families:
  - BPR-LightGCN graph family (SGL/SimGCL/XSimGCL/DirectAU/DNS) -- graph propagation + pairwise
  - item-item linear (EASE/ItemKNN/Turbo-CF)                    -- closed-form similarity scorer
  - text/LM semantic (TF-IDF/MiniLM/AlphaRec)                   -- review-text axis
MultiVAE is the 4th paradigm: a DENOISING VARIATIONAL AUTOENCODER that reconstructs the user's
full interaction vector under a MULTINOMIAL likelihood. Completely different inductive bias --
no graph, no pairwise ranking, no item-item kernel. It dominates MovieLens20M/Netflix and is
the backbone the GeoCF paper (arXiv 2410.03064) builds on. GeoCF's novelty is item-metadata
geometry, which is N/A here (anonymous gameID, no metadata), so we test the metadata-free
MultiVAE backbone itself.

Honest prior: MODERATE-LOW. EASE is the LINEAR autoencoder cousin and was REDUNDANT (corr_z
0.79 vs emb128), so MultiVAE may share signal. BUT its stochastic denoising + nonlinearity
prevents the identity-overfit that limits linear AEs, giving it a real chance to be either
(a) a stronger solo model or (b) genuinely orthogonal. The decisive measurement is corr_z vs
emb128 plus solo vs floor 0.684 and a parameter-free 50/50 z-blend vs ensemble ref 0.76505.

Scoring for this competition: train MultiVAE on the fold-train interaction matrix; for each
candidate (u,i), score = decoder logit for item i given user u's (denoised) input vector.
Per-user top-half decode as usual. Validation-only. No Kaggle submission.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (
    build_user_item_matrix, ensure_dir, evaluate_tophalf,
    load_pairs_csv, load_train_interactions, write_json,
)

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_SEED42_UNI = ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv"
EMB128_ENS_REF = 0.76505
POP_FLOOR = 0.684
NOISE = 0.0007


class MultiVAE(nn.Module):
    def __init__(self, n_items: int, hidden: int = 600, latent: int = 200, dropout: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, latent * 2)  # mu | logvar
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        for m in [self.enc1, self.enc2, self.dec1, self.dec2]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.latent = latent

    def encode(self, x):
        h = F.normalize(x, dim=1)
        h = self.drop(h)
        h = torch.tanh(self.enc1(h))
        h = self.enc2(h)
        return h[:, :self.latent], h[:, self.latent:]

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        h = torch.tanh(self.dec1(z))
        logits = self.dec2(h)
        return logits, mu, logvar


def multivae_loss(logits, x, mu, logvar, beta):
    log_softmax = F.log_softmax(logits, dim=1)
    nll = -(log_softmax * x).sum(dim=1).mean()
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return nll + beta * kld


def train_multivae(X, n_items, hidden, latent, dropout, lr, epochs, batch, device,
                   beta_cap, anneal_steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultiVAE(n_items, hidden, latent, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    n_users = X.shape[0]
    Xd = torch.FloatTensor(X.toarray()).to(device)
    step = 0
    beta = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_users)
        eloss = 0.0; nb = 0
        for s in range(0, n_users, batch):
            idx = perm[s:s + batch]
            xb = Xd[idx]
            beta = min(beta_cap, beta_cap * step / max(1, anneal_steps))
            logits, mu, logvar = model(xb)
            loss = multivae_loss(logits, xb, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += float(loss.item()); nb += 1; step += 1
        if (ep + 1) % 40 == 0 or ep == 0:
            print(f"  ep{ep+1}/{epochs} loss={eloss/nb:.4f} beta={beta:.3f} t={time.time()-t0:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(Xd)  # (n_users, n_items) logits
    return recon.cpu().numpy(), round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--hidden", type=int, default=600)
    ap.add_argument("--latent", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--beta-cap", type=float, default=0.2)
    ap.add_argument("--anneal-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"multivae_h{args.hidden}_z{args.latent}_d{args.dropout:g}_beta{args.beta_cap:g}_seed{args.seed}"

    sp_dir = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] {args.split}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    recon, secs = train_multivae(mat.tocsr(), len(items), args.hidden, args.latent,
                                 args.dropout, args.lr, args.epochs, args.batch,
                                 args.device, args.beta_cap, args.anneal_steps, args.seed)

    # score candidates: decoder logit for (user-row, item-col); mask is implicit (we read raw logit)
    scores = np.full(len(cand), np.nan, np.float32)
    for n, (uid, gid) in enumerate(cand[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ui = u2i.get(uid); ii = i2i.get(gid)
        if ui is not None and ii is not None:
            scores[n] = recon[ui, ii]
    cand = cand.copy(); cand["score_vae"] = scores
    cand["score_vae"] = cand["score_vae"].fillna(np.nanmin(scores[~np.isnan(scores)]) - 1.0)

    summ, _ = evaluate_tophalf(cand, "score_vae", label_col="Label", user_col="userID", id_col="ID")
    solo = round(float(summ["row_accuracy"]), 5)

    # corr + parameter-free blend vs emb128
    import pandas as pd
    e128 = pd.read_csv(EMB128_SEED42_UNI)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "score_cf"})
    m = cand.merge(e128, on="ID", how="inner")
    def zw(df, col):
        g = df.groupby("userID")[col]
        return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)).to_numpy()
    m["zt"] = zw(m, "score_vae"); m["zc"] = zw(m, "score_cf")
    corr_z = float(np.corrcoef(m["zt"], m["zc"])[0, 1])
    m["zb"] = 0.5 * m["zt"] + 0.5 * m["zc"]
    b, _ = evaluate_tophalf(m, "zb", label_col="Label", user_col="userID", id_col="ID")
    blend = round(float(b["row_accuracy"]), 5)
    blend_vs = round(blend - EMB128_ENS_REF, 5)

    if solo > EMB128_ENS_REF + NOISE:
        tier = "SOLO_UPGRADE"
        verdict = f"MultiVAE solo {solo} beats ensemble ref {EMB128_ENS_REF} by {solo-EMB128_ENS_REF:+.5f} -> escalate (seed ensemble + candidate)."
    elif blend_vs > NOISE and corr_z < 0.6:
        tier = "ORTHOGONAL_ADOPT"
        verdict = f"50/50 z-blend {blend} beats ref by {blend_vs:+.5f} with corr_z {corr_z:.3f} -> orthogonal gain, escalate."
    elif solo < POP_FLOOR:
        tier = "REJECT_WEAK"
        verdict = f"solo {solo} < floor {POP_FLOOR}; corr_z {corr_z:.3f}; blend {blend} ({blend_vs:+.5f}) -> too weak, VAE paradigm closed."
    else:
        tier = "REDUNDANT"
        verdict = f"solo {solo} (>=floor), corr_z {corr_z:.3f}, blend {blend} vs ref {blend_vs:+.5f} within/under noise -> redundant with backbone (like EASE), no new axis."

    summary = {"note": "MultiVAE latent-VAE reconstruction CF uniform gate. No submission.",
               "split": args.split, "rows": int(len(m)), "tag": tag, "train_seconds": secs,
               "solo_uniform": solo, "corr_withinuser_z": round(corr_z, 4),
               "blend50_uniform": blend, "emb128_ens_ref": EMB128_ENS_REF,
               "blend50_vs_ref": blend_vs, "pop_floor": POP_FLOOR, "noise": NOISE,
               "tier": tier, "verdict": verdict}
    outd = ensure_dir(out / args.split)
    cand[["ID", "userID", "gameID", "Label", "score_vae"]].to_csv(outd / "vae_scores.csv", index=False)
    write_json(outd / "summary.json", summary)
    print(f"\n[MULTIVAE GATE] solo={solo} corr_z={corr_z:.3f} blend50={blend} "
          f"vs_ref={blend_vs:+.5f} tier={tier} ({secs}s)", flush=True)
    print(f"verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
