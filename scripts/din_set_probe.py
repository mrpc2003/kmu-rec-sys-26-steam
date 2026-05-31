#!/usr/bin/env python3
"""DIN-style target-conditioned set encoder probe (GPT-5.5 research direction #1).

NEW PARADIGM — the one structural blind spot of LightGCN
--------------------------------------------------------
LightGCN produces ONE static user embedding (neighbourhood mean). This model instead
computes a CANDIDATE-SPECIFIC user representation via target attention (DIN, Zhou et al.
KDD 2018): the candidate game is the query, the user's played-game SET is keys/values, so
the user vector changes per candidate. This can capture higher-order CONDITIONAL set
interactions LightGCN structurally cannot:
    "user played A and B together -> C likely, but A alone or B alone insufficient."

Set, NOT sequence (the SASRec lesson)
-------------------------------------
No positional embeddings. The user's history is an unordered SET. SASRec failed because
this is a set-membership task (held-out positive is randomly masked, not chronological
"next"); target attention over an unordered set respects that.

Anti-collapse (GPT's warning)
-----------------------------
With median 21 interactions and only 2437 games, plain set pooling would relearn
co-occurrence and correlate ~0.97 with LightGCN. To stay genuinely target-conditioned the
score MLP consumes [q, attended, q*attended, mean_pool, max_pool] — the q*attended cross
term is what LightGCN's static dot-product cannot represent.

Leakage / trap guards
---------------------
- History sets built ONLY from the split's train_interactions (held-out positives already
  excluded by the split builder). Validation candidates are held-out positives + sampled
  negatives, none of which are in train history -> no leakage.
- Training mimics validation by LEAVE-ONE-OUT: mask a random train positive as the target,
  use the rest as history, draw uniform-unseen negatives (public-LB surrogate).
- Does NOT consume the LightGCN score as a feature -> cannot trivially reconstruct it.

Gate (identical to hyperbolic / SASRec): solo vs floor 0.684, corr_z vs emb128 4-seed
ensemble, 50/50 within-user z-blend. Promotion gated by Hermes on 3-split + paired McNemar.
Validation-only. NO Kaggle submission.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    load_train_interactions, load_pairs_csv, ensure_dir, write_json, evaluate_tophalf,
)

FLOOR = 0.684
EMB128_REF = 0.76505
NOISE = 0.0007


def emb128_uni_path(seed, split):
    if seed == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"


def load_emb128_ensemble(split):
    seeds = [42, 123, 2024, 7]
    paths = [emb128_uni_path(s, split) for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    base = pd.read_csv(paths[0])[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "e42"})
    for s, p in zip(seeds[1:], paths[1:]):
        d = pd.read_csv(p)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"e{s}"})
        base = base.merge(d, on="ID", how="inner")
    base["emb128"] = base[[f"e{s}" for s in seeds]].mean(axis=1)
    return base[["ID", "emb128"]]


class DINSetEncoder(nn.Module):
    def __init__(self, n_items, d, dropout=0.2):
        super().__init__()
        self.d = d
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        # target-attention scorer: f([q, k, q*k, q-k]) -> scalar weight
        self.att = nn.Sequential(
            nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )
        # final score MLP over [q, attended, q*attended, mean_pool, max_pool]
        self.score = nn.Sequential(
            nn.Linear(5 * d, 2 * d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1),
        )
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def user_repr(self, hist, target):
        """hist: (B, L) padded item ids (0=pad). target: (B,) candidate ids. -> score (B,)."""
        q = self.item_emb(target)                 # (B, d)
        H = self.item_emb(hist)                    # (B, L, d)
        mask = (hist != 0).float()                 # (B, L)
        B, L, d = H.shape
        qx = q.unsqueeze(1).expand(B, L, d)        # (B, L, d)
        att_in = torch.cat([qx, H, qx * H, qx - H], dim=-1)   # (B, L, 4d)
        w = self.att(att_in).squeeze(-1)           # (B, L)
        w = w.masked_fill(mask == 0, -1e9)
        a = torch.softmax(w, dim=1).unsqueeze(-1)  # (B, L, 1)
        attended = (a * H).sum(1)                  # (B, d) candidate-conditioned user vec
        # global pools (mask-aware)
        denom = mask.sum(1, keepdim=True).clamp_min(1.0)
        mean_pool = (H * mask.unsqueeze(-1)).sum(1) / denom
        Hm = H.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        max_pool = Hm.max(1).values
        feat = torch.cat([q, attended, q * attended, mean_pool, max_pool], dim=-1)  # (B, 5d)
        return self.score(feat).squeeze(-1)        # (B,)


def build_history(tr, item2idx):
    seqs = {}
    for u, g in tr.groupby("userID", sort=False):
        seqs[u] = [item2idx[x] for x in g["gameID"].tolist() if x in item2idx]
    return seqs


def pad_left(ids, maxlen):
    ids = ids[-maxlen:]
    out = [0] * (maxlen - len(ids)) + ids
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--pos-per-user", type=int, default=4, help="leave-one-out targets sampled per user per epoch")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="artifacts/din_set")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = f"din_d{args.emb_dim}_L{args.maxlen}_seed{args.seed}"

    sp = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp / "train_interactions.csv")
    cand = load_pairs_csv(sp / "candidates.csv")

    items = sorted(tr["gameID"].unique().tolist())
    item2idx = {g: i + 1 for i, g in enumerate(items)}
    n_items = len(item2idx)
    seqs = build_history(tr, item2idx)
    users = [u for u in seqs if len(seqs[u]) >= 2]
    seen = {u: set(seqs[u]) for u in users}
    print(f"[{tag}] {args.split}: {len(users)} users, {n_items} items, "
          f"{sum(len(v) for v in seqs.values())} interactions", flush=True)

    model = DINSetEncoder(n_items, args.emb_dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        # build leave-one-out training examples for this epoch
        hist_rows, tgt_rows, neg_rows = [], [], []
        for u in users:
            s = seqs[u]
            k = min(args.pos_per_user, len(s))
            picks = rng.choice(len(s), size=k, replace=False)
            for pi in picks:
                pos = s[pi]
                hist = s[:pi] + s[pi + 1:]          # leave-one-out: exclude target
                if not hist:
                    continue
                neg = int(rng.integers(1, n_items + 1))
                while neg in seen[u]:
                    neg = int(rng.integers(1, n_items + 1))
                hist_rows.append(pad_left(hist, args.maxlen))
                tgt_rows.append(pos); neg_rows.append(neg)
        H = torch.LongTensor(hist_rows).to(device)
        P = torch.LongTensor(tgt_rows).to(device)
        N = torch.LongTensor(neg_rows).to(device)
        perm = torch.randperm(len(H), device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, len(H), args.batch_size):
            idx = perm[i:i + args.batch_size]
            hb, pb, nb_ = H[idx], P[idx], N[idx]
            pos_s = model.user_repr(hb, pb)
            neg_s = model.user_repr(hb, nb_)
            loss = -F.logsigmoid(pos_s - neg_s).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item()); nb += 1
        avg = ep_loss / max(nb, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{tag}] epoch {epoch+1}/{args.epochs} loss={avg:.6f} elapsed={time.time()-started:.1f}s", flush=True)
        if not np.isfinite(avg):
            raise FloatingPointError(f"non-finite loss at epoch {epoch+1}")

    # ---- score candidates: history = FULL train set of the user, target = candidate ----
    model.eval()
    full_hist = {u: pad_left(seqs[u], args.maxlen) for u in seqs}
    cand = cand.copy()
    cu = cand["userID"].to_numpy(); cg = cand["gameID"].to_numpy()
    scores = np.full(len(cand), np.nan, dtype=np.float64)
    bs = 4096
    rows = []
    for r in range(len(cand)):
        u = cu[r]; g = cg[r]
        if u in full_hist and g in item2idx:
            rows.append((r, full_hist[u], item2idx[g]))
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            H = torch.LongTensor([c[1] for c in chunk]).to(device)
            T = torch.LongTensor([c[2] for c in chunk]).to(device)
            s = model.user_repr(H, T).cpu().numpy()
            for (r, _, _), v in zip(chunk, s):
                scores[r] = float(v)
    n_cold = int(np.isnan(scores).sum())
    scores = np.where(np.isnan(scores), -1e9, scores)
    cand["score_lightgcn"] = scores

    out = ensure_dir(Path(args.out_dir) / tag / args.split)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out / "lightgcn_scores.csv", index=False)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    solo = round(float(summ["row_accuracy"]), 5)

    ref = load_emb128_ensemble(args.split)
    corr_z = float("nan"); eq_blend = None
    if ref is not None:
        m = cand[["ID", "userID", "Label", "score_lightgcn"]].merge(ref, on="ID", how="inner")

        def zwu(df, col):
            g = df.groupby("userID")[col]
            return (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)

        m["zs"] = zwu(m, "score_lightgcn"); m["ze"] = zwu(m, "emb128")
        corr_z = round(float(np.corrcoef(m["zs"], m["ze"])[0, 1]), 4)
        m["blend"] = 0.5 * m["zs"] + 0.5 * m["ze"]
        eb, _ = evaluate_tophalf(m, "blend", label_col="Label", user_col="userID", id_col="ID")
        eq_blend = round(float(eb["row_accuracy"]), 5)

    if solo < FLOOR:
        tier = "REJECT_FLOOR"
        reason = f"solo_acc {solo:.5f} < floor {FLOOR}: set encoder failed to rank. Terminate."
    elif eq_blend is not None and eq_blend > EMB128_REF + NOISE and (np.isnan(corr_z) or corr_z < 0.9):
        tier = "SIGNAL_ESCALATE"
        reason = (f"eq_blend {eq_blend:.5f} > emb128_ref {EMB128_REF}+{NOISE} AND corr_z {corr_z} < 0.9: "
                  f"target-conditioned set signal adds orthogonal value. Promotion candidate (Hermes gates 3-split+paired).")
    else:
        tier = "REDUNDANT"
        reason = (f"solo_acc {solo:.5f} >= floor but (eq_blend {eq_blend} <= {EMB128_REF}+{NOISE} OR "
                  f"corr_z {corr_z} >= 0.9): conditional-set info redundant with graph CF. Terminate.")

    summary = {
        "note": "DIN-style target-conditioned set encoder probe. Validation-only. No Kaggle submission.",
        "paradigm": "GPT-5.5 research direction #1 (target attention, candidate-specific user repr)",
        "split": args.split,
        "config": {"emb_dim": args.emb_dim, "maxlen": args.maxlen, "dropout": args.dropout,
                   "lr": args.lr, "reg": args.reg, "epochs": args.epochs,
                   "pos_per_user": args.pos_per_user, "seed": args.seed},
        "n_users": len(users), "n_items": n_items, "n_cold_candidates": n_cold,
        "solo_acc": solo, "corr_z_vs_emb128": corr_z, "eq_blend_acc": eq_blend,
        "floor": FLOOR, "emb128_ref": EMB128_REF, "noise": NOISE,
        "eq_blend_minus_emb128_ref": (round(eq_blend - EMB128_REF, 5) if eq_blend is not None else None),
        "tier": tier, "tier_reason": reason, "solo_summary": summ,
    }
    write_json(out / "summary.json", summary)

    print("\n" + "=" * 80)
    print(f"[{tag}] solo_acc={solo}  corr_z={corr_z}  eq_blend={eq_blend} "
          f"(Δ vs emb128 {round((eq_blend - EMB128_REF),5) if eq_blend is not None else 'n/a'})")
    print(f"[{tag}] floor={FLOOR} emb128_ref={EMB128_REF} noise={NOISE}")
    print(f"[{tag}] TIER = {tier}")
    print(f"[{tag}] {reason}")
    print("=" * 80)
    print(f"summary.json: {out / 'summary.json'}")
    print(f"DIN_SET_DONE: {out / 'summary.json'} tier={tier}")


if __name__ == "__main__":
    main()
