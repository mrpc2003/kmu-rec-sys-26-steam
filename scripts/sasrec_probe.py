#!/usr/bin/env python3
"""SASRec (self-attentive sequential) probe for KMU RecSys 26 Steam played prediction.

NEW PARADIGM (vs every closed CF-scoring axis)
----------------------------------------------
Every axis closed so far — LightGCN / SGL / DirectAU / DNS / xSimGCL / item-CF / EASE /
ALS / MultiVAE / capacity / hard-neg / hyperbolic geometry — is an ORDER-FREE collaborative
scorer: it treats a user as a *set* of items and produces a (user,item) affinity. They have
all saturated at uniform ~0.765.

SASRec (Kang & McAuley, ICDM 2018) is a fundamentally different inductive bias: it encodes the
user's PLAY ORDER with a causal self-attention Transformer and predicts the next item
autoregressively. The user representation is the final hidden state after consuming the
time-ordered sequence. EDA confirms this is viable here: median sequence length 21, 97.7% of
users have >=10 plays, dates present for 100% of rows (99.3% orderable).

Clean isolation of the "sequence" hypothesis
---------------------------------------------
The candidate SCORE is the SAME Euclidean inner product as LightGCN:  s(u,i) = <h_u, emb_i>.
The ONLY thing that changes is how h_u is built: graph neighbour-averaging (LightGCN) vs
causal self-attention over the time-ordered sequence (SASRec). So a gain here is attributable
purely to ORDER information that order-free CF discards.

Leakage / trap guards (lessons from stacker 0.76245->0.75355 and candidate-marginal trap)
------------------------------------------------------------------------------------------
1. Sequences are built ONLY from the split's train_interactions.csv (held-out positives are
   excluded), exactly the LightGCN train protocol — no peeking at validation labels.
2. Training negatives are UNIFORM-UNSEEN (memory: public LB tracks the uniform-negative split),
   matching the BPR sampler family.
3. No (u,i)-level review-side features (hours/text on the candidate) — those cannot exist for
   test pairs and are the candidate-marginal popularity trap. Only the item-id sequence is used.
4. Same gate as the hyperbolic probe: solo_acc vs popularity floor, corr_z vs emb128 4-seed
   ensemble, 50/50 within-user z-blend. Promotion to a real candidate is gated by Hermes on a
   3-split panel + paired McNemar (single-split NOISE=0.0007 is below MDE=0.00355).

Validation-only. NO Kaggle submission, NO submission file written.
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

FLOOR = 0.684          # popularity baseline
EMB128_REF = 0.76505   # emb128 4-seed ensemble uniform (public 0.77745)
NOISE = 0.0007


# ----------------------------- emb128 reference -----------------------------
def emb128_uni_path(seed: int, split: str) -> Path:
    if seed == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"


def load_emb128_ensemble(split: str) -> pd.DataFrame | None:
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


# ----------------------------- SASRec model -----------------------------
class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int, maxlen: int, n_blocks: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_items = n_items
        self.d = d
        self.maxlen = maxlen
        # item id 0 = padding; real items 1..n_items
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d, eps=1e-8),
                "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d, eps=1e-8),
                "ff1": nn.Linear(d, d),
                "ff2": nn.Linear(d, d),
                "ff_drop": nn.Dropout(dropout),
            }) for _ in range(n_blocks)
        ])
        self.last_ln = nn.LayerNorm(d, eps=1e-8)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def seq_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) padded item ids (0=pad), left-padded so the last col is most recent.
        Returns per-position hidden states (B, L, d)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) * (self.d ** 0.5) + self.pos_emb(positions)
        x = self.emb_drop(x)
        pad_mask = (seq == 0)  # (B, L) True where pad -> ignored as keys
        # causal mask (L, L): True = disallowed (upper triangle, strictly future)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            q = blk["ln1"](x)
            attn_out, _ = blk["attn"](q, q, q, attn_mask=causal,
                                      key_padding_mask=pad_mask, need_weights=False)
            x = x + attn_out
            y = blk["ln2"](x)
            y = blk["ff2"](blk["ff_drop"](F.relu(blk["ff1"](y))))
            x = x + blk["ff_drop"](y)
        x = self.last_ln(x)
        # zero out pad positions so they never contribute downstream
        x = x * (~pad_mask).unsqueeze(-1).float()
        return x

    def final_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """User representation = hidden state at the LAST (most recent) real position.
        With left padding the most recent item is at column L-1."""
        h = self.seq_repr(seq)            # (B, L, d)
        return h[:, -1, :]                # (B, d)


# ----------------------------- sequence building -----------------------------
def build_sequences(tr: pd.DataFrame, maxlen: int):
    """Return: item2idx, per-user padded sequence array, user list.
    Order by (date, row_idx) ascending; keep the last `maxlen` items (left-padded)."""
    tr = tr.copy()
    tr["date"] = tr["date"].astype(str)
    sort_key = ["userID", "date"]
    if "row_idx" in tr.columns:
        sort_key.append("row_idx")
    tr = tr.sort_values(sort_key, kind="mergesort")

    items = sorted(tr["gameID"].unique().tolist())
    item2idx = {g: i + 1 for i, g in enumerate(items)}  # 0 = pad

    seqs = {}
    for u, g in tr.groupby("userID", sort=False):
        ids = [item2idx[x] for x in g["gameID"].tolist()]
        seqs[u] = ids
    return item2idx, seqs


def make_train_matrix(seqs: dict, users: list, maxlen: int):
    """Build (N, maxlen) left-padded input and target arrays for next-item training.
    Input  = items[:-1] (last maxlen), target = items[1:] aligned per position."""
    X = np.zeros((len(users), maxlen), dtype=np.int64)
    Y = np.zeros((len(users), maxlen), dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u]
        if len(s) < 2:
            continue
        inp = s[:-1][-maxlen:]
        tgt = s[1:][-maxlen:]
        X[r, maxlen - len(inp):] = inp
        Y[r, maxlen - len(tgt):] = tgt
    return X, Y


def make_score_matrix(seqs: dict, users: list, maxlen: int):
    """Full sequence (last maxlen, left-padded) for inference user representation."""
    S = np.zeros((len(users), maxlen), dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u][-maxlen:]
        S[r, maxlen - len(s):] = s
    return S


# ----------------------------- main -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="val_random_uniform_seed42")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="artifacts/sasrec")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = f"sasrec_d{args.emb_dim}_b{args.n_blocks}_h{args.n_heads}_L{args.maxlen}_seed{args.seed}"

    sp = ROOT / "artifacts/validation" / args.split
    tr = load_train_interactions(sp / "train_interactions.csv")
    cand = load_pairs_csv(sp / "candidates.csv")

    item2idx, seqs = build_sequences(tr, args.maxlen)
    n_items = len(item2idx)
    users = list(seqs.keys())
    print(f"[{tag}] {args.split}: {len(users)} users, {n_items} items, "
          f"{sum(len(v) for v in seqs.values())} interactions", flush=True)

    X, Y = make_train_matrix(seqs, users, args.maxlen)

    Xt = torch.LongTensor(X).to(device)
    Yt = torch.LongTensor(Y).to(device)
    # per-user seen-item set for uniform-unseen negative sampling
    seen = [set(seqs[u]) for u in users]

    model = SASRec(n_items, args.emb_dim, args.maxlen, args.n_blocks, args.n_heads, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.reg)

    n = len(users)
    n_batches = max(1, n // args.batch_size)
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            if len(idx) == 0:
                continue
            xb = Xt[idx]            # (B, L)
            yb = Yt[idx]            # (B, L) positive next items (0 = pad/no-target)
            B, L = xb.shape
            # uniform-unseen negatives, one per position
            neg = np.zeros((B, L), dtype=np.int64)
            for bi, ui in enumerate(idx):
                s = seen[ui]
                m = (yb[bi] != 0).cpu().numpy()
                k = int(m.sum())
                if k == 0:
                    continue
                draws = rng.integers(1, n_items + 1, size=k * 2)
                draws = [d for d in draws if d not in s][:k]
                while len(draws) < k:
                    d = int(rng.integers(1, n_items + 1))
                    if d not in s:
                        draws.append(d)
                neg[bi, m] = np.array(draws[:k], dtype=np.int64)
            negt = torch.LongTensor(neg).to(device)

            h = model.seq_repr(xb)                       # (B, L, d)
            pos_e = model.item_emb(yb)                   # (B, L, d)
            neg_e = model.item_emb(negt)                 # (B, L, d)
            pos_logit = (h * pos_e).sum(-1)              # (B, L)
            neg_logit = (h * neg_e).sum(-1)
            mask = (yb != 0).float()                     # only real target positions
            loss = -(F.logsigmoid(pos_logit) * mask + F.logsigmoid(-neg_logit) * mask).sum() / mask.sum().clamp_min(1.0)

            opt.zero_grad()
            loss.backward()
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if not finite:
                opt.zero_grad(); continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item())
        avg = ep_loss / n_batches
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{tag}] epoch {epoch+1}/{args.epochs} loss={avg:.6f} elapsed={time.time()-started:.1f}s", flush=True)
        if not np.isfinite(avg):
            raise FloatingPointError(f"non-finite loss at epoch {epoch+1}")

    # ---------------- score candidates ----------------
    model.eval()
    Smat = make_score_matrix(seqs, users, args.maxlen)
    user2row = {u: r for r, u in enumerate(users)}
    with torch.no_grad():
        Sten = torch.LongTensor(Smat).to(device)
        H = torch.zeros(len(users), args.emb_dim, device=device)
        bs = 1024
        for i in range(0, len(users), bs):
            H[i:i+bs] = model.final_hidden(Sten[i:i+bs])
        H = H.cpu().numpy()
        item_w = model.item_emb.weight.detach().cpu().numpy()  # (n_items+1, d)

    cand = cand.copy()
    scores = np.full(len(cand), np.nan, dtype=np.float64)
    cu = cand["userID"].to_numpy()
    cg = cand["gameID"].to_numpy()
    for r in range(len(cand)):
        ur = user2row.get(cu[r])
        ii = item2idx.get(cg[r])
        if ur is None or ii is None:
            continue
        scores[r] = float(H[ur] @ item_w[ii])
    # cold (unknown user/item) -> very low score so top-half ranking still works
    n_cold = int(np.isnan(scores).sum())
    scores = np.where(np.isnan(scores), -1e9, scores)
    cand["score_lightgcn"] = scores

    out = ensure_dir(Path(args.out_dir) / tag / args.split)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out / "lightgcn_scores.csv", index=False)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    solo = round(float(summ["row_accuracy"]), 5)

    # ---------------- corr_z + eq_blend vs emb128 ----------------
    ref = load_emb128_ensemble(args.split)
    corr_z = float("nan"); eq_blend = None
    if ref is not None:
        m = cand[["ID", "userID", "Label", "score_lightgcn"]].merge(ref, on="ID", how="inner")

        def zwu(df, col):
            g = df.groupby("userID")[col]
            return (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)

        m["zs"] = zwu(m, "score_lightgcn")
        m["ze"] = zwu(m, "emb128")
        corr_z = round(float(np.corrcoef(m["zs"], m["ze"])[0, 1]), 4)
        m["blend"] = 0.5 * m["zs"] + 0.5 * m["ze"]
        eb, _ = evaluate_tophalf(m, "blend", label_col="Label", user_col="userID", id_col="ID")
        eq_blend = round(float(eb["row_accuracy"]), 5)

    # ---------------- tier ----------------
    if solo < FLOOR:
        tier = "REJECT_FLOOR"
        reason = f"solo_acc {solo:.5f} < floor {FLOOR}: sequence model failed to rank. Terminate."
    elif eq_blend is not None and eq_blend > EMB128_REF + NOISE and (np.isnan(corr_z) or corr_z < 0.9):
        tier = "SIGNAL_ESCALATE"
        reason = (f"eq_blend {eq_blend:.5f} > emb128_ref {EMB128_REF}+{NOISE} AND corr_z {corr_z} < 0.9: "
                  f"sequence geometry adds orthogonal value. Promotion candidate (Hermes gates on 3-split+paired).")
    else:
        tier = "GEOMETRY_REDUNDANT"
        reason = (f"solo_acc {solo:.5f} >= floor but (eq_blend {eq_blend} <= {EMB128_REF}+{NOISE} OR "
                  f"corr_z {corr_z} >= 0.9): sequence info redundant with order-free CF. Terminate.")

    summary = {
        "note": "SASRec self-attentive sequential probe. Validation-only. No Kaggle submission.",
        "split": args.split,
        "config": {"emb_dim": args.emb_dim, "n_blocks": args.n_blocks, "n_heads": args.n_heads,
                   "maxlen": args.maxlen, "dropout": args.dropout, "lr": args.lr, "reg": args.reg,
                   "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed},
        "n_users": len(users), "n_items": n_items, "n_cold_candidates": n_cold,
        "solo_acc": solo, "corr_z_vs_emb128": corr_z, "eq_blend_acc": eq_blend,
        "floor": FLOOR, "emb128_ref": EMB128_REF, "noise": NOISE,
        "eq_blend_minus_emb128_ref": (round(eq_blend - EMB128_REF, 5) if eq_blend is not None else None),
        "tier": tier, "tier_reason": reason,
        "solo_summary": summ,
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
    print(f"SASREC_DONE: {out / 'summary.json'} tier={tier}")


if __name__ == "__main__":
    main()
