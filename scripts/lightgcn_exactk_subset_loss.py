#!/usr/bin/env python3
"""Exact-K conditional subset-loss fine-tuning for LightGCN — UNIFORM gate.

WHY this is the FIRST genuinely orthogonal lever after BPR-LightGCN saturation
------------------------------------------------------------------------------
Every closed bet was either (a) a NEW ENCODER on the same binary graph (SimGCL,
Turbo-CF, AlphaRec, MultiVAE, capacity frontier) or (b) a weak side axis (text,
hours). corr_z(emb64,128)=0.9747 and corr_z(128,192)=0.986 prove the BPR-LightGCN
FAMILY converges to the same ranking regardless of dimension/seed. The reason is
the OBJECTIVE: every member optimizes the same pairwise AUC surrogate (BPR), whose
Bayes-optimal solution is identical given the same graph signal.

This script keeps the SAME emb128 encoder and graph signal but changes the LOSS
GEOMETRY. The competition metric is NOT global AUC — it is per-user "pick exactly
K_u = |C_u|/2 of the candidates". The matching maximum-likelihood objective is the
conditional subset probability over all (2K_u choose K_u) subsets:

    P(P_u | C_u, |P_u|=K_u) = exp(sum_{i in P_u} s_ui) / e_{K_u}(exp(s) : j in C_u)

where e_k is the elementary symmetric polynomial (DP, O(n*K)). The negative log-lik:

    L_u = -sum_{i in P_u} s_ui + log e_{K_u}(exp(s_uj) : j in C_u)

KEY FACT (why this can differ from BPR while staying in the same family):
  * K=1: subset loss == BPR exactly (-log sigma(s_pos - s_neg)). No difference.
  * K>=2: the symmetric polynomial COUPLES candidates. The gradient concentrates on
    the rank-K / rank-K+1 BOUNDARY (the marginal inclusion probability), instead of
    treating every pos-neg pair independently like BPR. Since pairs.csv K_u has
    median 2 (p95 5), most candidate sets are K>=2, so the gradient genuinely differs.

CONFOUND CONTROL (3-way branch from ONE canonical BPR init):
  0. pretrained : emb128 BPR for --pretrain-epochs (the canonical 0.77745 code path).
  A. bpr_ft     : + --ft-epochs more BPR    (control for "just trained longer").
  B. subset_ft  : + --ft-epochs subset loss (the variant under test).
  Comparing B vs A isolates LOSS GEOMETRY from EXTRA TRAINING.

Negatives during fine-tune are UNIFORM (the proven public surrogate), never hard.
Validation candidate sets are NEVER trained on (scored only).

Gate (established thresholds):
  Delta(B vs A) on uniform row-acc >= 0.003                 -> CANDIDATE
  0.001 <= Delta < 0.003 AND paired McNemar p < 0.05        -> WEAK CANDIDATE
  Delta < 0.001                                             -> NOISE / CLOSE

Validation-only. No Kaggle submission. Single seed first (then ensemble only if it gates).
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix, load_pairs_csv, load_train_interactions,
    evaluate_tophalf, ensure_dir, write_json,
)
from lightgcn_train import LightGCN, build_norm_adj, score_candidates  # noqa: E402

SPLIT = "val_random_uniform_seed42"
EMB128_SINGLE_SEED_REF = 0.76205   # emb128 L4 reg1e-3 seed42 uniform single-seed
NOISE = 0.0007


# --------------------------------------------------------------------------------------
# Exact-K subset loss
# --------------------------------------------------------------------------------------
def log_elementary_symmetric_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """log e_k(exp(scores)) per row, numerically stable.

    scores: [B, n]  (n = 2k candidate scores per user)
    returns: [B]  = log of the k-th elementary symmetric polynomial of {exp(s_j)}.

    Stable: factor out the per-row max m. e_k(exp(s)) = exp(k*m) * e_k(exp(s-m)),
    so log e_k = k*m + log(dp_k) where dp is built from x_j = exp(s_j - m) <= 1.
    DP is autograd-safe (functional list updates, no in-place on leaf tensors).
    """
    B, n = scores.shape
    assert n >= k >= 1
    m, _ = scores.max(dim=1, keepdim=True)          # [B,1]
    x = torch.exp(scores - m)                       # [B,n], each <= 1
    # dp[t] holds e_t of the prefix processed so far, shape [B]
    dp = [torch.ones(B, device=scores.device, dtype=scores.dtype)]
    dp += [torch.zeros(B, device=scores.device, dtype=scores.dtype) for _ in range(k)]
    for j in range(n):
        xj = x[:, j]
        new_dp = [dp[0]]
        for t in range(1, k + 1):
            new_dp.append(dp[t] + dp[t - 1] * xj)
        dp = new_dp
    log_dp_k = torch.log(dp[k].clamp_min(1e-30))
    return k * m.squeeze(1) + log_dp_k


def subset_nll(scores: torch.Tensor, pos_mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean negative log-likelihood of the exact-K subset.

    scores:   [B, 2k]   candidate scores
    pos_mask: [B, 2k]   exactly k ones per row marking the true positives
    """
    pos_term = (scores * pos_mask).sum(dim=1)               # [B] = sum of pos scores
    log_norm = log_elementary_symmetric_k(scores, k)        # [B]
    return (-pos_term + log_norm).mean()


# --------------------------------------------------------------------------------------
# Training-candidate sampling (synthetic exact-K episodes from fold-train)
# --------------------------------------------------------------------------------------
def build_user_pos_lists(mat: sp.csr_matrix) -> list[np.ndarray]:
    """Per-user array of positive item indices (from the fold-train graph)."""
    out = []
    for u in range(mat.shape[0]):
        s, e = mat.indptr[u], mat.indptr[u + 1]
        out.append(mat.indices[s:e].copy())
    return out


def sample_subset_batch(
    user_pos: list[np.ndarray],
    seen_sets: list[set],
    n_items: int,
    k: int,
    batch_users: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a homogeneous batch of exact-K candidate sets.

    Each row: k positives (sampled from the user's train items) + k UNIFORM negatives
    (sampled from unseen items). Returns:
      users     [B]
      cand_items[B, 2k]    (first k are positives, last k are negatives)
      pos_mask  [B, 2k]    (1 for the first k columns, 0 otherwise)
    Only users with >= k train positives are eligible.
    """
    eligible = [u for u in range(len(user_pos)) if len(user_pos[u]) >= k]
    chosen = rng.choice(eligible, size=batch_users, replace=True)
    cand = np.zeros((batch_users, 2 * k), dtype=np.int64)
    for r, u in enumerate(chosen):
        pos_pool = user_pos[u]
        pos = rng.choice(pos_pool, size=k, replace=False)
        seen = seen_sets[u]
        negs = np.empty(k, dtype=np.int64)
        filled = 0
        while filled < k:
            candidate = int(rng.integers(0, n_items))
            if candidate in seen or candidate in negs[:filled]:
                continue
            negs[filled] = candidate
            filled += 1
        cand[r, :k] = pos
        cand[r, k:] = negs
    pos_mask = np.zeros((batch_users, 2 * k), dtype=np.float32)
    pos_mask[:, :k] = 1.0
    return chosen.astype(np.int64), cand, pos_mask


def empirical_k_distribution(pairs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """K_u distribution from pairs.csv: K_u = candidate_count // 2."""
    cc = pairs_df.groupby("userID").size().to_numpy()
    ku = (cc // 2).astype(int)
    ku = ku[ku >= 1]
    vals, counts = np.unique(ku, return_counts=True)
    probs = counts / counts.sum()
    return vals, probs


# --------------------------------------------------------------------------------------
# Fine-tune loops
# --------------------------------------------------------------------------------------
def finetune(
    model: LightGCN,
    adj: torch.Tensor,
    mat: sp.csr_matrix,
    user_pos: list[np.ndarray],
    seen_sets: list[set],
    n_items: int,
    mode: str,                       # "bpr" or "subset"
    epochs: int,
    steps_per_epoch: int,
    batch_users: int,
    lr: float,
    reg: float,
    k_vals: np.ndarray,
    k_probs: np.ndarray,
    device: str,
    rng: np.random.Generator,
    torch_seed: int,
) -> dict:
    torch.manual_seed(torch_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    started = time.time()
    losses = []
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            k = int(rng.choice(k_vals, p=k_probs))
            users, cand, pos_mask = sample_subset_batch(
                user_pos, seen_sets, n_items, k, batch_users, rng)
            users_t = torch.from_numpy(users).to(device)
            cand_t = torch.from_numpy(cand).to(device)          # [B,2k]
            pm_t = torch.from_numpy(pos_mask).to(device)        # [B,2k]

            user_emb, item_emb = model(adj)
            u = user_emb[users_t]                                # [B,D]
            ci = item_emb[cand_t]                                # [B,2k,D]
            scores = torch.einsum("bd,bnd->bn", u, ci)           # [B,2k]

            if mode == "subset":
                main_loss = subset_nll(scores, pm_t, k)
            elif mode == "bpr":
                # pairwise BPR over the same k pos x k neg candidates (mean over all pairs)
                pos_s = scores[:, :k]                            # [B,k]
                neg_s = scores[:, k:]                            # [B,k]
                diff = pos_s.unsqueeze(2) - neg_s.unsqueeze(1)   # [B,k,k]
                main_loss = -F.logsigmoid(diff).mean()
            else:
                raise ValueError(mode)

            reg_loss = reg * (u.norm(2).pow(2) + ci.norm(2).pow(2)) / users.shape[0]
            loss = main_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
        losses.append(ep_loss / steps_per_epoch)
    return {"mode": mode, "epochs": epochs, "final_loss": float(losses[-1]),
            "seconds": round(time.time() - started, 1)}


def extract_embeddings(model: LightGCN, adj: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        ue, ie = model(adj)
    return ue.cpu().numpy(), ie.cpu().numpy()


def score_and_eval(cand_df, ue, ie, u2i, i2i, col):
    df = cand_df.copy()
    df[col] = score_candidates(df, ue, ie, u2i, i2i)
    summ, pred = evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")
    return summ, pred, df


def mcnemar(corr_a: np.ndarray, corr_b: np.ndarray) -> dict:
    """Paired McNemar on row-level correctness. b: B-correct & A-wrong; c: A-correct & B-wrong."""
    a = corr_a.astype(bool)
    b_ = corr_b.astype(bool)
    b = int(np.sum(b_ & ~a))      # variant fixes
    c = int(np.sum(a & ~b_))      # variant breaks
    n = b + c
    if n == 0:
        return {"b_variant_fixes": b, "c_variant_breaks": c, "discordant": 0,
                "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    # survival function of chi2 with 1 dof
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b_variant_fixes": b, "c_variant_breaks": c, "discordant": n,
            "chi2": round(chi2, 4), "p_value": round(p, 5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=5e-4, help="fine-tune lr (lower than pretrain)")
    ap.add_argument("--pretrain-epochs", type=int, default=200)
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--batch-users", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4096, help="pretrain BPR batch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="artifacts/exactk_subset")
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir) / SPLIT)
    sp_dir = ROOT / "artifacts/validation" / SPLIT
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    pairs_full = load_pairs_csv(ROOT / "data/raw/public/data/pairs.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"[exactk] {SPLIT}: {n_users}u {n_items}i {mat.nnz}nnz", flush=True)

    k_vals, k_probs = empirical_k_distribution(pairs_full)
    print(f"[exactk] empirical K_u dist: {dict(zip(k_vals.tolist(), np.round(k_probs,3).tolist()))}", flush=True)

    device = args.device
    adj = build_norm_adj(mat, n_users, n_items).to(device)
    user_pos = build_user_pos_lists(mat)
    seen_sets = [set(arr.tolist()) for arr in user_pos]
    rng = np.random.default_rng(args.seed)

    # ---- Phase 0: canonical BPR pretrain ----------------------------------------------
    torch.manual_seed(args.seed)
    model = LightGCN(n_users, n_items, args.emb_dim, args.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    from lightgcn_train import sample_bpr_batch
    n_batches = max(1, mat.nnz // args.batch_size)
    t0 = time.time()
    for epoch in range(args.pretrain_epochs):
        model.train()
        for _ in range(n_batches):
            us, ps, ns = sample_bpr_batch(mat, args.batch_size, n_items, rng)
            ut = torch.from_numpy(us).long().to(device)
            pt = torch.from_numpy(ps).to(device)
            nt = torch.from_numpy(ns).to(device)
            ue, ie = model(adj)
            ps_ = (ue[ut] * ie[pt]).sum(1)
            ns_ = (ue[ut] * ie[nt]).sum(1)
            bpr = -F.logsigmoid(ps_ - ns_).mean()
            rl = args.reg * (model.user_emb(ut).norm(2).pow(2) + model.item_emb(pt).norm(2).pow(2)
                             + model.item_emb(nt).norm(2).pow(2)) / args.batch_size
            loss = bpr + rl
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  [pretrain] {epoch+1}/{args.pretrain_epochs} loss={loss.item():.5f} {time.time()-t0:.0f}s", flush=True)
    pretrain_state = copy.deepcopy(model.state_dict())
    print(f"[exactk] pretrain done {time.time()-t0:.0f}s", flush=True)

    # score pretrained
    ue0, ie0 = extract_embeddings(model, adj)
    summ0, pred0, _ = score_and_eval(cand, ue0, ie0, u2i, i2i, "score")

    # ---- Phase A: BPR fine-tune control -----------------------------------------------
    model_a = LightGCN(n_users, n_items, args.emb_dim, args.n_layers).to(device)
    model_a.load_state_dict(copy.deepcopy(pretrain_state))
    meta_a = finetune(model_a, adj, mat, user_pos, seen_sets, n_items, "bpr",
                      args.ft_epochs, args.steps_per_epoch, args.batch_users, args.ft_lr,
                      args.reg, k_vals, k_probs, device, np.random.default_rng(args.seed + 1), args.seed + 1)
    ue_a, ie_a = extract_embeddings(model_a, adj)
    summ_a, pred_a, _ = score_and_eval(cand, ue_a, ie_a, u2i, i2i, "score")

    # ---- Phase B: subset-loss fine-tune variant ---------------------------------------
    model_b = LightGCN(n_users, n_items, args.emb_dim, args.n_layers).to(device)
    model_b.load_state_dict(copy.deepcopy(pretrain_state))
    meta_b = finetune(model_b, adj, mat, user_pos, seen_sets, n_items, "subset",
                      args.ft_epochs, args.steps_per_epoch, args.batch_users, args.ft_lr,
                      args.reg, k_vals, k_probs, device, np.random.default_rng(args.seed + 1), args.seed + 1)
    ue_b, ie_b = extract_embeddings(model_b, adj)
    summ_b, pred_b, df_b = score_and_eval(cand, ue_b, ie_b, u2i, i2i, "score")

    # ---- align pred rows by ID for paired McNemar -------------------------------------
    pa = pred_a.sort_values("ID")["Correct"].to_numpy()
    pb = pred_b.sort_values("ID")["Correct"].to_numpy()
    p0 = pred0.sort_values("ID")["Correct"].to_numpy()
    mc_b_vs_a = mcnemar(pa, pb)
    mc_b_vs_0 = mcnemar(p0, pb)

    acc0 = round(float(summ0["row_accuracy"]), 5)
    acc_a = round(float(summ_a["row_accuracy"]), 5)
    acc_b = round(float(summ_b["row_accuracy"]), 5)
    d_b_a = round(acc_b - acc_a, 5)     # loss-geometry effect (isolated)
    d_b_0 = round(acc_b - acc0, 5)      # subset vs pretrained
    d_a_0 = round(acc_a - acc0, 5)      # extra-training effect

    if d_b_a >= 0.003:
        tier = "SUBSET_GAIN_CANDIDATE"
    elif d_b_a >= 0.001 and mc_b_vs_a["p_value"] < 0.05:
        tier = "SUBSET_WEAK_CANDIDATE_MCNEMAR"
    elif d_b_a >= -NOISE:
        tier = "SUBSET_NO_GAIN_NOISE"
    else:
        tier = "SUBSET_REGRESS"

    verdict = (f"subset_ft {acc_b} vs bpr_ft {acc_a} (loss-geometry isolated) = {d_b_a:+.5f}; "
               f"vs pretrained {acc0} = {d_b_0:+.5f}; bpr_ft extra-train effect = {d_a_0:+.5f}; "
               f"McNemar(B vs A) fixes={mc_b_vs_a['b_variant_fixes']} breaks={mc_b_vs_a['c_variant_breaks']} "
               f"p={mc_b_vs_a['p_value']}")

    result = {
        "split": SPLIT, "emb_dim": args.emb_dim, "n_layers": args.n_layers, "reg": args.reg,
        "seed": args.seed, "ft_epochs": args.ft_epochs, "steps_per_epoch": args.steps_per_epoch,
        "batch_users": args.batch_users, "ft_lr": args.ft_lr,
        "k_distribution": {int(k): float(p) for k, p in zip(k_vals, k_probs)},
        "acc_pretrained": acc0, "acc_bpr_ft": acc_a, "acc_subset_ft": acc_b,
        "delta_subset_vs_bprft_ISOLATED": d_b_a,
        "delta_subset_vs_pretrained": d_b_0,
        "delta_bprft_vs_pretrained_EXTRA_TRAIN": d_a_0,
        "mcnemar_subset_vs_bprft": mc_b_vs_a,
        "mcnemar_subset_vs_pretrained": mc_b_vs_0,
        "noise_band": NOISE, "tier": tier, "verdict": verdict,
        "meta_bpr_ft": meta_a, "meta_subset_ft": meta_b,
        "emb128_single_seed_ref": EMB128_SINGLE_SEED_REF,
    }
    write_json(out / "summary.json", result)
    df_b[["ID", "userID", "gameID", "Label", "score"]].to_csv(out / "subset_ft_scores.csv", index=False)
    print("\n" + "=" * 80, flush=True)
    print(f"[exactk] pretrained={acc0}  bpr_ft={acc_a}  subset_ft={acc_b}", flush=True)
    print(f"[exactk] ISOLATED loss-geometry Δ(subset−bpr_ft) = {d_b_a:+.5f}  (noise band ±{NOISE})", flush=True)
    print(f"[exactk] extra-train Δ(bpr_ft−pretrained) = {d_a_0:+.5f}", flush=True)
    print(f"[exactk] McNemar B vs A: fixes={mc_b_vs_a['b_variant_fixes']} breaks={mc_b_vs_a['c_variant_breaks']} p={mc_b_vs_a['p_value']}", flush=True)
    print(f"[exactk] TIER = {tier}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
