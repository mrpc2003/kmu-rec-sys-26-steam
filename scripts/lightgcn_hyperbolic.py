#!/usr/bin/env python3
"""Hyperbolic (Lorentz) LightGCN probe for KMU RecSys 26 Steam played prediction.

This mirrors `scripts/lightgcn_train.py` but swaps ONLY the decision geometry from
Euclidean inner product to the Lorentz hyperboloid, while KEEPING the strong
ranking objective (BPR / triplet) and the EXACT same uniform-unseen negative
sampler.  This isolates the A.2 hypothesis: "strong ranking loss, non-Euclidean
(hyperbolic) decision geometry only" — the one combination never yet tested.

Design (HGCF / standard hyperbolic-LightGCN pattern):
  1. Embeddings live on the Lorentz manifold (geoopt 0.5.1 `geoopt.Lorentz`,
     ManifoldParameter), initialised near the origin (head region) so training is
     free to push niche/tail items toward the boundary — the hierarchy encoding
     the hypothesis relies on.
  2. Graph propagation is done in the TANGENT SPACE AT THE ORIGIN:
       tan0 = logmap0(ego)            # tangent vectors at o, time-coord == 0
       T^(k) = A_norm @ T^(k-1)       # LightGCN propagation (linear, keeps v_0==0)
       T_mean = mean(T^0 .. T^L)      # LightGCN layer-mean, in tangent space
       final = expmap0(T_mean)        # back onto the manifold
     A_norm is the SAME symmetric-normalised adjacency as Euclidean LightGCN
     (`build_norm_adj` reused verbatim). Because every tangent vector at the
     origin has a zero time component and A_norm @ (.) mixes ROWS (nodes) not
     COLUMNS (coords), the zero time component is preserved through every layer,
     so `expmap0` receives a valid tangent vector.
  3. Score is the NEGATIVE Lorentz geodesic distance:  s(u,i) = -d_L(u, i).
  4. Loss is hyperbolic BPR / triplet (NOT InfoNCE):
       L = -logsigmoid( s(u,pos) - s(u,neg) )
         = -logsigmoid( d_L(u,neg) - d_L(u,pos) )
     negatives are uniform-unseen, identical to `sample_bpr_batch`.
  5. Optimiser: geoopt `RiemannianAdam` with `stabilize=1` (periodic on-manifold
     re-projection of the parameters).

Numerical stability (the part Hermes audits hardest):
  * The geodesic distance is HAND-ROLLED here rather than calling geoopt's
    `manifold.dist`.  geoopt's `arcosh` only clamps `arg**2 - 1 >= 1e-15`, which
    leaves an effective gradient magnitude near coincident points of ~3e7 and is
    exactly the blow-up the brief warns about.  Instead we clamp the arccosh
    ARGUMENT to `>= 1 + ARCOSH_EPS` (a hard floor on the cosh-distance), compute
    the Minkowski inner product and arccosh in float64 to avoid the catastrophic
    cancellation of `-x0*y0 + <x_s, y_s>`, then cast back.  `clamp(min=1+eps)`
    zeroes the gradient for sub-eps-close pairs (which carry no useful ranking
    signal anyway) instead of exploding it.
  * `expmap0` norm clamp is handled by geoopt's `_norm` (`clamp_min(.,1e-8)` ->
    norm >= 1e-4), so zero / origin tangent vectors map cleanly back to the
    origin with no NaN.
  * After propagation the manifold embeddings are re-projected with `projx`
    before scoring to absorb float drift off the hyperboloid.

Validation-only. NO Kaggle submission, NO submission file written.

References:
- He et al., LightGCN, SIGIR 2020.
- Sun et al., HGCF: Hyperbolic Graph Convolution Networks for Collaborative
  Filtering, WWW 2021.
- Nickel & Kiela, Learning Continuous Hierarchies in the Lorentz Model, ICML 2018.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    write_json,
)
from lightgcn_train import build_norm_adj, sample_bpr_batch  # noqa: E402

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SPLIT = "val_random_uniform_seed42"

FLOOR = 0.684
EMB128_REF = 0.76505
NOISE = 0.0007

# cosh(d) >= 1 + eps  =>  d >= acosh(1+eps) ~ sqrt(2*eps); floors the arccosh arg.
ARCOSH_EPS = 1e-6

# Caps the hyperbolic radius so cosh(radius) stays far inside the float32 range
# (overflow near radius 89); prevents the late-training expmap/logmap NaN blow-up.
MAX_TAN_NORM = 30.0

E128_PATHS = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}


def lorentz_neg_dist(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Negative Lorentz geodesic distance  -d_L(x, y),  shape-broadcast over rows.

    d_L^k(x, y) = sqrt(k) * arccosh( -<x, y>_L / k ),  <.,.>_L Minkowski inner.

    Numerically safe: float64 inner product (avoids -x0*y0 + <xs,ys> cancellation),
    arccosh argument hard-clamped to >= 1 + ARCOSH_EPS, arccosh = log(a+sqrt(a^2-1)).
    """
    xf = x.double()
    yf = y.double()
    kf = k.double()
    prod = xf * yf
    mink = -prod[..., 0:1].sum(dim=-1) + prod[..., 1:].sum(dim=-1)
    arg = (-mink) / kf
    arg = torch.clamp(arg, min=1.0 + ARCOSH_EPS)
    arcosh = torch.log(arg + torch.sqrt(arg * arg - 1.0))
    dist = torch.sqrt(kf) * arcosh
    return (-dist).to(x.dtype)


def clamp_tan_norm(tan: torch.Tensor, max_norm: float) -> torch.Tensor:
    spatial = tan[..., 1:]
    norm = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (max_norm / norm).clamp_max(1.0)
    out = tan.clone()
    out[..., 1:] = spatial * scale
    return out


class HyperbolicLightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, n_layers: int,
                 manifold: geoopt.Lorentz, init_scale: float = 1e-3, seed: int = 42):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.manifold = manifold
        gen = torch.Generator().manual_seed(seed)
        n = n_users + n_items
        tan = torch.zeros(n, emb_dim + 1)
        tan[:, 1:] = init_scale * torch.randn(n, emb_dim, generator=gen)
        pts = manifold.expmap0(tan)
        self.embedding = ManifoldParameter(pts, manifold=manifold)

    def propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego = self.embedding
        tan0 = self.manifold.logmap0(ego)
        acc = tan0
        cur = tan0
        for _ in range(self.n_layers):
            cur = torch.sparse.mm(adj, cur)
            acc = acc + cur
        tan_mean = acc / (self.n_layers + 1)
        tan_mean = clamp_tan_norm(tan_mean, MAX_TAN_NORM)
        final = self.manifold.expmap0(tan_mean)
        final = self.manifold.projx(final)
        user_final = final[: self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final, tan0


def train(
    interaction_matrix: sp.csr_matrix,
    n_users: int,
    n_items: int,
    emb_dim: int,
    n_layers: int,
    lr: float,
    reg: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, geoopt.Lorentz, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    manifold = geoopt.Lorentz(k=1.0)
    k_t = torch.tensor(1.0, device=device)

    adj = build_norm_adj(interaction_matrix, n_users, n_items).to(device)
    model = HyperbolicLightGCN(n_users, n_items, emb_dim, n_layers, manifold,
                               init_scale=1e-3, seed=seed).to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=lr, stabilize=1)

    n_interactions = interaction_matrix.nnz
    n_batches = max(1, n_interactions // batch_size)
    losses: list[float] = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_bpr_batch(
                interaction_matrix, batch_size, n_items, rng)
            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos_items).to(device)
            neg_t = torch.LongTensor(neg_items).to(device)

            user_emb, item_emb, tan0 = model.propagate(adj)
            u_emb = user_emb[users_t]
            p_emb = item_emb[pos_t]
            n_emb = item_emb[neg_t]

            pos_scores = lorentz_neg_dist(u_emb, p_emb, k_t)
            neg_scores = lorentz_neg_dist(u_emb, n_emb, k_t)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            tu = tan0[users_t]
            tp = tan0[n_users + pos_t]
            tn = tan0[n_users + neg_t]
            reg_loss = reg * (
                tu.pow(2).sum(dim=1).mean()
                + tp.pow(2).sum(dim=1).mean()
                + tn.pow(2).sum(dim=1).mean()
            )

            loss = bpr_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            # Root-cause fix for the epoch~140-152 NaN blow-up. The Lorentz distance
            # gradient factor 1/sqrt(arg^2-1) explodes (~707 at the arccosh-arg floor,
            # unbounded above it) for rare near-coincident (u,i) pairs, even though the
            # distance VALUE is clamped. Two scientifically-neutral GNN stabilizers:
            #   (1) skip any batch whose grads are non-finite (drop a rare inf/nan batch
            #       instead of poisoning the embedding), and
            #   (2) clip the global grad norm to cap ordinary spikes.
            # Neither changes the geometry hypothesis being tested.
            finite = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if not finite:
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                tan = manifold.logmap0(model.embedding)
                tan = clamp_tan_norm(tan, MAX_TAN_NORM)
                model.embedding.copy_(manifold.expmap0(tan))
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [HypLightGCN] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
        if not np.isfinite(avg_loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch+1}: {avg_loss}")

    model.eval()
    with torch.no_grad():
        user_final, item_final, _ = model.propagate(adj)
    user_np = user_final.detach().cpu().numpy()
    item_np = item_final.detach().cpu().numpy()

    meta = {
        "geometry": "lorentz_hyperbolic",
        "manifold_k": 1.0,
        "emb_dim": emb_dim,
        "ambient_dim": emb_dim + 1,
        "n_layers": n_layers,
        "lr": lr,
        "reg": reg,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "arcosh_eps": ARCOSH_EPS,
        "score": "neg_lorentz_geodesic_distance",
        "loss": "hyperbolic_bpr_triplet",
        "negatives": "uniform_unseen",
        "optimizer": "RiemannianAdam(stabilize=1)",
        "final_loss": float(losses[-1]),
        "train_seconds": round(time.time() - started, 1),
        "device": device,
        "seed": seed,
    }
    return user_np, item_np, manifold, meta


def score_candidates(
    candidates: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_to_idx: dict,
    item_to_idx: dict,
    device: str,
) -> np.ndarray:
    n = len(candidates)
    ui = np.full(n, -1, dtype=np.int64)
    gi = np.full(n, -1, dtype=np.int64)
    for r, (uid, gid) in enumerate(candidates[["userID", "gameID"]].itertuples(index=False)):
        u = user_to_idx.get(str(uid))
        g = item_to_idx.get(str(gid))
        if u is not None:
            ui[r] = u
        if g is not None:
            gi[r] = g
    mask = (ui >= 0) & (gi >= 0)
    scores = np.full(n, -1e9, dtype=np.float32)
    if mask.any():
        k_t = torch.tensor(1.0, device=device)
        ue = torch.from_numpy(user_emb[ui[mask]]).to(device)
        ie = torch.from_numpy(item_emb[gi[mask]]).to(device)
        with torch.no_grad():
            s = lorentz_neg_dist(ue, ie, k_t).cpu().numpy().astype(np.float32)
        scores[mask] = s
    return scores


def load_e128_ensemble() -> pd.DataFrame:
    base = None
    cols = []
    for s, p in E128_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d, on="ID")
        cols.append(f"s{s}")
    base["s_emb128"] = base[cols].mean(axis=1)
    return base[["ID", "s_emb128"]]


def within_user_z(df: pd.DataFrame, col: str, user_col: str = "userID") -> pd.Series:
    g = df.groupby(user_col)[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-root", default="artifacts/validation")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out-dir", default="artifacts/hyperbolic")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_dir = Path(args.validation_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")

    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)
    n_users, n_items = len(users), len(items)
    print(f"[HypLightGCN] {args.split}: {n_users} users, {n_items} items, {mat.nnz} interactions",
          flush=True)

    user_emb, item_emb, manifold, train_meta = train(
        mat, n_users, n_items,
        emb_dim=args.emb_dim, n_layers=args.n_layers, lr=args.lr, reg=args.reg,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device, seed=args.seed,
    )

    scores = score_candidates(candidates, user_emb, item_emb, user_to_idx, item_to_idx, args.device)
    candidates = candidates.copy()
    candidates["score_lightgcn"] = scores

    solo_summary, _ = evaluate_tophalf(
        candidates, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    solo_acc = float(solo_summary["row_accuracy"])

    e128 = load_e128_ensemble()
    m = candidates[["ID", "userID", "Label", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "s_hyp"}).merge(e128, on="ID")
    m["z_hyp"] = within_user_z(m, "s_hyp")
    m["z_128"] = within_user_z(m, "s_emb128")
    corr_z = float(m["z_hyp"].corr(m["z_128"]))
    m["z_blend"] = 0.5 * m["z_hyp"] + 0.5 * m["z_128"]
    eq_blend_acc = float(evaluate_tophalf(
        m, "z_blend", label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])

    if solo_acc < FLOOR:
        tier = "REJECT_FLOOR"
        tier_reason = (f"solo_acc {solo_acc:.5f} < floor {FLOOR}: geometry broke ranking "
                       f"(SGL fate reconfirmed). Terminate.")
    elif corr_z < 0.9 and eq_blend_acc > EMB128_REF + NOISE:
        tier = "SIGNAL_ESCALATE"
        tier_reason = (f"solo_acc {solo_acc:.5f} >= floor, corr_z {corr_z:.4f} < 0.9, "
                       f"eq_blend {eq_blend_acc:.5f} > {EMB128_REF}+{NOISE}: orthogonal + blend gain "
                       f"-> 3-split panel promotion candidate (Hermes gates).")
    else:
        tier = "GEOMETRY_REDUNDANT"
        tier_reason = (f"solo_acc {solo_acc:.5f} >= floor but "
                       f"(eq_blend {eq_blend_acc:.5f} <= {EMB128_REF}+{NOISE} OR corr_z {corr_z:.4f} >= 0.9): "
                       f"geometry redundant with Euclidean base. Terminate (strong negative).")

    run_tag = f"hyp_emb{args.emb_dim}_L{args.n_layers}_reg{args.reg:g}_seed{args.seed}"
    out_dir = ensure_dir(Path(args.out_dir) / run_tag / args.split)
    candidates[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(
        out_dir / "lightgcn_scores.csv", index=False)

    payload = {
        "note": "Hyperbolic (Lorentz) LightGCN single-seed emb64 probe. Validation-only. No Kaggle submission.",
        "split": args.split,
        "config": {
            "emb_dim": args.emb_dim, "n_layers": args.n_layers, "lr": args.lr,
            "reg": args.reg, "epochs": args.epochs, "batch_size": args.batch_size,
            "seed": args.seed, "device": args.device,
        },
        "solo_acc": round(solo_acc, 5),
        "corr_z_vs_emb128": round(corr_z, 4),
        "eq_blend_acc": round(eq_blend_acc, 5),
        "floor": FLOOR,
        "emb128_ref": EMB128_REF,
        "noise": NOISE,
        "eq_blend_minus_emb128_ref": round(eq_blend_acc - EMB128_REF, 5),
        "tier": tier,
        "tier_reason": tier_reason,
        "solo_summary": solo_summary,
        "train_meta": train_meta,
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, payload)

    print("\n" + "=" * 80, flush=True)
    print(f"[HypLightGCN] solo_acc={solo_acc:.5f}  corr_z={corr_z:.4f}  "
          f"eq_blend={eq_blend_acc:.5f} (Δ vs emb128 {eq_blend_acc - EMB128_REF:+.5f})", flush=True)
    print(f"[HypLightGCN] floor={FLOOR} emb128_ref={EMB128_REF} noise={NOISE}", flush=True)
    print(f"[HypLightGCN] TIER = {tier}", flush=True)
    print(f"[HypLightGCN] {tier_reason}", flush=True)
    print("=" * 80, flush=True)
    print(f"summary.json: {summary_path}", flush=True)
    print(f"SISYPHUS_HYPERBOLIC_DONE: {summary_path} tier={tier}", flush=True)


if __name__ == "__main__":
    main()
