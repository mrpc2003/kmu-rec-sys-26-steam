#!/usr/bin/env python3
"""LightGCN capacity-frontier probe — UNIFORM gate (parameterized emb/layers/reg, single seed).

WHY this is the most defensible "continue" (2026-05-31)
-------------------------------------------------------
Every closed bet was a WEAK ORTHOGONAL side-axis. The backbone itself is the ONLY signal
source proven to move the real public LB: emb64 4-seed ensemble uniform 0.76145 -> emb128
0.76505 (+0.0036), which transferred to public 0.77125 -> 0.77745. The capacity frontier
ABOVE emb128 (emb192/256/320) was swept ONLY on the hard samplers (sqrtpop/recent/popbin) in
the original hparam sweep -- NEVER on the uniform split, which we later proved is the public
surrogate. So those emb256 numbers are an INVALID gate (wrong distribution). This probe closes
that gap: it gates capacity on the uniform split, the distribution that actually decides the LB.

Honest prior: LOW-MODERATE. corr(z64,z128)=0.9747 signals BPR-LightGCN saturation, and a small
dense graph (165k interactions) risks overfitting at emb256+. But emb64->128 was a clean
monotone uniform gain, so the curve's next point is worth one real measurement. Gate: a single
emb>=192 seed must clearly beat the emb128 single-seed uniform (~0.762) to justify a seed
ensemble; matching/below means capacity has plateaued and the frontier is closed.

Canonical train_lightgcn (same code path as every gated LightGCN result). Validation-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix, load_pairs_csv, load_train_interactions,
    evaluate_tophalf, ensure_dir, write_json,
)
from lightgcn_train import train_lightgcn, score_candidates  # noqa: E402

SPLIT = "val_random_uniform_seed42"
EMB128_SINGLE_SEED_REF = 0.76205   # emb128 L4 reg1e-3 seed42 uniform (from 8-seed gate per-seed table)
EMB128_ENS_REF = 0.76505           # emb128 4-seed ensemble uniform (public 0.77745)
NOISE = 0.0007


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, required=True)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"emb{args.emb_dim}_L{args.n_layers}_reg{args.reg:.0e}_seed{args.seed}"

    sp_dir = ROOT / "artifacts/validation" / SPLIT
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] {SPLIT}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    ue, ie, meta = train_lightgcn(
        mat, len(users), len(items), emb_dim=args.emb_dim, n_layers=args.n_layers,
        lr=args.lr, reg=args.reg, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )
    cand = cand.copy()
    cand["score_lightgcn"] = score_candidates(cand, ue, ie, u2i, i2i)
    summ, _ = evaluate_tophalf(cand, "score_lightgcn", label_col="Label",
                               user_col="userID", id_col="ID")
    acc = round(float(summ["row_accuracy"]), 5)

    vs_single = round(acc - EMB128_SINGLE_SEED_REF, 5)
    vs_ens = round(acc - EMB128_ENS_REF, 5)
    if vs_single > NOISE:
        tier = "CAPACITY_GAIN_CHECK_ENSEMBLE"
        verdict = (f"single {tag} {acc} beats emb128 single-seed {EMB128_SINGLE_SEED_REF} by "
                   f"{vs_single:+.5f} > noise -> capacity frontier still rising; build seed ensemble.")
    elif vs_single >= -NOISE:
        tier = "CAPACITY_PLATEAU"
        verdict = (f"single {tag} {acc} ~ emb128 single-seed {EMB128_SINGLE_SEED_REF} "
                   f"({vs_single:+.5f}) -> capacity plateaued; no clear gain from more dims.")
    else:
        tier = "CAPACITY_REGRESS_OVERFIT"
        verdict = (f"single {tag} {acc} < emb128 single-seed {EMB128_SINGLE_SEED_REF} by "
                   f"{vs_single:+.5f} -> higher capacity overfits this small dense graph.")

    outd = ensure_dir(out / SPLIT)
    cand[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(
        outd / "lightgcn_scores.csv", index=False)
    write_json(outd / "summary.json", {"tag": tag, "row_accuracy": acc,
               "emb128_single_seed_ref": EMB128_SINGLE_SEED_REF, "emb128_ens_ref": EMB128_ENS_REF,
               "vs_single_seed": vs_single, "vs_ens": vs_ens, "tier": tier,
               "verdict": verdict, "meta": meta})
    print(f"[{tag}] uniform row_acc={acc} | vs emb128 single {vs_single:+.5f} | "
          f"vs ens {vs_ens:+.5f} | tier={tier} ({meta['train_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
