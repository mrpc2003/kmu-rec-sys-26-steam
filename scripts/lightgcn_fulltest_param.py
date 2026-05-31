"""Parameterized LightGCN full-train -> test-pair raw scores (for arbitrary config + seed).

Used to materialize a submittable ensemble for a sweep-selected config (e.g. emb128 L4
reg1e-3) that beat the emb64 ensemble on the uniform public-surrogate split. Uses the
CANONICAL train_lightgcn from lightgcn_train.py — the same routine that produced the
uniform validation scores — so the full-train test scores are consistent with the gated
validation result (no train/infer routine mismatch).

Saves (ID, userID, gameID, score_lightgcn) so the aggregator can average raw scores across
seeds and per-user top-half decode. No Kaggle submission.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Eager-import pandas' CSV writer so a long training run followed by a concurrent to_csv()
# across parallel workers cannot hit a lazy-import race (observed:
# ModuleNotFoundError: No module named 'pandas.io.formats.csvs' when 3 workers finish together).
import pandas  # noqa: F401
import pandas.io.formats.csvs  # noqa: F401

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (
    DEFAULT_DATA_DIR, build_user_item_matrix, load_train_json,
    load_pairs_csv, ensure_dir, write_json,
)
from lightgcn_train import train_lightgcn, score_candidates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dim", type=int, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir))
    tag = f"emb{args.emb_dim}_L{args.n_layers}_reg{args.reg:.0e}_seed{args.seed}"

    tr = load_train_json(DEFAULT_DATA_DIR / "train.json")
    pairs = load_pairs_csv(DEFAULT_DATA_DIR / "pairs.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[{tag}] FULL: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    ue, ie, meta = train_lightgcn(
        mat, len(users), len(items),
        emb_dim=args.emb_dim, n_layers=args.n_layers, lr=args.lr, reg=args.reg,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device, seed=args.seed,
    )
    pairs = pairs.copy()
    pairs["score_lightgcn"] = score_candidates(pairs, ue, ie, u2i, i2i)
    csv_path = out / "test.csv"
    pairs[["ID", "userID", "gameID", "score_lightgcn"]].to_csv(csv_path, index=False)
    write_json(out / "meta.json", {"tag": tag, "config": {
        "emb_dim": args.emb_dim, "n_layers": args.n_layers, "lr": args.lr,
        "reg": args.reg, "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "device": args.device}, "train_meta": meta})
    print(f"[{tag}] done -> {csv_path} (final_loss={meta.get('final_loss')}, {meta.get('train_seconds')}s)", flush=True)


if __name__ == "__main__":
    main()
