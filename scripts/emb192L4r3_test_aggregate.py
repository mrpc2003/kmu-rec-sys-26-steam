"""Aggregate emb192_L4_reg1e-3 4-seed full-train test scores -> submittable candidate.

Mirror of emb128L4r3_test_aggregate.py (same canonical lightgcn_fulltest_param.py source
column score_lightgcn, same per-user top-half decode, same preflight + SHA), for the emb192
backbone that beat emb128 on the uniform public surrogate. Averages raw test-pair scores
across seeds {42,123,2024,7} and writes a submission CSV + preflight + SHA.

ONLY run this after the emb192 4-seed uniform gate confirms UPGRADE (emb192 ensemble beats
emb128 ensemble 0.76505 by > noise 0.0007). No Kaggle submission -- candidate + preflight
only; submission needs 우현's explicit one-file approval.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, ensure_dir  # noqa: E402

SEEDS = [42, 123, 2024, 7]
FT_DIR = ROOT / "artifacts/lightgcn_emb192L4r3_fulltest"
EMB128_CAND = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv"
OUT_DIR = ROOT / "artifacts/lightgcn_emb192L4r3_fulltest/test_candidate"
OUT_JSON = ROOT / "reports/20260531_emb192L4r3_test_candidate.json"
OUT_MD = ROOT / "reports/20260531_emb192L4r3_test_candidate.md"


def main() -> None:
    test_files = {s: FT_DIR / f"seed{s}" / "test.csv" for s in SEEDS}
    missing = [s for s, p in test_files.items() if not p.exists()]
    if missing:
        print(f"[wait] emb192 full-test seeds not ready: {missing}")
        return

    mt = None
    cols = []
    for s in SEEDS:
        d = pd.read_csv(test_files[s])[["ID", "userID", "gameID", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"score_seed{s}"})
        mt = d if mt is None else mt.merge(d[["ID", f"score_seed{s}"]], on="ID", how="inner")
        cols.append(f"score_seed{s}")

    mt["score_ens"] = mt[cols].mean(axis=1)
    pred = predict_tophalf(mt, "score_ens", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(OUT_DIR)
    csv_path = out_dir / "candidate_lightgcn_emb192L4r3_seed_ens.csv"
    sub.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad_users = int((g.p != g.n // 2).sum())
    ids_ok = list(sub["ID"]) == list(range(len(sub)))
    labels_ok = set(sub["Label"].unique()).issubset({0, 1})

    row_diff = None
    if EMB128_CAND.exists():
        e128 = pd.read_csv(EMB128_CAND).rename(columns={"Label": "L128"})
        cmp = sub.rename(columns={"Label": "L192"}).merge(e128, on="ID")
        row_diff = int((cmp.L192 != cmp.L128).sum())

    summary = {
        "note": "emb192_L4_reg1e-3 4-seed ensemble test candidate. No Kaggle submission.",
        "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()),
        "bad_users": bad_users, "ids_contiguous": ids_ok, "labels_binary": labels_ok,
        "rowdiff_vs_emb128_ens": row_diff,
        "rowdiff_frac": round(row_diff / len(sub), 4) if row_diff is not None else None,
        "seeds": SEEDS,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = [
        "# emb192_L4_reg1e-3 4-seed ensemble — test candidate\n",
        f"- file: `{csv_path}`",
        f"- sha256: `{sha}`",
        f"- rows={len(sub)} label_1={int(sub.Label.sum())} label_0={int((1-sub.Label).sum())} bad_users={bad_users}",
        f"- ids_contiguous={ids_ok} labels_binary={labels_ok}",
        (f"- rowdiff vs emb128 ensemble (public 0.77745): {row_diff} "
         f"({100*row_diff/len(sub):.2f}%)" if row_diff is not None else "- emb128 candidate not found for diff"),
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[emb192 candidate] {csv_path}")
    print(f"  sha={sha}")
    print(f"  rows={len(sub)} label1={int(sub.Label.sum())} label0={int((1-sub.Label).sum())} "
          f"bad_users={bad_users} ids_ok={ids_ok} labels_ok={labels_ok}")
    if row_diff is not None:
        print(f"  rowdiff vs emb128 ens: {row_diff} ({100*row_diff/len(sub):.2f}%)")


if __name__ == "__main__":
    main()
