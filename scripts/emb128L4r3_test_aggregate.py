"""Aggregate emb128_L4_reg1e-3 4-seed full-train test scores -> submittable candidate.

Averages raw test-pair scores across seeds {42,123,2024,7} (column score_lightgcn, all
produced by the canonical lightgcn_fulltest_param.py), per-user top-half decodes, and
writes a submission CSV + preflight + SHA. Mirrors lightgcn_seed_ensemble_aggregate.py
exactly (same decode, same balance checks) so the emb128 candidate is directly comparable
to the submitted emb64 ensemble (public 0.77125).

Gate context: emb128 4-seed ensemble scored uniform 0.76505 = +0.0036 over the emb64
ensemble (0.76145), > the single-seed noise band (0.0007) -> genuine upgrade on the
public surrogate. Projected public via transfer ratio ~1.26: ~0.776 (extrapolated, since
the ratio was fit on emb64; treat magnitude as indicative, direction as solid).

No Kaggle submission inside this script. Produces the candidate + preflight only; the
autonomous runner handles any Kaggle submission after safety gates.
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
from recsys_played_utils import predict_tophalf, ensure_dir

SEEDS = [42, 123, 2024, 7]
FT_DIR = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest"
EMB64_CAND = ROOT / "artifacts/lightgcn_seed_ensemble/test_candidate/candidate_lightgcn_seed_ens.csv"
OUT_DIR = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate"
OUT_JSON = ROOT / "reports/20260530_emb128L4r3_test_candidate.json"
OUT_MD = ROOT / "reports/20260530_emb128L4r3_test_candidate.md"


def main() -> None:
    test_files = {s: FT_DIR / f"seed{s}" / "test.csv" for s in SEEDS}
    missing = [s for s, p in test_files.items() if not p.exists()]
    if missing:
        print(f"[wait] emb128 full-test seeds not ready: {missing}")
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
    csv_path = out_dir / "candidate_lightgcn_emb128L4r3_seed_ens.csv"
    sub.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # preflight
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad_users = int((g.p != g.n // 2).sum())
    ids_ok = list(sub["ID"]) == list(range(len(sub)))
    labels_ok = set(sub["Label"].unique()).issubset({0, 1})

    # diff vs submitted emb64 ensemble (public 0.77125)
    row_diff = None
    if EMB64_CAND.exists():
        e64 = pd.read_csv(EMB64_CAND).rename(columns={"Label": "L64"})
        cmp = sub.rename(columns={"Label": "L128"}).merge(e64, on="ID")
        row_diff = int((cmp.L128 != cmp.L64).sum())

    summary = {
        "note": "emb128_L4_reg1e-3 4-seed ensemble test candidate. No Kaggle submission.",
        "file": str(csv_path),
        "sha256": sha,
        "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()),
        "label_0": int((1 - sub.Label).sum()),
        "bad_users": bad_users,
        "ids_contiguous": ids_ok,
        "labels_binary": labels_ok,
        "rowdiff_vs_emb64_ens": row_diff,
        "rowdiff_frac": round(row_diff / len(sub), 4) if row_diff is not None else None,
        "seeds": SEEDS,
        "uniform_gate": {"emb128_ens": 0.76505, "emb64_ens": 0.76145, "delta": 0.0036},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = [
        "# emb128_L4_reg1e-3 4-seed ensemble — test candidate\n",
        f"- file: `{csv_path}`",
        f"- sha256: `{sha}`",
        f"- rows={len(sub)} label_1={int(sub.Label.sum())} label_0={int((1-sub.Label).sum())} bad_users={bad_users}",
        f"- ids_contiguous={ids_ok} labels_binary={labels_ok}",
        (f"- rowdiff vs submitted emb64 ensemble (public 0.77125): {row_diff} "
         f"({100*row_diff/len(sub):.2f}%)" if row_diff is not None else "- emb64 candidate not found for diff"),
        "\n## Gate\n",
        "uniform (public surrogate): emb128 ensemble **0.76505** vs emb64 ensemble 0.76145 "
        "(+0.0036, > single-seed noise 0.0007). Projected public ~0.776 via transfer ratio "
        "1.26 (extrapolated from emb64; direction solid, magnitude indicative).",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[emb128 candidate] {csv_path}")
    print(f"  sha={sha}")
    print(f"  rows={len(sub)} label1={int(sub.Label.sum())} label0={int((1-sub.Label).sum())} "
          f"bad_users={bad_users} ids_ok={ids_ok} labels_ok={labels_ok}")
    if row_diff is not None:
        print(f"  rowdiff vs emb64 ens: {row_diff} ({100*row_diff/len(sub):.2f}%)")


if __name__ == "__main__":
    main()
