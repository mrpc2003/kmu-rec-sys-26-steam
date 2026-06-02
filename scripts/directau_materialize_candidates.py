"""Materialize DirectAU candidates IF the uniform gate says it's a new axis.

Two products, both PARAMETER-FREE (no validation-label tuning => no stacker trap):
  A) DirectAU 4-seed ensemble (raw cosine score mean) -> standalone candidate
  B) Equal-weight (50/50) within-user z-blend of [emb128 ensemble] + [DirectAU ensemble]
     -> the diversity play that could crack the 21.4% "neither correct" uniform rows

Only run this AFTER directau_gate.py returns STRONG_SOLO or NEW_AXIS for the chosen gamma.
Requires DirectAU full-train test scores at artifacts/directau_fulltest/g{GAMMA}/seed{S}/test.csv
(produce with: lightgcn_directau.py --mode test --gamma G --seed S --out-dir ...).

Emits candidate CSVs + SHA + preflight. NO Kaggle submission inside this script; the
autonomous runner handles any Kaggle submission after safety gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, ensure_dir

SEEDS = [42, 123, 2024, 7]
EMB128_TEST = {s: ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv" for s in SEEDS}
EMB128_CAND = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv"


def load_test(p, col="score_lightgcn"):
    return pd.read_csv(p)[["ID", "userID", "gameID", col]].rename(columns={col: "s"})


def mean_ens(files):
    base = None
    cols = []
    for s, p in files.items():
        d = load_test(p).rename(columns={"s": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["ens"] = base[cols].mean(axis=1)
    return base


def wz(df, c):
    g = df.groupby("userID")[c]
    return (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)


def preflight(sub, label_col="Label"):
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=(label_col, "sum"))
    bad = int((g.p != g.n // 2).sum())
    ids_ok = list(sub["ID"]) == list(range(len(sub)))
    return {"rows": int(len(sub)), "label_1": int(sub[label_col].sum()),
            "label_0": int((1 - sub[label_col]).sum()), "bad_users": bad,
            "ids_contiguous": ids_ok}


def write_candidate(sub, path, label_col="Label"):
    out = sub[["ID", label_col]].rename(columns={label_col: "Label"}).sort_values("ID")
    ensure_dir(path.parent)
    out.to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha, preflight(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma", required=True, help="winning gamma, e.g. 1.0")
    args = ap.parse_args()

    dau_files = {s: ROOT / f"artifacts/directau_fulltest/g{args.gamma}/seed{s}/test.csv" for s in SEEDS}
    missing = [s for s, p in dau_files.items() if not p.exists()] + \
              [s for s, p in EMB128_TEST.items() if not p.exists()]
    if missing:
        print(f"[wait] missing test scores for seeds: {sorted(set(missing))}")
        return

    dau = mean_ens(dau_files)[["ID", "userID", "gameID", "ens"]].rename(columns={"ens": "sdau"})
    e128 = mean_ens(EMB128_TEST)[["ID", "ens"]].rename(columns={"ens": "s128"})
    m = dau.merge(e128, on="ID")

    out_dir = ensure_dir(ROOT / f"artifacts/directau_fulltest/candidates_g{args.gamma}")
    summary = {"gamma": args.gamma, "products": {}}

    # A) DirectAU-only ensemble candidate
    predA = predict_tophalf(m.rename(columns={"sdau": "s"}), "s", label_col=None, user_col="userID", id_col="ID")
    subA = predA[["ID", "Pred"]].rename(columns={"Pred": "Label"})
    pa = out_dir / "candidate_directau_seed_ens.csv"
    shaA, pfA = write_candidate(subA, pa)
    summary["products"]["directau_only"] = {"file": str(pa), "sha256": shaA, **pfA}

    # B) parameter-free 50/50 within-user z-blend of emb128 + DirectAU
    m["z128"] = wz(m, "s128")
    m["zdau"] = wz(m, "sdau")
    m["zblend"] = 0.5 * m["z128"] + 0.5 * m["zdau"]
    predB = predict_tophalf(m.rename(columns={"zblend": "s"}), "s", label_col=None, user_col="userID", id_col="ID")
    subB = predB[["ID", "Pred"]].rename(columns={"Pred": "Label"})
    pb = out_dir / "candidate_emb128_directau_zblend5050.csv"
    shaB, pfB = write_candidate(subB, pb)
    summary["products"]["emb128_directau_zblend5050"] = {"file": str(pb), "sha256": shaB, **pfB}

    # diffs vs the submitted emb128 ensemble (public 0.77745)
    if EMB128_CAND.exists():
        e = pd.read_csv(EMB128_CAND).rename(columns={"Label": "L128"})
        for name, sub in [("directau_only", subA), ("emb128_directau_zblend5050", subB)]:
            cmp = sub.rename(columns={"Label": "Lx"}).merge(e, on="ID")
            rd = int((cmp.Lx != cmp.L128).sum())
            summary["products"][name]["rowdiff_vs_emb128_public"] = rd
            summary["products"][name]["rowdiff_frac"] = round(rd / len(cmp), 4)

    (ROOT / f"reports/20260531_directau_candidates_g{args.gamma}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
