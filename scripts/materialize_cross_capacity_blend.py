"""Materialize the cross-capacity blend test candidate (emb128 4-seed (+) emb192 4-seed).

Parameter-free raw-mean blend (matches the uniform gate's blend_raw which scored best).
No validation-label learning. No Kaggle submission. Reports rowdiff vs the already-submitted
emb128 4-seed candidate (public 0.77745) so we know if this is a genuinely different read.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, ensure_dir  # noqa: E402

SEEDS = [42, 123, 2024, 7]


def e128_test(s): return ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv"
def e192_test(s): return ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv"


def ens_mean(path_fn, tag):
    base = pd.read_csv(path_fn(42))[["ID", "userID", "gameID", "score_lightgcn"]].rename(
        columns={"score_lightgcn": f"{tag}_42"})
    cols = [f"{tag}_42"]
    for s in SEEDS[1:]:
        d = pd.read_csv(path_fn(s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"{tag}_{s}"})
        base = base.merge(d, on="ID", how="inner"); cols.append(f"{tag}_{s}")
    base[f"{tag}_ens"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", f"{tag}_ens"]]


def main():
    a = ens_mean(e128_test, "e128")
    b = ens_mean(e192_test, "e192")
    m = a.merge(b[["ID", "e192_ens"]], on="ID", how="inner")
    m["blend"] = m[["e128_ens", "e192_ens"]].mean(axis=1)

    pred = predict_tophalf(m, "blend", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(ROOT / "artifacts/cross_capacity_blend/test_candidate")
    csv_path = out_dir / "candidate_cross_capacity_emb128_emb192_blend.csv"
    sub.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # structural validity: each user must have exactly half positives
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad_users = int((g.p != g.n // 2).sum())

    # rowdiff vs already-submitted emb128 4-seed (public 0.77745)
    ref = pd.read_csv(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv").rename(columns={"Label": "Lref"})
    cmp = sub.rename(columns={"Label": "Lnew"}).merge(ref, on="ID")
    rowdiff = int((cmp.Lnew != cmp.Lref).sum())

    out = {
        "candidate": "cross_capacity_emb128_emb192_blend (raw-mean, parameter-free)",
        "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()),
        "bad_users": bad_users, "rowdiff_vs_emb128_4seed": rowdiff,
        "rowdiff_frac": round(rowdiff / len(cmp), 4),
        "uniform_gate": "TIED (+0.0005 vs emb128 ref, within noise 0.0007)",
        "ref_emb128_4seed_public": 0.77745,
        "note": "Daily-quota public READ candidate. Different prediction vector via capacity diversity. No final-2 consumption. Candidate-only script; autonomous runner handles any Kaggle submission after safety gates.",
    }
    (ROOT / "reports/20260601_cross_capacity_blend_candidate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
