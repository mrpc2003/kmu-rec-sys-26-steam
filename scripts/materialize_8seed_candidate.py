"""Force-materialize the 8-seed emb128 test candidate (raw-mean, per-user top-half).

The 8-seed uniform gate is TIED (not UPGRADE), so the gate script does NOT auto-write
the candidate. This forces materialization anyway as a risk-free daily-quota public READ
(daily submission != final-2 consumption). Reports rowdiff vs the submitted emb128 4-seed
(public 0.77745) so we know how different the prediction actually is. No Kaggle submission.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, ensure_dir  # noqa: E402

BASE_SEEDS = [42, 123, 2024, 7]
NEW_SEEDS = [11, 99, 2025, 314]
ALL = BASE_SEEDS + NEW_SEEDS


def test_path(s): return ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv"


def main():
    miss = [s for s in ALL if not test_path(s).exists()]
    if miss:
        print(f"[abort] missing test.csv for seeds {miss}")
        return
    t = pd.read_csv(test_path(42))[["ID", "userID", "gameID", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "s42"})
    cols = ["s42"]
    for s in ALL[1:]:
        d = pd.read_csv(test_path(s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"s{s}"})
        t = t.merge(d, on="ID", how="inner"); cols.append(f"s{s}")
    t["e"] = t[cols].mean(axis=1)
    pred = predict_tophalf(t, "e", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate_8seed")
    csv_path = out_dir / "candidate_lightgcn_emb128L4r3_8seed_ens.csv"
    sub.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad = int((g.p != g.n // 2).sum())
    ref4 = pd.read_csv(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv").rename(columns={"Label": "L4"})
    cmp = sub.rename(columns={"Label": "L8"}).merge(ref4, on="ID")
    rd = int((cmp.L8 != cmp.L4).sum())

    out = {
        "candidate": "emb128_L4_reg1e-3 8-seed ensemble (raw-mean, per-user top-half)",
        "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()), "bad_users": bad,
        "rowdiff_vs_emb128_4seed": rd, "rowdiff_frac": round(rd / len(cmp), 4),
        "uniform_gate": "TIED (8seed 0.76465 vs 4seed 0.76505, Δ=-0.0004, within noise 0.0007)",
        "ref_emb128_4seed_public": 0.77745,
        "note": "Daily-quota public READ candidate. Pure variance reduction, no validation-label learning. Slightly LOWER uniform than 4-seed. No final-2 consumption. Candidate-only script; autonomous runner handles any Kaggle submission after safety gates.",
    }
    (ROOT / "reports/20260601_emb128_8seed_candidate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
