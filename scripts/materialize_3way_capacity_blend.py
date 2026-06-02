"""3-way capacity blend (emb64 + emb128 + emb192, all 4-seed ens, raw-mean) + uniform gate.

Zero GPU, parameter-free, no validation-label learning, no Kaggle submission.
Reports uniform row_acc + rowdiff vs emb128 4-seed (public 0.77745) and vs the 2-way
(128+192) blend, so we know if the extra emb64 axis adds genuine prediction diversity.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, predict_tophalf, ensure_dir  # noqa: E402

SPLIT = "val_random_uniform_seed42"
SEEDS = [42, 123, 2024, 7]
EMB128_REF = 0.76505
NOISE = 0.0007


# ---- test.csv paths ----
def e64_test(s): return ROOT / ("artifacts/lightgcn_emb64L3r4_fulltest/seed%d/test.csv" % s)
def e128_test(s): return ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv"
def e192_test(s): return ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv"

# ---- uniform score paths ----
def e64_uni(s):
    base = "emb64_L3_r4" if s == 42 else f"emb64_L3_r4_seed{s}"
    return ROOT / "artifacts/capacity_uniform" / base / SPLIT / "lightgcn_scores.csv"
def e128_uni(s):
    if s == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{s}" / SPLIT / "lightgcn_scores.csv"
def e192_uni(s):
    base = "emb192_L4_r3" if s == 42 else f"emb192_L4_r3_seed{s}"
    return ROOT / "artifacts/capacity_uniform" / base / SPLIT / "lightgcn_scores.csv"


def ens(path_fn, tag, has_label):
    cols0 = ["ID", "userID", "gameID"] + (["Label"] if has_label else []) + ["score_lightgcn"]
    base = pd.read_csv(path_fn(42))[cols0].rename(columns={"score_lightgcn": f"{tag}_42"})
    cc = [f"{tag}_42"]
    for s in SEEDS[1:]:
        d = pd.read_csv(path_fn(s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"{tag}_{s}"})
        base = base.merge(d, on="ID", how="inner"); cc.append(f"{tag}_{s}")
    base[f"{tag}"] = base[cc].mean(axis=1)
    keep = ["ID", "userID", "gameID"] + (["Label"] if has_label else []) + [tag]
    return base[keep]


def acc(df, col):
    summ, _ = evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")
    return round(float(summ["row_accuracy"]), 5)


def probe_paths():
    """Verify the emb64 capacity uniform + fulltest artifacts actually exist."""
    missing = []
    for s in SEEDS:
        if not e64_uni(s).exists(): missing.append(("uni64", s, str(e64_uni(s))))
        if not e64_test(s).exists(): missing.append(("test64", s, str(e64_test(s))))
    return missing


def main():
    missing = probe_paths()
    if missing:
        print("[emb64 artifacts MISSING] cannot build 3-way blend:")
        for m in missing[:8]:
            print("  ", m)
        # fall back: report what emb64 dirs DO exist
        import glob
        print("\n[existing emb64-ish dirs]")
        for p in sorted(glob.glob(str(ROOT / "artifacts/*emb64*")) + glob.glob(str(ROOT / "artifacts/capacity_uniform/*emb64*")) + glob.glob(str(ROOT / "artifacts/lightgcn_seed_ensemble*"))):
            print("  ", p)
        return

    # ---- uniform gate ----
    u64 = ens(e64_uni, "e64", True)
    u128 = ens(e128_uni, "e128", True)
    u192 = ens(e192_uni, "e192", True)
    m = u64.merge(u128[["ID", "e128"]], on="ID").merge(u192[["ID", "e192"]], on="ID")
    m["blend3"] = m[["e64", "e128", "e192"]].mean(axis=1)
    m["blend2"] = m[["e128", "e192"]].mean(axis=1)
    a3, a2 = acc(m, "blend3"), acc(m, "blend2")
    d3 = round(a3 - EMB128_REF, 5)
    tier = "SIGNAL" if d3 > NOISE else ("WORSE" if d3 < -NOISE else "TIED")

    # ---- materialize test candidate ----
    t64 = ens(e64_test, "e64", False)
    t128 = ens(e128_test, "e128", False)
    t192 = ens(e192_test, "e192", False)
    t = t64.merge(t128[["ID", "e128"]], on="ID").merge(t192[["ID", "e192"]], on="ID")
    t["blend3"] = t[["e64", "e128", "e192"]].mean(axis=1)
    pred = predict_tophalf(t, "blend3", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")
    out_dir = ensure_dir(ROOT / "artifacts/cross_capacity_blend/test_candidate_3way")
    csv_path = out_dir / "candidate_cross_capacity_emb64_128_192_blend.csv"
    sub.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad_users = int((g.p != g.n // 2).sum())
    ref128 = pd.read_csv(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv").rename(columns={"Label": "Lref"})
    rd128 = int((sub.rename(columns={"Label": "Ln"}).merge(ref128, on="ID").eval("Ln != Lref")).sum())
    blend2 = pd.read_csv(ROOT / "artifacts/cross_capacity_blend/test_candidate/candidate_cross_capacity_emb128_emb192_blend.csv").rename(columns={"Label": "L2"})
    rd2 = int((sub.rename(columns={"Label": "Ln"}).merge(blend2, on="ID").eval("Ln != L2")).sum())

    out = {
        "candidate": "cross_capacity_emb64_128_192_blend (raw-mean, parameter-free)",
        "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()), "bad_users": bad_users,
        "uniform_blend3": a3, "uniform_blend2_128_192": a2, "ref_emb128": EMB128_REF,
        "delta3_vs_ref": d3, "tier": tier,
        "rowdiff_vs_emb128_4seed": rd128, "rowdiff_frac_vs_emb128": round(rd128 / len(sub), 4),
        "rowdiff_vs_2way_blend": rd2, "rowdiff_frac_vs_2way": round(rd2 / len(sub), 4),
        "note": "Daily-quota public READ candidate (3-capacity diversity). No final-2 consumption. Candidate-only script; autonomous runner handles any Kaggle submission after safety gates.",
    }
    (ROOT / "reports/20260601_cross_capacity_3way_candidate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
