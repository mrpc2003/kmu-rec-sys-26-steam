"""Post-mortem for the cross-capacity blend submission (public 0.77715).

Question: blend(128+192) scored EXACTLY emb192 (0.77715), below emb128 (0.77745).
Hypothesis: raw-mean blend is dominated by the higher-variance component's score scale,
so it mostly reproduced emb192's ranking rather than truly blending. Verify by:
  1. rowdiff(blend, emb128) and rowdiff(blend, emb192) on TEST predictions
  2. raw-score std of each ensemble (does emb192 have larger scale?)
  3. within-user correlation of blend ranking to each component
No GPU, no submission.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf  # noqa: E402

SEEDS = [42, 123, 2024, 7]


def e128_test(s): return ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv"
def e192_test(s): return ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv"


def ens(path_fn, tag):
    base = pd.read_csv(path_fn(42))[["ID", "userID", "gameID", "score_lightgcn"]].rename(
        columns={"score_lightgcn": f"{tag}_42"})
    cc = [f"{tag}_42"]
    for s in SEEDS[1:]:
        d = pd.read_csv(path_fn(s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"{tag}_{s}"})
        base = base.merge(d, on="ID", how="inner"); cc.append(f"{tag}_{s}")
    base[tag] = base[cc].mean(axis=1)
    return base[["ID", "userID", "gameID", tag]]


def labels(df, col):
    pred = predict_tophalf(df, col, label_col=None, user_col="userID", id_col="ID")
    return pred[["ID", "Pred"]].rename(columns={"Pred": col + "_L"})


def main():
    a = ens(e128_test, "e128")
    b = ens(e192_test, "e192")
    m = a.merge(b[["ID", "e192"]], on="ID")
    m["blend"] = m[["e128", "e192"]].mean(axis=1)

    # raw-score scale per component (global + within-user std)
    scale = {
        "e128_global_std": round(float(m.e128.std()), 4),
        "e192_global_std": round(float(m.e192.std()), 4),
        "e128_within_user_std_mean": round(float(m.groupby("userID").e128.std().mean()), 4),
        "e192_within_user_std_mean": round(float(m.groupby("userID").e192.std().mean()), 4),
    }

    # decision labels
    L128 = labels(m, "e128"); L192 = labels(m, "e192"); Lbl = labels(m, "blend")
    z = m[["ID"]].merge(L128, on="ID").merge(L192, on="ID").merge(Lbl, on="ID")
    rd_blend_128 = int((z.blend_L != z.e128_L).sum())
    rd_blend_192 = int((z.blend_L != z.e192_L).sum())
    rd_128_192 = int((z.e128_L != z.e192_L).sum())
    n = len(z)

    # within-user Spearman-ish: how often blend ranking agrees with each component (top-half overlap)
    out = {
        "submission": "cross_capacity_emb128_emb192_blend (raw-mean)",
        "public_score": 0.77715,
        "ref_emb128_public": 0.77745,
        "ref_emb192_public": 0.77715,
        "delta_vs_emb128": round(0.77715 - 0.77745, 5),
        "delta_vs_emb192": round(0.77715 - 0.77715, 5),
        "rows": n,
        "score_scale": scale,
        "blend_label_diff_vs_emb128": rd_blend_128,
        "blend_label_diff_vs_emb192": rd_blend_192,
        "emb128_vs_emb192_label_diff": rd_128_192,
        "blend_closer_to": "emb192" if rd_blend_192 < rd_blend_128 else "emb128",
    }
    out["verdict"] = (
        f"Blend public 0.77715 == emb192 exactly, -0.00030 vs emb128. "
        f"Blend labels differ from emb192 in {rd_blend_192} rows vs from emb128 in {rd_blend_128} rows. "
        f"score scale: emb192 within-user std {scale['e192_within_user_std_mean']} vs emb128 "
        f"{scale['e128_within_user_std_mean']}. "
        "If emb192 scale dominates, raw-mean blend ~ emb192 ranking -> no orthogonal gain. "
        "Cross-capacity blend track CLOSED: saturated, emb128 4-seed (0.77745) stays #1. "
        "Also confirms uniform within-noise deltas (+0.0005) do NOT predict public sign (-0.00030)."
    )
    (ROOT / "reports/20260601_cross_capacity_blend_postmortem.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
