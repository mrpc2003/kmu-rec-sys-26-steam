"""LightGCN ⟂ Stage2 complementarity analysis on validation (CPU-only, no GPU).

Questions:
  1. Where does LightGCN err on each validation split?
  2. Does Stage2 catch LightGCN's errors? (cross-tab right/wrong)
  3. Does any blend (LightGCN + Stage2 mean-z) beat LightGCN alone on validation?

Decoding uses the canonical predict_tophalf/evaluate_tophalf (same as baseline).
Outputs JSON + Markdown to reports/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, evaluate_tophalf

SPLITS = [
    "val_random_sqrtpop_seed42",
    "val_recent_sqrtpop_seed42",
    "val_random_popbin_seed42",
]
LGCN_BASELINE = {
    "val_random_sqrtpop_seed42": 0.6748,
    "val_recent_sqrtpop_seed42": 0.6396,
    "val_random_popbin_seed42":  0.6020,
}

OUT_JSON = ROOT / "reports/20260530_lightgcn_stage2_complementarity.json"
OUT_MD = ROOT / "reports/20260530_lightgcn_stage2_complementarity.md"


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0.0


def within_user_z(df: pd.DataFrame, col: str, user_col: str = "userID") -> pd.Series:
    return df.groupby(user_col)[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0
    )


def decode_acc(df: pd.DataFrame, score_col: str) -> tuple[float, pd.DataFrame]:
    summary, pred = evaluate_tophalf(df, score_col, label_col="Label",
                                     user_col="userID", id_col="ID")
    return float(summary["row_accuracy"]), pred


def main():
    results = {}
    for split in SPLITS:
        lg = pd.read_csv(ROOT / f"artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv")
        s2 = pd.read_csv(ROOT / f"artifacts/scores/{split}_stage2_blend/merged_blend_scores.csv")

        # Merge on ID; keep LightGCN score and stage2 mean-z blend (the submitted axis)
        s2_cols = ["ID", "score_blend_mean_z", "pop_count",
                   "score_itemknn_bm25_top3", "score_ease_lambda1000"]
        m = lg.merge(s2[s2_cols], on="ID", how="inner")
        n = len(m)

        # 1) Decode each alone
        lg_acc, lg_pred = decode_acc(m, "score_lightgcn")
        s2_acc, s2_pred = decode_acc(m, "score_blend_mean_z")

        # Per-row correctness
        lg_correct = (lg_pred.sort_values("ID")["Pred"].astype(int).values ==
                      lg_pred.sort_values("ID")["Label"].astype(int).values)
        s2_correct = (s2_pred.sort_values("ID")["Pred"].astype(int).values ==
                      s2_pred.sort_values("ID")["Label"].astype(int).values)

        both_right = int((lg_correct & s2_correct).sum())
        both_wrong = int((~lg_correct & ~s2_correct).sum())
        lg_only = int((lg_correct & ~s2_correct).sum())
        s2_only = int((~lg_correct & s2_correct).sum())

        # Oracle upper bound: pick whichever is right per row
        oracle = (lg_correct | s2_correct).mean()

        # 2) Blends
        m["z_lg"] = zscore(m["score_lightgcn"])
        m["z_s2"] = zscore(m["score_blend_mean_z"])
        m["wz_lg"] = within_user_z(m, "score_lightgcn")
        m["wz_s2"] = within_user_z(m, "score_blend_mean_z")

        blend_results = {}
        for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            m["blend_global"] = w * m["z_lg"] + (1 - w) * m["z_s2"]
            acc_g, _ = decode_acc(m, "blend_global")
            m["blend_wuser"] = w * m["wz_lg"] + (1 - w) * m["wz_s2"]
            acc_w, _ = decode_acc(m, "blend_wuser")
            blend_results[f"w{w:.1f}"] = {
                "global_z": round(acc_g, 5),
                "within_user_z": round(acc_w, 5),
            }

        best_blend = max(
            [(k, v["global_z"], "global") for k, v in blend_results.items()]
            + [(k, v["within_user_z"], "wuser") for k, v in blend_results.items()],
            key=lambda x: x[1],
        )

        results[split] = {
            "rows": n,
            "lightgcn_acc": round(lg_acc, 5),
            "stage2_acc": round(s2_acc, 5),
            "lightgcn_baseline_ref": LGCN_BASELINE[split],
            "crosstab": {
                "both_right": both_right,
                "both_wrong": both_wrong,
                "lightgcn_only_right": lg_only,
                "stage2_only_right": s2_only,
                "lightgcn_only_right_frac": round(lg_only / n, 4),
                "stage2_only_right_frac": round(s2_only / n, 4),
            },
            "oracle_upper_bound": round(float(oracle), 5),
            "blends": blend_results,
            "best_blend": {"tag": best_blend[0], "acc": round(best_blend[1], 5),
                           "mode": best_blend[2],
                           "beats_lightgcn": bool(best_blend[1] > lg_acc)},
        }
        print(f"\n== {split} ==")
        print(f"  LightGCN={lg_acc:.5f}  Stage2={s2_acc:.5f}")
        print(f"  crosstab: both_right={both_right} both_wrong={both_wrong} "
              f"LG_only={lg_only} S2_only={s2_only}")
        print(f"  oracle_upper={oracle:.5f}")
        print(f"  best_blend={best_blend[0]}/{best_blend[2]}={best_blend[1]:.5f} "
              f"(beats LG: {best_blend[1] > lg_acc})")

    # Aggregate
    mean_lg = np.mean([results[s]["lightgcn_acc"] for s in SPLITS])
    mean_s2 = np.mean([results[s]["stage2_acc"] for s in SPLITS])
    mean_best_blend = np.mean([results[s]["best_blend"]["acc"] for s in SPLITS])
    mean_oracle = np.mean([results[s]["oracle_upper_bound"] for s in SPLITS])

    summary = {
        "mean_lightgcn": round(float(mean_lg), 5),
        "mean_stage2": round(float(mean_s2), 5),
        "mean_best_blend": round(float(mean_best_blend), 5),
        "mean_oracle_upper": round(float(mean_oracle), 5),
        "blend_beats_lightgcn_mean": bool(mean_best_blend > mean_lg),
        "splits": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# LightGCN ⟂ Stage2 Complementarity (validation)\n"]
    md.append(f"- mean LightGCN: **{mean_lg:.5f}**")
    md.append(f"- mean Stage2: {mean_s2:.5f}")
    md.append(f"- mean best-blend: {mean_best_blend:.5f} "
              f"({'beats' if mean_best_blend > mean_lg else 'does NOT beat'} LightGCN)")
    md.append(f"- mean oracle upper bound: {mean_oracle:.5f}\n")
    md.append("## Per-split crosstab (LightGCN vs Stage2)\n")
    md.append("| split | LG | S2 | both✓ | both✗ | LG-only✓ | S2-only✓ | oracle | best-blend |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        r = results[s]
        ct = r["crosstab"]
        bb = r["best_blend"]
        md.append(
            f"| {s.replace('val_','').replace('_seed42','')} "
            f"| {r['lightgcn_acc']:.4f} | {r['stage2_acc']:.4f} "
            f"| {ct['both_right']} | {ct['both_wrong']} "
            f"| {ct['lightgcn_only_right']} | {ct['stage2_only_right']} "
            f"| {r['oracle_upper_bound']:.4f} "
            f"| {bb['acc']:.4f} ({bb['tag']}/{bb['mode']}) |"
        )
    md.append("\n## Interpretation\n")
    s2_only_total = sum(results[s]["crosstab"]["stage2_only_right"] for s in SPLITS)
    lg_only_total = sum(results[s]["crosstab"]["lightgcn_only_right"] for s in SPLITS)
    md.append(f"- Stage2-only-right rows (LightGCN missed, Stage2 caught): {s2_only_total}")
    md.append(f"- LightGCN-only-right rows (Stage2 missed, LightGCN caught): {lg_only_total}")
    if mean_best_blend > mean_lg:
        md.append(f"- A blend improves mean validation by {mean_best_blend - mean_lg:+.5f} "
                  f"→ worth materializing a blended candidate.")
    else:
        md.append("- No blend beats LightGCN alone on mean validation "
                  "→ LightGCN single-axis remains the strongest; focus on the hparam sweep "
                  "and orthogonal new axes rather than Stage2 blending.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}\nsaved: {OUT_MD}")
    print(f"\nMEAN: LightGCN={mean_lg:.5f} best_blend={mean_best_blend:.5f} "
          f"oracle={mean_oracle:.5f}")


if __name__ == "__main__":
    main()
