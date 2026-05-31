#!/usr/bin/env python3
"""Paired (McNemar-style) emb192 vs emb128 4-seed ensemble analysis on the UNIFORM split.

WHY paired, not aggregate
-------------------------
emb192 ens uniform 0.76615 vs emb128 ens 0.76505 = +0.0011, only 1.6x the 0.0007 noise band.
Comparing two aggregate accuracies at that margin is high-variance. The two candidates differ
on only 3.40% of test rows, so a PAIRED analysis on the uniform validation split (same rows,
same labels) is far more decisive: restrict to rows where the two ensembles' top-half decode
DISAGREE, and ask which model is right more often there (McNemar). A clear off-diagonal skew
toward emb192 = genuine improvement; a coin-flip split = the +0.0011 is noise.

Builds BOTH 4-seed ensembles on the uniform split from per-seed raw scores, decodes per-user
top-half, and computes the 2x2 contingency + McNemar exact binomial p-value. No submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, evaluate_tophalf  # noqa: E402

SPLIT = "val_random_uniform_seed42"
SEEDS = [42, 123, 2024, 7]

EMB128_PATHS = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}
EMB192_PATHS = {
    42:   ROOT / "artifacts/capacity_uniform/emb192_L4_r3" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed7" / SPLIT / "lightgcn_scores.csv",
}
OUT_JSON = ROOT / "reports/20260531_emb192_vs_emb128_paired_uniform.json"
OUT_MD = ROOT / "reports/20260531_emb192_vs_emb128_paired_uniform.md"


def build_ensemble_pred(paths: dict[int, Path]) -> pd.DataFrame:
    base = None
    cols = []
    for s in SEEDS:
        d = pd.read_csv(paths[s])[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID", how="inner")
        cols.append(f"s{s}")
    base["ens"] = base[cols].mean(axis=1)
    pred = predict_tophalf(base, "ens", label_col="Label", user_col="userID", id_col="ID")
    # predict_tophalf returns Pred; keep Label for correctness
    out = base[["ID", "userID", "Label"]].merge(pred[["ID", "Pred"]], on="ID")
    out["correct"] = (out["Pred"] == out["Label"]).astype(int)
    return out


def main() -> None:
    a = build_ensemble_pred(EMB128_PATHS).rename(columns={"Pred": "P128", "correct": "C128"})
    b = build_ensemble_pred(EMB192_PATHS).rename(columns={"Pred": "P192", "correct": "C192"})
    m = a[["ID", "userID", "Label", "P128", "C128"]].merge(
        b[["ID", "P192", "C192"]], on="ID", how="inner")

    n = len(m)
    acc128 = round(m["C128"].mean(), 5)
    acc192 = round(m["C192"].mean(), 5)

    # disagreement rows in the per-row decode
    disagree = m[m["P128"] != m["P192"]]
    n_dis = int(len(disagree))
    # McNemar cells: among rows where they differ, who is correct
    b192_right = int(((m.C192 == 1) & (m.C128 == 0)).sum())   # emb192 right, emb128 wrong
    b128_right = int(((m.C128 == 1) & (m.C192 == 0)).sum())   # emb128 right, emb192 wrong
    both_right = int(((m.C192 == 1) & (m.C128 == 1)).sum())
    both_wrong = int(((m.C192 == 0) & (m.C128 == 0)).sum())

    # McNemar exact binomial test on the discordant pairs (b192_right vs b128_right)
    nd = b192_right + b128_right
    p_two_sided = float(stats.binomtest(b192_right, nd, 0.5).pvalue) if nd > 0 else 1.0
    net = b192_right - b128_right
    # row-level decode accuracy (note: decode-row != top-half row_accuracy metric, but paired)

    if nd == 0:
        verdict = "Ensembles are identical on the uniform decode -> emb192 adds nothing here."
        tier = "IDENTICAL"
    elif net > 0 and p_two_sided < 0.05:
        verdict = (f"emb192 wins {b192_right} vs {b128_right} on discordant rows "
                   f"(net +{net}), McNemar p={p_two_sided:.4f} < 0.05 -> SIGNIFICANT paired gain. "
                   f"The +0.0011 aggregate is a real improvement, not noise. Submission is justified.")
        tier = "SIGNIFICANT_GAIN"
    elif net > 0:
        verdict = (f"emb192 wins {b192_right} vs {b128_right} on discordant rows (net +{net}), "
                   f"but McNemar p={p_two_sided:.4f} >= 0.05 -> directionally positive yet NOT "
                   f"statistically significant. The +0.0011 edge is within paired noise; a "
                   f"submission is a low-confidence bet (real downside risk it lands <= 0.77745).")
        tier = "POSITIVE_NOT_SIGNIFICANT"
    else:
        verdict = (f"emb192 does NOT win on discordant rows ({b192_right} vs {b128_right}, "
                   f"net {net}) -> the aggregate +0.0011 is noise/decode artifact. Do not submit; "
                   f"keep emb128 (public 0.77745).")
        tier = "NO_PAIRED_GAIN"

    summary = {
        "note": "Paired emb192 vs emb128 4-seed ensemble on uniform split. No submission.",
        "split": SPLIT, "rows": n,
        "decode_row_acc_emb128": acc128, "decode_row_acc_emb192": acc192,
        "disagree_rows": n_dis, "disagree_frac": round(n_dis / n, 4),
        "mcnemar": {"emb192_right_emb128_wrong": b192_right,
                    "emb128_right_emb192_wrong": b128_right,
                    "both_right": both_right, "both_wrong": both_wrong,
                    "discordant_pairs": nd, "net_emb192_minus_emb128": net,
                    "p_two_sided": round(p_two_sided, 5)},
        "tier": tier, "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# emb192 vs emb128 — PAIRED (McNemar) on uniform split\n",
          f"- split `{SPLIT}` rows={n} | decode-row acc: emb128 {acc128}, emb192 {acc192}",
          f"- disagreement rows: {n_dis} ({100*n_dis/n:.2f}%)\n",
          "## McNemar 2x2 (per-row decode correctness)\n",
          "| | emb128 right | emb128 wrong |", "|---|---:|---:|",
          f"| **emb192 right** | {both_right} | {b192_right} |",
          f"| **emb192 wrong** | {b128_right} | {both_wrong} |",
          f"\n- discordant pairs: {nd} | net (emb192−emb128): **{net:+d}** | McNemar two-sided p = **{p_two_sided:.4f}**",
          f"- **tier: {tier}**\n", f"{verdict}\n",
          "## Why paired\n",
          "The +0.0011 aggregate uniform gain is 1.6x the 0.0007 noise band — borderline. Since "
          "the two candidates differ on only ~3.4% of rows, a paired McNemar test on the same "
          "rows de-noises the comparison far better than comparing two aggregate accuracies, and "
          "directly informs whether a submission is justified."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[PAIRED] emb128_acc={acc128} emb192_acc={acc192} | discordant={nd} "
          f"emb192_right={b192_right} emb128_right={b128_right} net={net:+d} "
          f"p={p_two_sided:.4f} tier={tier}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
