#!/usr/bin/env python3
"""Cross-capacity blend (emb128 + emb192) — UNIFORM gate + paired McNemar vs emb128.

POST-ANALYSIS LEVER (2026-05-31)
--------------------------------
Submission post-mortem: emb128 4-seed (public 0.77745) and emb192 4-seed (public 0.77715) are
near-tied on the real LB but DISAGREE on 3.40% of test rows -> two strong models with
different capacity making different errors. Every closed axis was either (a) strong backbone +
WEAK side-axis, or (b) same-config seed averaging (8-seed, saturated). A cross-CAPACITY blend
of two STRONG models is a genuinely untested diversity axis: corr(z128,z192) was never
measured, and 3.4% disagreement implies corr < 1.

Method (parameter-free, same bar as every axis):
  z128 = within-user z(mean of 4 emb128 raw scores)
  z192 = within-user z(mean of 4 emb192 raw scores)
  blend = 0.5*z128 + 0.5*z192   (50/50, parameter-free -- no grid-tuned weight = no stacker-trap)
Also report the straight 8-raw-mean for reference.

Honest prior given the post-mortem: surrogate reliable resolution is ~Δuniform 0.003; a blend
gain below that is within the unreliable zone. BUT diversity-driven variance reduction across
two genuinely different strong models is more robust than a single-config tweak, so this is
worth a paired-gated measurement. Decision rule: only escalate to a candidate if the blend
beats emb128 4-seed 0.76505 AND paired McNemar p < 0.05 (the emb192 lesson: ignore p>=0.05).

Validation-only. No Kaggle submission.
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
from recsys_played_utils import evaluate_tophalf, predict_tophalf  # noqa: E402

SPLIT = "val_random_uniform_seed42"
SEEDS = [42, 123, 2024, 7]
EMB128_ENS_REF = 0.76505
NOISE = 0.0007

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
OUT_JSON = ROOT / "reports/20260531_cross_capacity_blend_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_cross_capacity_blend_uniform_gate.md"


def zwithin(df, col, user="userID"):
    g = df.groupby(user)[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)).to_numpy()


def ens_mean(paths):
    base = None; cols = []
    for s in SEEDS:
        d = pd.read_csv(paths[s])[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID", how="inner")
        cols.append(f"s{s}")
    base["ens"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "ens"]]


def acc_of(df, col):
    s, _ = evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")
    return round(float(s["row_accuracy"]), 5)


def main() -> None:
    a = ens_mean(EMB128_PATHS).rename(columns={"ens": "e128"})
    b = ens_mean(EMB192_PATHS).rename(columns={"ens": "e192"})
    m = a.merge(b[["ID", "e192"]], on="ID", how="inner")

    acc128 = acc_of(m, "e128")
    acc192 = acc_of(m, "e192")

    m["z128"] = zwithin(m, "e128")
    m["z192"] = zwithin(m, "e192")
    corr_z = float(np.corrcoef(m["z128"], m["z192"])[0, 1])

    m["blend_z"] = 0.5 * m["z128"] + 0.5 * m["z192"]
    acc_blend_z = acc_of(m, "blend_z")
    # straight 8-raw-mean reference (scale-biased but reported)
    m["blend_raw8"] = 0.5 * m["e128"] + 0.5 * m["e192"]
    acc_blend_raw = acc_of(m, "blend_raw8")

    vs_ref = round(acc_blend_z - EMB128_ENS_REF, 5)

    # paired McNemar: blend_z decode vs emb128 decode, on uniform labels
    pa = predict_tophalf(m, "e128", label_col="Label", user_col="userID", id_col="ID")[["ID", "Pred"]].rename(columns={"Pred": "P128"})
    pb = predict_tophalf(m, "blend_z", label_col="Label", user_col="userID", id_col="ID")[["ID", "Pred"]].rename(columns={"Pred": "PBL"})
    mm = m[["ID", "Label"]].merge(pa, on="ID").merge(pb, on="ID")
    mm["C128"] = (mm.P128 == mm.Label).astype(int)
    mm["CBL"] = (mm.PBL == mm.Label).astype(int)
    bl_right = int(((mm.CBL == 1) & (mm.C128 == 0)).sum())
    e128_right = int(((mm.C128 == 1) & (mm.CBL == 0)).sum())
    nd = bl_right + e128_right
    p_two = float(stats.binomtest(bl_right, nd, 0.5).pvalue) if nd > 0 else 1.0
    disagree = int((mm.P128 != mm.PBL).sum())

    if vs_ref > NOISE and p_two < 0.05 and bl_right > e128_right:
        tier = "ESCALATE_MATERIALIZE"
        verdict = (f"blend_z {acc_blend_z} beats emb128 ens {EMB128_ENS_REF} by {vs_ref:+.5f} "
                   f"AND paired McNemar significant (blend {bl_right} vs emb128 {e128_right}, "
                   f"p={p_two:.4f}) -> genuine diversity gain. Materialize 8-model candidate, gate to 우현.")
    elif vs_ref > NOISE:
        tier = "POSITIVE_NOT_SIGNIFICANT"
        verdict = (f"blend_z {acc_blend_z} > ref by {vs_ref:+.5f} but paired McNemar NOT significant "
                   f"(blend {bl_right} vs emb128 {e128_right}, p={p_two:.4f}). Same trap as emb192 "
                   f"(+0.0011 uniform -> -0.0003 public). Surrogate resolution ~0.003; do NOT submit "
                   f"on this alone. corr_z(128,192)={corr_z:.3f}.")
    else:
        tier = "NO_GAIN"
        verdict = (f"blend_z {acc_blend_z} vs ref {EMB128_ENS_REF} = {vs_ref:+.5f} (<= noise). "
                   f"corr_z(128,192)={corr_z:.3f}; cross-capacity blend gives no uniform gain. "
                   f"Keep emb128 4-seed (public 0.77745).")

    summary = {
        "note": "Cross-capacity emb128+emb192 blend uniform gate + paired McNemar. No submission.",
        "split": SPLIT, "rows": int(len(m)),
        "emb128_ens_uniform": acc128, "emb192_ens_uniform": acc192,
        "corr_withinuser_z_128_192": round(corr_z, 4),
        "blend_z_uniform": acc_blend_z, "blend_raw8_uniform": acc_blend_raw,
        "emb128_ens_ref": EMB128_ENS_REF, "blend_vs_ref": vs_ref, "noise": NOISE,
        "mcnemar": {"blend_right_emb128_wrong": bl_right, "emb128_right_blend_wrong": e128_right,
                    "discordant": nd, "p_two_sided": round(p_two, 5), "decode_disagree_rows": disagree},
        "tier": tier, "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# Cross-capacity blend emb128⊕emb192 — UNIFORM gate + paired McNemar\n",
          f"- split `{SPLIT}` rows={len(m)}",
          f"- emb128 ens {acc128} | emb192 ens {acc192} | **corr_z(128,192) = {corr_z:.4f}**",
          f"- 50/50 z-blend uniform: **{acc_blend_z}** (raw-8-mean {acc_blend_raw}) vs emb128 ref {EMB128_ENS_REF} → **{vs_ref:+.5f}**",
          f"- paired McNemar: blend right/emb128 wrong = {bl_right}, emb128 right/blend wrong = {e128_right}, p = **{p_two:.4f}**",
          f"- **tier: {tier}** — {verdict}\n",
          "## Why this axis\n",
          "Post-mortem: emb128 (0.77745) and emb192 (0.77715) tie on real LB but disagree on 3.40% "
          "of rows -> two strong, partly-decorrelated models. A cross-capacity blend is the one "
          "diversity play not yet tested (vs weak side-axes and same-config seed averaging). Gated "
          "with paired McNemar because the emb192 submission proved surrogate Δ < 0.003 is unreliable."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[CROSS-CAP] emb128={acc128} emb192={acc192} corr_z={corr_z:.4f} "
          f"blend_z={acc_blend_z} (vs ref {vs_ref:+.5f}) | McNemar blend={bl_right} "
          f"emb128={e128_right} p={p_two:.4f} | tier={tier}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
