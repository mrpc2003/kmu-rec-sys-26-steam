"""Compare sweep-best LightGCN configs on the UNIFORM split (public surrogate).

Decision context
----------------
The sweep ranked configs by hard-sampler mean (sqrtpop/recent/popbin). But the OOD gate
proved public (0.76245) tracks the UNIFORM split, and the mechanism test showed reg/arch
changes that help on hard samplers can hurt on uniform. So a sweep config is only a real
upgrade if it beats the anchor ON UNIFORM — and to displace the current best candidate it
must also clear the seed ensemble's uniform score.

References (uniform split, seed42, emb64 L3 reg1e-4 verified path):
  - anchor (submitted single LightGCN)          : 0.75445
  - 4-seed ensemble (current ROBUST_GAIN cand.) : 0.76145

For each sweep-best config we trained seed42 on the uniform split via lightgcn_train.py
(report json holds results[0].summary.row_accuracy).

No Kaggle submission. Report-only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
ANCHOR_UNIFORM = 0.75445
ENSEMBLE_UNIFORM = 0.76145

# config tag -> sweep hard-sampler mean Δ (for context in the table)
CONFIGS = {
    "emb128_L4_reg1e-03": +0.00326,
    "emb128_L4_reg1e-04": +0.00310,
    "emb128_L3_reg1e-03": +0.00223,
}
OUT_JSON = ROOT / "reports/20260530_sweep_best_uniform_gate.json"
OUT_MD = ROOT / "reports/20260530_sweep_best_uniform_gate.md"


def load_uniform_acc(tag: str):
    p = ROOT / f"reports/20260530_sweepuni_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return round(float(d["results"][0]["summary"]["row_accuracy"]), 5)


def main() -> None:
    rows = []
    missing = []
    for tag, hard_delta in CONFIGS.items():
        acc = load_uniform_acc(tag)
        if acc is None:
            missing.append(tag)
            continue
        rows.append({
            "tag": tag,
            "sweep_hard_mean_delta": hard_delta,
            "uniform_acc": acc,
            "vs_anchor": round(acc - ANCHOR_UNIFORM, 5),
            "vs_ensemble": round(acc - ENSEMBLE_UNIFORM, 5),
            "beats_anchor": acc > ANCHOR_UNIFORM + 0.0005,
            "beats_ensemble": acc > ENSEMBLE_UNIFORM + 0.0005,
        })

    rows.sort(key=lambda r: -r["uniform_acc"])
    summary = {
        "note": "Sweep-best configs gated on uniform (public surrogate). No Kaggle submission.",
        "anchor_uniform": ANCHOR_UNIFORM,
        "ensemble_uniform": ENSEMBLE_UNIFORM,
        "configs": rows,
        "missing": missing,
    }
    if rows:
        best = rows[0]
        if best["beats_ensemble"]:
            verdict = (f"{best['tag']} BEATS the seed ensemble on uniform "
                       f"({best['uniform_acc']} > {ENSEMBLE_UNIFORM}) — new strongest candidate.")
        elif best["beats_anchor"]:
            verdict = (f"{best['tag']} beats the anchor on uniform but NOT the seed ensemble "
                       f"({best['uniform_acc']} vs ens {ENSEMBLE_UNIFORM}) — keep ensemble as the candidate.")
        else:
            verdict = (f"No sweep config beats the anchor on uniform (best {best['tag']}="
                       f"{best['uniform_acc']} vs anchor {ANCHOR_UNIFORM}); the hard-sampler gain "
                       f"did NOT transfer to the public surrogate — keep seed ensemble.")
        summary["verdict"] = verdict
    else:
        summary["verdict"] = "no configs ready"

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# Sweep-best configs — UNIFORM gate (public surrogate)\n",
          f"- anchor uniform: **{ANCHOR_UNIFORM}**  |  seed ensemble uniform: **{ENSEMBLE_UNIFORM}**",
          (f"- missing (still training): {missing}" if missing else "- all configs evaluated"),
          f"- **verdict: {summary['verdict']}**\n",
          "| config | sweep hard Δmean | uniform acc | vs anchor | vs ensemble |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['sweep_hard_mean_delta']:+.5f} | {r['uniform_acc']} "
                  f"| {r['vs_anchor']:+.5f} | {r['vs_ensemble']:+.5f} |")
    md.append("\n## Interpretation\n"
              "A config that gained on the hard-sampler mean but does not beat the anchor on "
              "uniform is an artifact of the wrong surrogate (see mechanism test: reg/arch "
              "changes that suppress popularity help on hard samplers, hurt on uniform). Only a "
              "config that beats the seed ensemble on uniform would displace the current candidate.")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(summary["verdict"])
    for r in rows:
        print(f"  {r['tag']}: uniform={r['uniform_acc']} vs_anchor={r['vs_anchor']:+.5f} "
              f"vs_ens={r['vs_ensemble']:+.5f}")
    if missing:
        print(f"  [waiting] {missing}")


if __name__ == "__main__":
    main()
