#!/usr/bin/env python3
"""Collect XSimGCL lambda-sweep uniform results -> single verdict table.

Reads each config's summary.json (row_accuracy on the uniform public-surrogate split) and
compares against the emb128 4-seed ensemble reference 0.76505. Adoption bar is the same
parameter-free gate used for every other axis: a config must beat the ref by > noise 0.0007
to be a real upgrade; if even the best config is below the popularity floor 0.684, the whole
contrastive-CF axis is closed as REJECTED (consistent with SGL/DirectAU).

No Kaggle submission. Report-only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_ENS_REF = 0.76505
POP_FLOOR = 0.684
NOISE = 0.0007
CONFIGS = [
    ("lam0.02_eps0.1", "artifacts/xsimgcl_uniform/lam0.02_eps0.1"),
    ("lam0.05_eps0.1", "artifacts/xsimgcl_uniform/lam0.05_eps0.1"),
    ("lam0.1_eps0.1",  "artifacts/xsimgcl_uniform/lam0.1_eps0.1"),
    ("lam0.2_eps0.2",  "artifacts/xsimgcl_uniform/lam0.2_eps0.2"),
]
SPLIT = "val_random_uniform_seed42"
OUT_JSON = ROOT / "reports/20260531_xsimgcl_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_xsimgcl_uniform_gate.md"


def main() -> None:
    rows, missing = [], []
    for label, d in CONFIGS:
        p = ROOT / d / SPLIT / "summary.json"
        if not p.exists():
            missing.append(label); continue
        s = json.loads(p.read_text())
        rows.append((label, round(float(s["row_accuracy"]), 5),
                     s.get("meta", {}).get("train_seconds")))
    if missing:
        print(f"[wait] not ready: {missing}", flush=True)
    if not rows:
        print("no results yet"); return

    rows.sort(key=lambda r: r[1], reverse=True)
    best_label, best_acc, _ = rows[0]
    vs_ref = round(best_acc - EMB128_ENS_REF, 5)
    if best_acc < POP_FLOOR:
        tier = "REJECT_AXIS_CLOSED"
        verdict = (f"best XSimGCL ({best_label}={best_acc}) < popularity floor {POP_FLOOR}. "
                   f"Same failure as SGL/DirectAU -> contrastive-CF axis CLOSED with evidence: "
                   f"InfoNCE-uniformity (not edge-dropout) is what breaks fine per-user ranking "
                   f"on this small balanced-reranking dataset.")
    elif vs_ref > NOISE:
        tier = "UPGRADE_CANDIDATE"
        verdict = (f"XSimGCL ({best_label}={best_acc}) beats emb128 ref {EMB128_ENS_REF} by "
                   f"{vs_ref:+.5f} > noise {NOISE} -> materialize full-test candidate, gate to user.")
    else:
        tier = "TIED_OR_WEAK"
        verdict = (f"best XSimGCL ({best_label}={best_acc}) >= floor but vs emb128 ref = "
                   f"{vs_ref:+.5f}, within/under noise {NOISE} -> not a parameter-free upgrade; "
                   f"contrastive term gives no clear gain over the strong backbone.")

    summary = {
        "note": "XSimGCL lambda-sweep uniform gate (emb128 L4 reg1e-3 backbone). No submission.",
        "split": SPLIT, "emb128_ens_ref": EMB128_ENS_REF, "pop_floor": POP_FLOOR,
        "noise": NOISE, "results": [{"config": l, "uniform_acc": a, "train_s": t} for l, a, t in rows],
        "best": {"config": best_label, "uniform_acc": best_acc, "vs_ref": vs_ref},
        "missing": missing, "tier": tier, "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# XSimGCL λ-sweep — UNIFORM gate (public surrogate)\n",
          f"- backbone: emb128 L4 reg1e-3 + XSimGCL CL term | split `{SPLIT}`",
          f"- emb128 4-seed ref (public 0.77745): **{EMB128_ENS_REF}** | pop floor {POP_FLOOR} | noise ±{NOISE}",
          f"- **best: {best_label} = {best_acc}  (vs ref {vs_ref:+.5f})**",
          f"- **tier: {tier}** — {verdict}\n",
          "| config | uniform acc | train s |", "|---|---:|---:|"]
    for l, a, t in rows:
        md.append(f"| {l} | {a} | {t} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[XSIMGCL GATE] best={best_label} acc={best_acc} ref={EMB128_ENS_REF} "
          f"vs_ref={vs_ref:+.5f} tier={tier}", flush=True)
    for l, a, t in rows:
        print(f"  {l}: {a} ({t}s)", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
