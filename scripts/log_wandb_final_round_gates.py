#!/usr/bin/env python3
"""Log the final-round orthogonal-axis gate verdicts to W&B (no-submit record).

This round closed the orthogonal-axis search with two negative gates:
  - MiniLM semantic text axis  (REJECT_WEAK)
  - emb128 4->8 seed expansion (TIED)
Both are meaningful progress: they document that the verifiable orthogonal-axis search is
exhausted and lock final-2. Logged as no-submit runs so the experiment record is complete.

Usage:
  env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home \
    uv run --with wandb --with pandas python scripts/log_wandb_final_round_gates.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from wandb_recsys_utils import init_run, flatten_numeric  # noqa: E402

GATES = [
    {
        "name": "gate-minilm-semantic-axis",
        "json": ROOT / "reports/20260531_semantic_minilm_uniform.json",
        "job_type": "orthogonal-axis-gate",
        "summary_keys": ["text_solo_uniform", "corr_withinuser_z", "blend50_uniform",
                         "blend50_vs_ref", "emb128_ens_ref", "tier"],
    },
    {
        "name": "gate-emb128-8seed-expansion",
        "json": ROOT / "reports/20260531_emb128_8seed_uniform_gate.json",
        "job_type": "seed-ensemble-gate",
        "summary_keys": ["ens4_uniform", "ens8_uniform", "ens8_vs_4seed_ref",
                         "noise_band", "tier"],
    },
]


def main() -> None:
    records = []
    for g in GATES:
        if not g["json"].exists():
            print(f"[skip] missing {g['json']}", flush=True)
            continue
        data = json.loads(g["json"].read_text(encoding="utf-8"))
        metrics = flatten_numeric(data)
        run = init_run(
            name=g["name"],
            job_type=g["job_type"],
            tags=["kaggle", "recsys", "steam", "no-submit", "orthogonal-axis-search",
                  "negative-result", data.get("tier", "")],
            notes=data.get("verdict", "")[:500],
            config={"report_json": str(g["json"]), "split": data.get("split"),
                    "verdict": data.get("verdict")},
        )
        if metrics:
            run.log(metrics)
        for k in g["summary_keys"]:
            if k in data:
                run.summary[k] = data[k]
        run.summary["verdict"] = data.get("verdict")
        records.append({"name": run.name, "id": run.id, "url": run.url, "tier": data.get("tier")})
        print(f"[logged] {run.name} tier={data.get('tier')} -> {run.url}", flush=True)
        run.finish()

    out = ROOT / "reports/20260531_wandb_final_round_gates_runs.json"
    out.write_text(json.dumps({"runs": records}, indent=2, ensure_ascii=False))
    print(f"\nsaved run index: {out}")
    for r in records:
        print(f"  {r['tier']:12s} {r['name']}  {r['url']}")


if __name__ == "__main__":
    main()
