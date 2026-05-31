"""Log today's gate/postmortem JSON reports to W&B (no-submit, summary only).

Covers:
  - 8-seed expansion uniform gate (TIED)
  - cross-capacity 2-way blend uniform gate (TIED / NO_GAIN)
  - cross-capacity blend submission postmortem (public 0.77715)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from wandb_recsys_utils import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT  # noqa: E402
import wandb  # noqa: E402

REPORTS = [
    {
        "file": "reports/20260531_emb128_8seed_uniform_gate.json",
        "run_name": "emb128_8seed_expansion_gate",
        "group": "variance_reduction",
        "tags": ["no-submit", "8seed", "ensemble-expansion", "uniform-gate", "TIED"],
    },
    {
        "file": "reports/20260531_cross_capacity_blend_uniform_gate.json",
        "run_name": "cross_capacity_128_192_uniform_gate",
        "group": "capacity_blend",
        "tags": ["no-submit", "cross-capacity", "emb128", "emb192", "uniform-gate", "NO_GAIN"],
    },
    {
        "file": "reports/20260601_cross_capacity_blend_postmortem.json",
        "run_name": "cross_capacity_blend_submission_postmortem",
        "group": "capacity_blend",
        "tags": ["no-submit", "postmortem", "public-read", "cross-capacity", "TIED"],
    },
]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def main() -> None:
    for spec in REPORTS:
        path = ROOT / spec["file"]
        if not path.exists():
            print(f"[skip] {spec['file']} not found")
            continue
        data = json.loads(path.read_text())
        flat = flatten(data)
        run = wandb.init(
            project=DEFAULT_WANDB_PROJECT,
            entity=DEFAULT_WANDB_ENTITY,
            name=spec["run_name"],
            group=spec.get("group"),
            tags=spec["tags"],
            notes=f"Gate/postmortem report: {spec['file']}. No Kaggle submission.",
            config=flat,
            reinit=True,
        )
        # log numeric metrics
        metrics = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        if metrics:
            wandb.log(metrics)
        run.finish()
        print(f"[logged] {spec['run_name']} -> {run.url}")


if __name__ == "__main__":
    main()
