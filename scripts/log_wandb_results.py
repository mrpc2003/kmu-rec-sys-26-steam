#!/usr/bin/env python3
"""Log existing KMU RecSys Steam validation/scoring outputs to W&B.

Example:
  env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home \
    uv run --with wandb --with pandas python scripts/log_wandb_results.py \
      --validation-root artifacts/validation \
      --score-dir artifacts/scores/val_random_sqrtpop_seed42_proto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wandb_recsys_utils import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    log_score_dir,
    log_validation_summary,
    parse_tags,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    ap.add_argument("--mode", default=None, choices=["online", "offline", "disabled"], help="Optional WANDB_MODE override")
    ap.add_argument("--score-dir", action="append", default=[], help="Score/evaluation directory to log; may repeat")
    ap.add_argument("--validation-root", default=None, help="Validation root containing validation_splits_summary.json")
    ap.add_argument("--run-name-prefix", default="", help="Prefix for generated run names")
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", default="manual-log")
    ap.add_argument("--notes", default="KMU RecSys 26 Steam experiment tracking log. No Kaggle submission performed.")
    ap.add_argument("--no-artifacts", action="store_true", help="Shortcut for --artifact-mode none")
    ap.add_argument("--artifact-mode", default="summary", choices=["none", "summary", "full"], help="W&B artifact upload level: summary JSON/MD only by default; full also uploads large CSVs")
    args = ap.parse_args()

    tags = parse_tags(args.tags)
    run_records = []

    if args.validation_root:
        run_name = f"{args.run_name_prefix}validation-splits" if args.run_name_prefix else None
        run = log_validation_summary(
            args.validation_root,
            project=args.project,
            entity=args.entity,
            run_name=run_name,
            tags=tags,
            notes=args.notes,
            mode=args.mode,
            log_artifacts=args.artifact_mode != "none" and not args.no_artifacts,
        )
        run_records.append({"kind": "validation", "name": run.name, "id": run.id, "url": run.url})
        run.finish()

    for score_dir in args.score_dir:
        p = Path(score_dir)
        run_name = f"{args.run_name_prefix}{p.name}" if args.run_name_prefix else p.name
        run = log_score_dir(
            p,
            project=args.project,
            entity=args.entity,
            run_name=run_name,
            group=args.group,
            tags=tags,
            notes=args.notes,
            mode=args.mode,
            extra_config={"logged_by": "scripts/log_wandb_results.py"},
            log_artifacts=args.artifact_mode != "none" and not args.no_artifacts,
            artifact_mode=args.artifact_mode,
        )
        run_records.append({"kind": "score_dir", "score_dir": str(p), "name": run.name, "id": run.id, "url": run.url})
        run.finish()

    print(json.dumps({"project": args.project, "entity": args.entity, "runs": run_records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
