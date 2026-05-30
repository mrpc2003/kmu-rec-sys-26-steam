#!/usr/bin/env python3
"""Weights & Biases helpers for KMU RecSys 26 Steam experiments.

The module intentionally imports wandb lazily so the core Kaggle scripts can run
without W&B installed.  Use through --wandb flags or the log_wandb_results.py
wrapper.  Do not print or read API keys here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

DEFAULT_WANDB_PROJECT = "kmu-rec-sys-26-steam"
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or "mrpc2003-kookmin-university"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_numeric(obj: Any, prefix: str = "") -> dict[str, float | int | bool]:
    """Flatten nested JSON-ish objects, keeping only scalar numeric/bool values."""
    out: dict[str, float | int | bool] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_numeric(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}/{i}" if prefix else str(i)
            out.update(flatten_numeric(v, key))
    elif isinstance(obj, bool):
        out[prefix] = obj
    elif isinstance(obj, int):
        out[prefix] = int(obj)
    elif isinstance(obj, float):
        if obj == obj and obj not in (float("inf"), float("-inf")):
            out[prefix] = float(obj)
    return out


def parse_tags(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x) for x in raw]


def init_run(
    *,
    project: str | None = None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    config: dict[str, Any] | None = None,
    mode: str | None = None,
):
    import wandb

    if mode:
        os.environ["WANDB_MODE"] = mode
    return wandb.init(
        project=project or os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
        entity=entity or os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags or [],
        notes=notes,
        config=config or {},
    )


def add_existing_file_to_artifact(artifact, path: Path, *, logical_name: str | None = None) -> None:
    if path.exists() and path.is_file():
        artifact.add_file(str(path), name=logical_name)


def summarize_prediction_dir(pred_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not pred_dir.exists():
        return rows
    for p in sorted(pred_dir.glob("*.csv")):
        try:
            n_rows = 0
            label_sum: int | None = None
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "Label" in reader.fieldnames:
                    label_sum = 0
                for row in reader:
                    n_rows += 1
                    if label_sum is not None:
                        label_sum += int(float(row.get("Label") or 0))
            row = {
                "file": p.name,
                "rows": int(n_rows),
                "label1": label_sum,
                "sha256": file_sha256(p),
            }
        except Exception as exc:  # pragma: no cover - diagnostics only
            row = {"file": p.name, "error": str(exc), "sha256": file_sha256(p)}
        rows.append(row)
    return rows


def log_score_dir(
    score_dir: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
    group: str | None = None,
    job_type: str = "score-eval",
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    extra_config: dict[str, Any] | None = None,
    log_artifacts: bool = True,
    artifact_mode: str = "summary",
):
    """Create a W&B run for one score/evaluation directory.

    artifact_mode:
      - "summary": upload JSON/Markdown summaries only
      - "full": also upload candidate_scores.csv and prediction CSVs
      - "none": do not upload artifacts
    """
    import wandb

    score_dir = Path(score_dir)
    eval_path = score_dir / "evaluation_summary.json"
    if not eval_path.exists() and (score_dir / "blend_evaluation_summary.json").exists():
        eval_path = score_dir / "blend_evaluation_summary.json"
    eval_summary = read_json(eval_path) if eval_path.exists() else {}
    prediction_summary = summarize_prediction_dir(score_dir / "prediction_csv")

    config: dict[str, Any] = {
        "score_dir": str(score_dir),
        "has_evaluation_summary": eval_path.exists(),
        "prediction_files": prediction_summary,
    }
    if extra_config:
        config.update(extra_config)
    if isinstance(eval_summary, dict):
        config["summary_file"] = eval_path.name if eval_path.exists() else None
        config["score_columns"] = eval_summary.get("score_columns", [])
        config["base_columns"] = eval_summary.get("base_columns", [])
        config["blend_columns"] = eval_summary.get("blend_columns", [])

    run = init_run(
        project=project,
        entity=entity,
        name=run_name or score_dir.name,
        group=group,
        job_type=job_type,
        tags=parse_tags(tags) + ["kaggle", "recsys", "steam", "no-submit"],
        notes=notes,
        config=config,
        mode=mode,
    )
    metrics: dict[str, Any] = {}
    scores = eval_summary.get("scores", []) if isinstance(eval_summary, dict) else []
    if scores:
        best = scores[0]
        metrics.update(
            {
                "best/row_accuracy": best.get("row_accuracy"),
                "best/per_user_mean_accuracy": best.get("per_user_mean_accuracy"),
                "best/predicted_positive_total": best.get("predicted_positive_total"),
                "best/true_positive_total": best.get("true_positive_total"),
                "best/score_col_index": 0,
            }
        )
        run.summary["best_score_col"] = best.get("score_col")
        table = wandb.Table(columns=["rank", "score_col", "row_accuracy", "per_user_mean_accuracy", "pred_pos", "true_pos"])
        for idx, s in enumerate(scores, 1):
            table.add_data(
                idx,
                s.get("score_col"),
                s.get("row_accuracy"),
                s.get("per_user_mean_accuracy"),
                s.get("predicted_positive_total"),
                s.get("true_positive_total"),
            )
            col = str(s.get("score_col", f"score_{idx}"))
            safe_col = col.replace("/", "_")
            metrics[f"score/{safe_col}/row_accuracy"] = s.get("row_accuracy")
            metrics[f"score/{safe_col}/per_user_mean_accuracy"] = s.get("per_user_mean_accuracy")
        wandb.log({"tables/score_summary": table})

    readme = eval_summary.get("readme_raw_baseline") if isinstance(eval_summary, dict) else None
    if isinstance(readme, dict):
        metrics.update({f"readme_raw/{k}": v for k, v in readme.items() if isinstance(v, (int, float, bool))})

    if prediction_summary:
        pred_table = wandb.Table(columns=["file", "rows", "label1", "sha256"])
        for row in prediction_summary:
            pred_table.add_data(row.get("file"), row.get("rows"), row.get("label1"), row.get("sha256"))
        wandb.log({"tables/prediction_files": pred_table})
        metrics["prediction_file_count"] = len(prediction_summary)

    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool)) and v == v}
    if metrics:
        wandb.log(metrics)

    artifact_mode = "none" if not log_artifacts else artifact_mode
    if artifact_mode not in {"none", "summary", "full"}:
        raise ValueError(f"unknown artifact_mode={artifact_mode!r}")
    if artifact_mode != "none":
        artifact = wandb.Artifact(
            name=f"{score_dir.name}-score-artifacts",
            type="recsys-score-dir",
            description="KMU RecSys 26 Steam score/evaluation artifacts; no Kaggle submission performed.",
            metadata={"score_dir": str(score_dir), "artifact_mode": artifact_mode, "sha256_files": {}},
        )
        summary_paths = [
            score_dir / "evaluation_summary.json",
            score_dir / "evaluation_summary.md",
            score_dir / "blend_evaluation_summary.json",
            score_dir / "blend_evaluation_summary.md",
        ]
        if artifact_mode == "full":
            summary_paths.append(score_dir / "candidate_scores.csv")
        for path in summary_paths:
            if path.exists():
                artifact.metadata["sha256_files"][path.name] = file_sha256(path)
                add_existing_file_to_artifact(artifact, path)
        if artifact_mode == "full":
            pred_dir = score_dir / "prediction_csv"
            if pred_dir.exists():
                for p in sorted(pred_dir.glob("*.csv")):
                    artifact.metadata["sha256_files"][f"prediction_csv/{p.name}"] = file_sha256(p)
                    add_existing_file_to_artifact(artifact, p, logical_name=f"prediction_csv/{p.name}")
        run.log_artifact(artifact)

    run.summary["score_dir"] = str(score_dir)
    run.summary["artifact_mode"] = artifact_mode
    run.summary["logged_artifacts"] = artifact_mode != "none"
    return run


def log_validation_summary(
    validation_root: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    log_artifacts: bool = True,
):
    import wandb

    validation_root = Path(validation_root)
    summary_path = validation_root / "validation_splits_summary.json"
    summaries = read_json(summary_path) if summary_path.exists() else []
    run = init_run(
        project=project,
        entity=entity,
        name=run_name or f"validation-splits-{validation_root.name}",
        job_type="validation-split-build",
        tags=parse_tags(tags) + ["kaggle", "recsys", "steam", "validation", "no-submit"],
        notes=notes,
        config={"validation_root": str(validation_root), "split_count": len(summaries)},
        mode=mode,
    )
    table = wandb.Table(
        columns=[
            "name",
            "holdout",
            "negative",
            "candidate_rows",
            "candidate_users",
            "heldout_positive_rows",
            "skipped_users",
            "overlap_with_fold_train",
            "missing_user_rows_vs_fold_train",
            "missing_item_rows_vs_fold_train",
            "negative_pop_p50",
            "actual_pair_pop_p50",
        ]
    )
    for s in summaries:
        table.add_data(
            s.get("name"),
            s.get("holdout"),
            s.get("negative"),
            s.get("candidate_rows"),
            s.get("candidate_users"),
            s.get("heldout_positive_rows"),
            s.get("skipped_users"),
            s.get("overlap_with_fold_train"),
            s.get("missing_user_rows_vs_fold_train"),
            s.get("missing_item_rows_vs_fold_train"),
            (s.get("validation_negative_item_popularity") or {}).get("p50"),
            (s.get("actual_pairs_item_popularity_in_fold_train") or {}).get("p50"),
        )
    wandb.log({"tables/validation_splits": table})
    metrics = {
        "validation/split_count": len(summaries),
        "validation/total_candidate_rows": sum(int(s.get("candidate_rows", 0)) for s in summaries),
        "validation/total_overlap_with_fold_train": sum(int(s.get("overlap_with_fold_train", 0)) for s in summaries),
        "validation/total_missing_user_rows": sum(int(s.get("missing_user_rows_vs_fold_train", 0)) for s in summaries),
        "validation/total_missing_item_rows": sum(int(s.get("missing_item_rows_vs_fold_train", 0)) for s in summaries),
    }
    wandb.log(metrics)
    if log_artifacts and summary_path.exists():
        artifact = wandb.Artifact(
            name=f"{validation_root.name}-validation-splits",
            type="recsys-validation-splits",
            description="KMU RecSys 26 Steam validation split summaries; candidate/train CSVs remain local unless separately logged.",
            metadata={"validation_root": str(validation_root), "summary_sha256": file_sha256(summary_path)},
        )
        artifact.add_file(str(summary_path), name="validation_splits_summary.json")
        for p in sorted(validation_root.glob("val_*/summary.json")):
            artifact.add_file(str(p), name=f"{p.parent.name}/summary.json")
        run.log_artifact(artifact)
    run.summary["validation_root"] = str(validation_root)
    return run
