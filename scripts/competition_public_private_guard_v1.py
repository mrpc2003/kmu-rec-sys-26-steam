#!/usr/bin/env python3
"""Competition public/private guard v1 (NO-SUBMIT).

This CLI converts already-submitted public-failed candidates into a reusable
negative-control guard. It writes reports only, never writes under submissions/,
and never creates or scores a new full-test candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
CURRENT_BEST_PATH = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
CALIBRATION_PATH = ROOT / "reports/boundary_public_failure_calibration.csv"

NOISE_BAND = 0.0007
HIGH_FAILED_OVERLAP_FRAC = 0.75
MODERATE_FAILED_OVERLAP_FRAC = 0.50
BOUNDARY_LE3_RISK_FRAC = 0.95


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    assert_allowed_report_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def assert_allowed_report_path(path: Path) -> None:
    resolved = path.resolve()
    submissions = (ROOT / "submissions").resolve()
    if resolved == submissions or submissions in resolved.parents:
        raise ValueError(f"Refusing to write under submissions/: {path}")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    x = float(text)
    return x if math.isfinite(x) else None


def safe_int(value: Any) -> int:
    x = safe_float(value)
    return 0 if x is None else int(round(x))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def label_name(row: dict[str, str], path: Path) -> str:
    for col in ("Played", "Label"):
        if col in row:
            return col
    raise ValueError(f"No Played/Label column in {path}")


def read_labels(path: Path) -> dict[int, int]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    label = label_name(rows[0], path)
    out: dict[int, int] = {}
    for row in rows:
        out[int(row["ID"])] = int(row[label])
    return out


def parse_counts(raw: Any) -> dict[str, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    obj = json.loads(text)
    return {str(k): int(v) for k, v in obj.items()}


def build_changed_sets(calibration: list[dict[str, str]]) -> dict[str, set[int]]:
    current = read_labels(CURRENT_BEST_PATH)
    changed_sets: dict[str, set[int]] = {}
    missing: list[str] = []
    for row in calibration:
        cid = str(row["candidate_id"])
        path = ROOT / str(row["file"])
        if not path.exists():
            missing.append(cid)
            continue
        candidate = read_labels(path)
        common = current.keys() & candidate.keys()
        changed_sets[cid] = {row_id for row_id in common if current[row_id] != candidate[row_id]}
    if missing:
        raise SystemExit(f"Missing historical candidate files: {missing}")
    return changed_sets


def create_overlap_matrix(calibration: list[dict[str, str]], changed_sets: dict[str, set[int]]) -> list[dict[str, Any]]:
    family_by_id = {row["candidate_id"]: row["family"] for row in calibration}
    rows: list[dict[str, Any]] = []
    ids = sorted(changed_sets)
    for a in ids:
        for b in ids:
            set_a = changed_sets[a]
            set_b = changed_sets[b]
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            rows.append(
                {
                    "candidate_a": a,
                    "family_a": family_by_id.get(a),
                    "candidate_b": b,
                    "family_b": family_by_id.get(b),
                    "row_diff_a": len(set_a),
                    "row_diff_b": len(set_b),
                    "overlap_count": inter,
                    "jaccard": None if union == 0 else inter / union,
                    "frac_of_a": None if not set_a else inter / len(set_a),
                    "frac_of_b": None if not set_b else inter / len(set_b),
                    "same_family": family_by_id.get(a) == family_by_id.get(b),
                }
            )
    return rows


def median_public_loss_by_family(calibration: list[dict[str, str]]) -> dict[str, float]:
    losses: dict[str, list[float]] = {}
    for row in calibration:
        public_delta = safe_float(row.get("public_delta_vs_current_best"))
        if public_delta is None:
            continue
        losses.setdefault(row["family"], []).append(max(0.0, -public_delta))
    return {family: median(values) for family, values in losses.items()}


def other_union(candidate_id: str, changed_sets: dict[str, set[int]]) -> set[int]:
    out: set[int] = set()
    for cid, ids in changed_sets.items():
        if cid != candidate_id:
            out |= ids
    return out


def score_negative_controls(
    calibration: list[dict[str, str]], changed_sets: dict[str, set[int]]
) -> list[dict[str, Any]]:
    family_losses = median_public_loss_by_family(calibration)
    scored: list[dict[str, Any]] = []
    for row in calibration:
        cid = str(row["candidate_id"])
        family = str(row["family"])
        changed = changed_sets[cid]
        others = other_union(cid, changed_sets)
        overlap_other = len(changed & others)
        overlap_frac_other = None if not changed else overlap_other / len(changed)
        public_delta = safe_float(row.get("public_delta_vs_current_best"))
        validation_delta = safe_float(row.get("validation_delta"))
        boundary_le3 = safe_float(row.get("changed_pct_boundary_le3")) or 0.0
        family_public_loss = family_losses.get(family, 0.0)
        overlap_penalty = 0.0
        if overlap_frac_other is not None and overlap_frac_other >= HIGH_FAILED_OVERLAP_FRAC:
            overlap_penalty = NOISE_BAND
        elif overlap_frac_other is not None and overlap_frac_other >= MODERATE_FAILED_OVERLAP_FRAC:
            overlap_penalty = NOISE_BAND / 2.0
        boundary_penalty = NOISE_BAND if boundary_le3 >= BOUNDARY_LE3_RISK_FRAC else 0.0
        guard_delta = None
        if validation_delta is not None:
            guard_delta = validation_delta - family_public_loss - overlap_penalty - boundary_penalty
        mismatch = bool(validation_delta is not None and validation_delta > 0 and public_delta is not None and public_delta < 0)
        if mismatch and guard_delta is not None and guard_delta <= 0:
            verdict = "REJECT_NEGATIVE_CONTROL"
        elif overlap_penalty > 0 or boundary_penalty > 0:
            verdict = "FLAG_HIGH_FALSE_POSITIVE_RISK"
        else:
            verdict = "OBSERVE"
        scored.append(
            {
                "candidate_id": cid,
                "family": family,
                "row_diff_vs_current_best_fulltest": len(changed),
                "public_delta_vs_current_best": public_delta,
                "validation_delta": validation_delta,
                "normal_validation_positive_public_negative": mismatch,
                "family_public_loss_penalty": family_public_loss,
                "overlap_with_other_failed_rows": overlap_other,
                "overlap_frac_with_other_failed_rows": overlap_frac_other,
                "overlap_penalty": overlap_penalty,
                "changed_pct_boundary_le3": boundary_le3,
                "boundary_penalty": boundary_penalty,
                "guard_adjusted_validation_delta": guard_delta,
                "guard_verdict": verdict,
                "source_note": row.get("source_note"),
            }
        )
    return scored


def boundary_bucket_counts(row: dict[str, str]) -> dict[str, int]:
    total = safe_int(row.get("row_diff_vs_current_best_fulltest"))
    le1 = int(round(total * (safe_float(row.get("changed_pct_boundary_le1")) or 0.0)))
    le3 = int(round(total * (safe_float(row.get("changed_pct_boundary_le3")) or 0.0)))
    le5 = int(round(total * (safe_float(row.get("changed_pct_boundary_le5")) or 0.0)))
    le1 = max(0, min(total, le1))
    le3 = max(0, min(total, le3))
    le5 = max(0, min(total, le5))
    return {
        "boundary_le1": le1,
        "boundary_1to3": max(0, le3 - le1),
        "boundary_3to5": max(0, le5 - le3),
        "boundary_gt5": max(0, total - le5),
    }


def weighted_mean(values: list[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return None if denominator == 0 else numerator / denominator


def create_bucket_audit(calibration: list[dict[str, str]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_by_id = {row["candidate_id"]: row for row in scored}
    expanded: list[dict[str, Any]] = []
    dimensions = {
        "candidate_count_bucket": "changed_candidate_count_bucket_counts",
        "user_degree_decile": "changed_user_degree_decile_counts",
        "item_degree_decile": "changed_item_degree_decile_counts",
    }
    for row in calibration:
        cid = row["candidate_id"]
        score = score_by_id[cid]
        public_delta = safe_float(row.get("public_delta_vs_current_best"))
        validation_delta = safe_float(row.get("validation_delta"))
        mismatch = bool(score["normal_validation_positive_public_negative"])
        for dimension, col in dimensions.items():
            for bucket, count in parse_counts(row.get(col)).items():
                expanded.append(
                    {
                        "dimension": dimension,
                        "bucket": bucket,
                        "candidate_id": cid,
                        "family": row.get("family"),
                        "changed_rows": count,
                        "public_delta": public_delta,
                        "validation_delta": validation_delta,
                        "mismatch": mismatch,
                        "overlap_frac": score.get("overlap_frac_with_other_failed_rows"),
                    }
                )
        for bucket, count in boundary_bucket_counts(row).items():
            expanded.append(
                {
                    "dimension": "boundary_distance_bucket",
                    "bucket": bucket,
                    "candidate_id": cid,
                    "family": row.get("family"),
                    "changed_rows": count,
                    "public_delta": public_delta,
                    "validation_delta": validation_delta,
                    "mismatch": mismatch,
                    "overlap_frac": score.get("overlap_frac_with_other_failed_rows"),
                }
            )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in expanded:
        groups.setdefault((str(row["dimension"]), str(row["bucket"])), []).append(row)
    bucket_rows: list[dict[str, Any]] = []
    for (dimension, bucket), rows in groups.items():
        total = sum(int(row["changed_rows"]) for row in rows)
        mismatch_rows = sum(int(row["changed_rows"]) for row in rows if row["mismatch"])
        public_negative_rows = sum(
            int(row["changed_rows"]) for row in rows if (row["public_delta"] is not None and row["public_delta"] < 0)
        )
        families = sorted({str(row["family"]) for row in rows if row.get("family")})
        note = "false_positive_bucket" if total > 0 and mismatch_rows / total >= 0.5 else "observe"
        bucket_rows.append(
            {
                "dimension": dimension,
                "bucket": bucket,
                "total_changed_rows_across_controls": total,
                "candidate_count": len({str(row["candidate_id"]) for row in rows}),
                "families": ",".join(families),
                "mismatch_changed_rows": mismatch_rows,
                "mismatch_row_frac": None if total == 0 else mismatch_rows / total,
                "public_negative_changed_rows": public_negative_rows,
                "public_negative_row_frac": None if total == 0 else public_negative_rows / total,
                "weighted_mean_public_delta": weighted_mean(
                    [(row["public_delta"], int(row["changed_rows"])) for row in rows]
                ),
                "weighted_mean_validation_delta": weighted_mean(
                    [(row["validation_delta"], int(row["changed_rows"])) for row in rows]
                ),
                "weighted_mean_overlap_frac_with_other_failed": weighted_mean(
                    [(row["overlap_frac"], int(row["changed_rows"])) for row in rows]
                ),
                "guard_note": note,
            }
        )
    return sorted(
        bucket_rows,
        key=lambda row: (
            0 if row["guard_note"] == "false_positive_bucket" else 1,
            -(row["mismatch_row_frac"] or 0.0),
            -int(row["total_changed_rows_across_controls"]),
        ),
    )


def summarize(scored: list[dict[str, Any]], overlap: list[dict[str, Any]]) -> dict[str, Any]:
    valid_pairs = [
        row
        for row in scored
        if row["validation_delta"] is not None and row["public_delta_vs_current_best"] is not None
    ]
    mismatches = [row for row in valid_pairs if row["normal_validation_positive_public_negative"]]
    rejected = [row for row in mismatches if (row["guard_adjusted_validation_delta"] or 0.0) <= 0]
    separated = sorted({row["family"] for row in rejected})
    required = {"boundary_scoreblend", "frontier_z_boundary", "tagcf_boundary", "als_residual_rankblend"}
    offdiag = [row for row in overlap if row["candidate_a"] != row["candidate_b"]]
    xs = [float(row["validation_delta"]) for row in valid_pairs]
    ys = [float(row["public_delta_vs_current_best"]) for row in valid_pairs]
    verdict = "PASS_NEGATIVE_CONTROLS_SEPARATED"
    if not required.issubset(set(separated)) or len(rejected) != len(mismatches) or len(mismatches) < 4:
        verdict = "FAIL_GUARD_NOT_CALIBRATED"
    return {
        "artifact": "competition_public_private_guard_v1",
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_candidate_materialized": False,
        "public_lb_feedback_used": True,
        "current_best_public": 0.77825,
        "seed42_base_accuracy": 0.7650530106021204,
        "negative_controls": len(scored),
        "valid_validation_public_pairs": len(valid_pairs),
        "normal_validation_positive_public_negative_controls": len(mismatches),
        "guard_rejected_mismatch_controls": len(rejected),
        "validation_public_delta_correlation": pearson(xs, ys),
        "required_false_positive_families": sorted(required),
        "separated_false_positive_families": sorted(set(separated) & required),
        "max_offdiag_overlap_jaccard": max((row["jaccard"] or 0.0) for row in offdiag) if offdiag else None,
        "max_offdiag_overlap_frac_of_a": max((row["frac_of_a"] or 0.0) for row in offdiag) if offdiag else None,
        "guard_verdict": verdict,
        "outputs": {
            "md": "reports/competition_public_private_guard_v1.md",
            "json": "reports/competition_public_private_guard_v1.json",
            "overlap_matrix_csv": "reports/competition_failed_family_overlap_matrix.csv",
            "bucket_audit_csv": "reports/competition_false_positive_bucket_audit.csv",
        },
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_md(path: Path, summary: dict[str, Any], scored: list[dict[str, Any]], bucket: list[dict[str, Any]]) -> None:
    lines = [
        "# competition public/private guard v1",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- full_test_candidate_materialized: false",
        "- public_lb_feedback_used: true, only as negative-control calibration from already submitted historical candidates",
        "",
        "## verdict",
        "",
        f"`{summary['guard_verdict']}`",
        "",
        "This is a guard harness, not a model. It makes the competition-writeup lesson operational: do not trust small normal-validation deltas when they resemble already public-failed row families.",
        "",
        "## negative-control separation",
        "",
        f"- controls checked: {summary['negative_controls']}",
        f"- validation/public pairs with normal validation positive but public negative: {summary['normal_validation_positive_public_negative_controls']}",
        f"- those rejected by guard-adjusted delta: {summary['guard_rejected_mismatch_controls']}",
        f"- separated families: {', '.join(summary['separated_false_positive_families'])}",
        f"- validation/public delta correlation on usable controls: {fmt(summary['validation_public_delta_correlation'])}",
        "",
        "| candidate | family | val delta | public delta | diff rows | other-failed overlap | boundary≤3 | guard delta | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(scored, key=lambda r: (str(r["guard_verdict"]), str(r["family"]), str(r["candidate_id"]))):
        lines.append(
            f"| `{row['candidate_id']}` | {row['family']} | {fmt(row['validation_delta'])} | "
            f"{fmt(row['public_delta_vs_current_best'])} | {int(row['row_diff_vs_current_best_fulltest'])} | "
            f"{fmt(row['overlap_frac_with_other_failed_rows'], 3)} | {fmt(row['changed_pct_boundary_le3'], 3)} | "
            f"{fmt(row['guard_adjusted_validation_delta'])} | `{row['guard_verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## bucket-level false-positive audit",
            "",
            "Top buckets below are historical danger zones. They are descriptive guard features, not public-LB tuning thresholds.",
            "",
            "| dimension | bucket | changed rows | mismatch frac | public-negative frac | mean public delta | mean validation delta | note |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in bucket[:16]:
        lines.append(
            f"| {row['dimension']} | {row['bucket']} | {int(row['total_changed_rows_across_controls'])} | "
            f"{fmt(row['mismatch_row_frac'], 3)} | {fmt(row['public_negative_row_frac'], 3)} | "
            f"{fmt(row['weighted_mean_public_delta'])} | {fmt(row['weighted_mean_validation_delta'])} | {row['guard_note']} |"
        )
    lines.extend(
        [
            "",
            "## how to use for the next smoke",
            "",
            "1. Keep the normal seed42 smoke gate: solo, fixed z-blend, fixes/breaks, corr_z, and no head-only lift.",
            "2. Add this guard report before any escalation. A small positive validation delta is not enough if the row/bucket profile matches `false_positive_bucket` rows.",
            "3. Do not create a full-test candidate only to measure overlap. Full-test overlap is allowed here only because these are already-existing historical negative controls.",
            "4. If a legacy candidate file already exists, compare its changed-row mask to `reports/competition_failed_family_overlap_matrix.csv`; high overlap with failed families is a reject/flag, not a retune prompt.",
            "5. For validation-only new smoke, compare changed-row bucket profile against `reports/competition_false_positive_bucket_audit.csv` and require guard-adjusted evidence above the existing `+0.0007` noise band.",
            "",
            "## outputs",
            "",
            "- `reports/competition_public_private_guard_v1.md`",
            "- `reports/competition_public_private_guard_v1.json`",
            "- `reports/competition_failed_family_overlap_matrix.csv`",
            "- `reports/competition_false_positive_bucket_audit.csv`",
            "",
            "COMPETITION_PUBLIC_PRIVATE_GUARD_V1_DONE",
        ]
    )
    assert_allowed_report_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPORTS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    md_path = out_dir / "competition_public_private_guard_v1.md"
    json_path = out_dir / "competition_public_private_guard_v1.json"
    overlap_path = out_dir / "competition_failed_family_overlap_matrix.csv"
    bucket_path = out_dir / "competition_false_positive_bucket_audit.csv"
    for path in (md_path, json_path, overlap_path, bucket_path):
        assert_allowed_report_path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = read_csv_rows(CALIBRATION_PATH)
    changed_sets = build_changed_sets(calibration)
    overlap = create_overlap_matrix(calibration, changed_sets)
    scored = score_negative_controls(calibration, changed_sets)
    bucket = create_bucket_audit(calibration, scored)
    summary = summarize(scored, overlap)
    payload = {
        **summary,
        "negative_control_rows": scored,
        "constants": {
            "noise_band": NOISE_BAND,
            "high_failed_overlap_frac": HIGH_FAILED_OVERLAP_FRAC,
            "moderate_failed_overlap_frac": MODERATE_FAILED_OVERLAP_FRAC,
            "boundary_le3_risk_frac": BOUNDARY_LE3_RISK_FRAC,
        },
    }

    write_csv_rows(
        overlap_path,
        overlap,
        [
            "candidate_a",
            "family_a",
            "candidate_b",
            "family_b",
            "row_diff_a",
            "row_diff_b",
            "overlap_count",
            "jaccard",
            "frac_of_a",
            "frac_of_b",
            "same_family",
        ],
    )
    write_csv_rows(
        bucket_path,
        bucket,
        [
            "dimension",
            "bucket",
            "total_changed_rows_across_controls",
            "candidate_count",
            "families",
            "mismatch_changed_rows",
            "mismatch_row_frac",
            "public_negative_changed_rows",
            "public_negative_row_frac",
            "weighted_mean_public_delta",
            "weighted_mean_validation_delta",
            "weighted_mean_overlap_frac_with_other_failed",
            "guard_note",
        ],
    )
    json_path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md_path, summary, scored, bucket)
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
