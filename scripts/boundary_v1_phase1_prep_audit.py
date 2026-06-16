#!/usr/bin/env python3
"""Boundary v1 phase-1 preparation audit (NO-SUBMIT).

This script consumes the phase-0 calibration artifacts and creates the prep reports
needed before a scored boundary specialist cross-fit run.  It never submits to Kaggle,
never creates a candidate CSV, and does not train a model.

Outputs:
- reports/boundary_v1_row_filter_audit_19998_vs_19996.md
- reports/boundary_v1_row_filter_audit_19998_vs_19996.json
- reports/boundary_v1_public_failed_overlap_matrix.csv
- reports/boundary_v1_public_failed_changed_row_summary.csv
- reports/boundary_v1_diffband_precision_curve_transfer_adjusted.csv
- reports/boundary_v1_phase1_gate_policy.md
- reports/boundary_v1_scored_split20_crossfit_eval.md  (readiness note, not an eval result)
- reports/boundary_v1_calibration_ledger.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "data/raw/public/data/pairs.csv"
PANEL20_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
PANEL20_SUMMARY = PANEL20_ROOT / "validation_splits_summary.json"
CURRENT_BEST_PATH = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
PHASE0_CALIBRATION = ROOT / "reports/boundary_public_failure_calibration.csv"
PHASE0_CURVE = ROOT / "reports/boundary_v1_diffband_precision_curve.csv"

PUBLIC_DENOM_EST = 9999
TRANSFER_PENALTY_LOW = 0.02
TRANSFER_PENALTY_HIGH = 0.05


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return None if not np.isfinite(x) else x
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def label_col(df: pd.DataFrame) -> str:
    for col in ("Played", "Label"):
        if col in df.columns:
            return col
    raise ValueError(f"No Played/Label column in {df.columns.tolist()}")


def read_labels(path: Path, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = label_col(df)
    return df[["ID", col]].rename(columns={col: out_col}).assign(**{out_col: lambda d: d[out_col].astype(int)})


def bucket_candidate_count(x: int) -> str:
    if x <= 2:
        return "02"
    if x <= 4:
        return "03-04"
    if x <= 6:
        return "05-06"
    if x <= 10:
        return "07-10"
    return "11+"


def audit_row_filter(out_md: Path, out_json: Path) -> dict[str, Any]:
    pairs = pd.read_csv(PAIRS_PATH)
    if list(pairs.columns) != ["ID", "userID", "gameID"]:
        raise ValueError(f"Unexpected pairs columns: {pairs.columns.tolist()}")
    splits = json.loads(PANEL20_SUMMARY.read_text(encoding="utf-8"))
    raw_user_counts = pairs.groupby("userID").size().astype(int)
    raw_bad_even = raw_user_counts[raw_user_counts % 2 != 0]

    split_rows: list[dict[str, Any]] = []
    dropped_user_union: dict[str, dict[str, Any]] = {}
    for split in splits:
        name = split["name"]
        cand_path = ROOT / split["out_dir"] / "candidates.csv"
        if not cand_path.exists():
            raise FileNotFoundError(cand_path)
        cand = pd.read_csv(cand_path)
        cand_user_counts = cand.groupby("userID").size().astype(int)
        cand_pos_counts = cand.groupby("userID")["Label"].sum().astype(int)
        dropped_users = sorted(set(raw_user_counts.index) - set(cand_user_counts.index))
        extra_users = sorted(set(cand_user_counts.index) - set(raw_user_counts.index))
        dropped_rows = pairs[pairs["userID"].isin(dropped_users)].copy()
        for uid in dropped_users:
            user_pairs = pairs[pairs["userID"] == uid]
            dropped_user_union[uid] = {
                "userID": uid,
                "raw_pair_rows": int(len(user_pairs)),
                "raw_pair_ids": user_pairs["ID"].astype(int).tolist(),
                "raw_pair_games": user_pairs["gameID"].astype(str).tolist(),
            }
        bad_even_after = cand_user_counts[cand_user_counts % 2 != 0]
        bad_half = cand_pos_counts[cand_pos_counts * 2 != cand_user_counts.loc[cand_pos_counts.index]]
        split_rows.append(
            {
                "split": name,
                "candidate_rows": int(len(cand)),
                "candidate_users": int(cand["userID"].nunique()),
                "heldout_positive_rows": int(cand["Label"].sum()),
                "expected_positive_rows_from_half": int(len(cand) // 2),
                "dropped_user_count_vs_pairs": len(dropped_users),
                "dropped_users": dropped_users,
                "dropped_pair_row_count": int(len(dropped_rows)),
                "dropped_pair_ids": dropped_rows["ID"].astype(int).tolist(),
                "extra_user_count_vs_pairs": len(extra_users),
                "all_candidate_counts_even_after_filter": len(bad_even_after) == 0,
                "all_positive_counts_half_after_filter": len(bad_half) == 0,
                "summary_adjusted_users": int(split.get("adjusted_users", -1)),
                "summary_skipped_users": int(split.get("skipped_users", -1)),
                "summary_adjusted_user_examples": split.get("adjusted_user_examples", []),
            }
        )

    unique_dropped_users = sorted(dropped_user_union.values(), key=lambda x: x["userID"])
    payload = {
        "artifact": "boundary_v1_row_filter_audit_19998_vs_19996",
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "public_lb_feedback_used": False,
        "raw_pairs": {
            "path": str(PAIRS_PATH.relative_to(ROOT)),
            "rows": int(len(pairs)),
            "users": int(pairs["userID"].nunique()),
            "items": int(pairs["gameID"].nunique()),
            "all_candidate_counts_even": len(raw_bad_even) == 0,
            "odd_candidate_count_users": raw_bad_even.index.astype(str).tolist(),
            "per_user_count_summary": {
                "min": int(raw_user_counts.min()),
                "median": float(raw_user_counts.median()),
                "max": int(raw_user_counts.max()),
                "mean": float(raw_user_counts.mean()),
            },
        },
        "panel20": {
            "path": str(PANEL20_SUMMARY.relative_to(ROOT)),
            "split_count": len(splits),
            "candidate_rows_unique": sorted({r["candidate_rows"] for r in split_rows}),
            "candidate_users_unique": sorted({r["candidate_users"] for r in split_rows}),
            "heldout_positive_rows_unique": sorted({r["heldout_positive_rows"] for r in split_rows}),
            "all_splits_even_after_filter": all(r["all_candidate_counts_even_after_filter"] for r in split_rows),
            "all_splits_positive_half_after_filter": all(r["all_positive_counts_half_after_filter"] for r in split_rows),
            "unique_dropped_users_vs_pairs": unique_dropped_users,
            "split_rows": split_rows,
        },
        "interpretation": {
            "mismatch_status": "explained_by_one_user_filtered_from_all_panel20_splits",
            "reason_from_split_summary": "The split builder keeps at least one fold-train interaction per user. User u57101927 has train_n=1, requested_k=1, actual_k=0, so that user is skipped.",
            "effect": "raw pairs 19998 rows / 4737 users -> panel20 19996 rows / 4736 users; 2 raw pair rows from the skipped user are absent.",
            "phase1_risk": "Boundary scored evaluation must either keep this same panel filter or explicitly document a different policy before comparing diff-band precision.",
        },
    }
    write_json(out_json, payload)

    lines = [
        "# boundary v1 row-filter audit — 19,998 vs 19,996",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- public_lb_feedback_used: false",
        "",
        "## 판정",
        "",
        "19,998 rows / 4,737 users와 19,996 rows / 4,736 users 차이는 artifact mismatch가 아니라, panel20 split builder의 의도된 사용자 필터링으로 설명된다.",
        "문제 사용자는 `u57101927` 한 명이고, raw `pairs.csv`에는 2개 row가 있다. 해당 사용자는 train interaction이 1개라 holdout을 만들면 fold-train에 남길 interaction이 없어져 panel에서 제외된다.",
        "",
        "## 숫자 확인",
        "",
        "| 항목 | rows | users | 비고 |",
        "|---|---:|---:|---|",
        f"| raw pairs.csv | {payload['raw_pairs']['rows']} | {payload['raw_pairs']['users']} | all candidate counts even = {payload['raw_pairs']['all_candidate_counts_even']} |",
        f"| panel20 candidates | {payload['panel20']['candidate_rows_unique']} | {payload['panel20']['candidate_users_unique']} | heldout positives = {payload['panel20']['heldout_positive_rows_unique']} |",
        "",
        "## 제외된 사용자",
        "",
        "| userID | raw pair rows | raw pair IDs | raw games |",
        "|---|---:|---|---|",
    ]
    for row in unique_dropped_users:
        lines.append(
            f"| `{row['userID']}` | {row['raw_pair_rows']} | {row['raw_pair_ids']} | {row['raw_pair_games']} |"
        )
    lines.extend(
        [
            "",
            "## phase-1 주의점",
            "",
            "scored boundary evaluation에서는 이 panel20 필터를 그대로 쓰거나, 다른 필터를 쓰는 경우 denominator와 user boundary를 별도로 기록해야 한다.",
            "diff-band precision curve를 public rows로 환산할 때도 panel 기준 denominator와 raw pairs 기준 denominator를 섞지 않는다.",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def create_overlap_matrix(out_matrix: Path, out_summary: Path) -> dict[str, Any]:
    cal = pd.read_csv(PHASE0_CALIBRATION)
    current = read_labels(CURRENT_BEST_PATH, "current")
    changed_sets: dict[str, set[int]] = {}
    summary_rows: list[dict[str, Any]] = []
    for _, row in cal.iterrows():
        path = ROOT / str(row["file"])
        if not bool(row.get("exists", True)) or not path.exists():
            continue
        cand = read_labels(path, "candidate")
        merged = current.merge(cand, on="ID", validate="one_to_one")
        ids = set(merged.loc[merged["current"] != merged["candidate"], "ID"].astype(int).tolist())
        cid = str(row["candidate_id"])
        changed_sets[cid] = ids
        summary_rows.append(
            {
                "candidate_id": cid,
                "family": row.get("family"),
                "row_diff_vs_current_best_fulltest": len(ids),
                "public_score": row.get("public_score"),
                "public_delta_vs_current_best": row.get("public_delta_vs_current_best"),
                "validation_flip_precision": row.get("validation_flip_precision"),
            }
        )

    matrix_rows: list[dict[str, Any]] = []
    ids_sorted = sorted(changed_sets)
    for a in ids_sorted:
        for b in ids_sorted:
            set_a = changed_sets[a]
            set_b = changed_sets[b]
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            matrix_rows.append(
                {
                    "candidate_a": a,
                    "candidate_b": b,
                    "row_diff_a": len(set_a),
                    "row_diff_b": len(set_b),
                    "overlap_count": inter,
                    "jaccard": None if union == 0 else inter / union,
                    "frac_of_a": None if len(set_a) == 0 else inter / len(set_a),
                    "frac_of_b": None if len(set_b) == 0 else inter / len(set_b),
                }
            )
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean(matrix_rows)).to_csv(out_matrix, index=False)
    pd.DataFrame(clean(summary_rows)).to_csv(out_summary, index=False)
    return {
        "candidate_count": len(ids_sorted),
        "matrix_rows": len(matrix_rows),
        "max_offdiag_jaccard": max(
            [r["jaccard"] for r in matrix_rows if r["candidate_a"] != r["candidate_b"] and r["jaccard"] is not None] or [0.0]
        ),
    }


def operational_top2_gate(band: int, theoretical: float | None) -> float | None:
    if theoretical is None:
        return None
    if band == 150:
        return 0.72
    if band == 300:
        return 0.65
    return min(0.99, theoretical + TRANSFER_PENALTY_HIGH)


def operational_top1_gate(band: int, theoretical: float | None) -> float | None:
    if theoretical is None:
        return None
    if band == 300:
        return 0.85
    if band >= 500:
        return 0.74
    return None


def create_adjusted_gate_policy(out_csv: Path, out_md: Path) -> dict[str, Any]:
    curve = pd.read_csv(PHASE0_CURVE)
    rows: list[dict[str, Any]] = []
    for _, r in curve.iterrows():
        band = int(r["band_total_row_diff"])
        top2 = None if pd.isna(r["required_precision_top2_gap"]) else float(r["required_precision_top2_gap"])
        top1 = None if pd.isna(r["required_precision_top1_gap"]) else float(r["required_precision_top1_gap"])
        rows.append(
            {
                "band_total_row_diff": band,
                "expected_public_changed_M_if_half": float(r["expected_public_changed_M_if_half"]),
                "theoretical_top2_precision": top2,
                "phase1_validation_gate_top2_min": operational_top2_gate(band, top2),
                "theoretical_top1_precision": top1,
                "phase1_validation_gate_top1_min": operational_top1_gate(band, top1),
                "transfer_penalty_reference_low": TRANSFER_PENALTY_LOW,
                "transfer_penalty_reference_high": TRANSFER_PENALTY_HIGH,
                "note": "Adjusted upward from theoretical public precision because phase-0 boundary families transferred from validation ~0.505-0.512 to public implied ~0.46-0.49.",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean(rows)).to_csv(out_csv, index=False)

    def fmt(x: Any) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
            return "불가능"
        return f"{float(x):.3f}"

    lines = [
        "# boundary v1 phase-1 gate policy",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- public_lb_feedback_used: true",
        "- candidate_csv_written: false",
        "",
        "## 보정 이유",
        "",
        "phase-0에서 기존 boundary 계열은 validation flip precision 0.505~0.512였지만 public implied precision은 0.46~0.49로 내려갔다.",
        "그래서 phase-1 제출 검토 gate는 이론상 필요한 public precision보다 높게 잡는다.",
        "",
        "## 운영 gate",
        "",
        "| total diff | public changed est. | top2 이론 | top2 validation gate | top1 이론 | top1 validation gate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['band_total_row_diff']} | {row['expected_public_changed_M_if_half']:.1f} | "
            f"{fmt(row['theoretical_top2_precision'])} | {fmt(row['phase1_validation_gate_top2_min'])} | "
            f"{fmt(row['theoretical_top1_precision'])} | {fmt(row['phase1_validation_gate_top1_min'])} |"
        )
    lines.extend(
        [
            "",
            "## 제출 전 고정 조건",
            "",
            "- scored boundary cross-fit 평가 전까지 candidate CSV 생성 금지.",
            "- 300 diff band에서 2~3등권 net gain이 보이지 않으면 full-test candidate 생성 금지.",
            "- 1등 도전은 300 diff band 0.85 근처 또는 500+ diff band 0.73~0.75 유지가 없으면 제출 목표로 보지 않는다.",
            "- 기존 public-failed boundary rows와 overlap이 높으면 precision이 좋아 보여도 candidate로 보지 않는다.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "out_csv": str(out_csv), "out_md": str(out_md)}


def create_scored_eval_readiness(out_md: Path) -> dict[str, Any]:
    split_dirs = sorted([p for p in PANEL20_ROOT.iterdir() if p.is_dir() and p.name.startswith("val_random_uniform_seed")])
    score_files = []
    for p in split_dirs:
        score_files.extend([x for x in p.rglob("*.csv") if "score" in x.name.lower() or "lightgcn" in x.name.lower()])
    payload = {
        "artifact": "boundary_v1_scored_split20_crossfit_eval_readiness",
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "existing_metadata_panel20_available": True,
        "score_coverage_expanded_to_30_50": False,
        "scored_boundary_eval_ready": False,
        "panel20_split_dirs": len(split_dirs),
        "panel20_score_files_detected_under_split_dirs": len(score_files),
    }
    lines = [
        "# boundary v1 scored split20 cross-fit eval — readiness note",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- existing_metadata_panel20_available: true",
        "- score_coverage_expanded_to_30_50: false",
        "- scored_boundary_eval_ready: false",
        "",
        "## 상태",
        "",
        "이 파일은 scored cross-fit 결과가 아니다. 현재 panel20에는 candidates/train_interactions/summary metadata만 있고, split별 LightGCN score coverage가 연결되지 않았다.",
        "따라서 ridge logistic / pairwise logistic cross-fit 평가를 바로 실행할 수 없다.",
        "",
        "## 확인값",
        "",
        f"- panel20 split dirs: {payload['panel20_split_dirs']}",
        f"- score files detected under panel20 split dirs: {payload['panel20_score_files_detected_under_split_dirs']}",
        "",
        "## 다음 작업",
        "",
        "1. panel20 split별 emb128/emb192/current-best proxy score coverage를 생성하거나 연결한다.",
        "2. 기존 public-failed rows와 overlap penalty를 feature/gate에 반영한다.",
        "3. score coverage가 준비된 뒤에만 `boundary_v1_diffband_precision_curve_scored.csv`를 생성한다.",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_filter = audit_row_filter(
        out_dir / "boundary_v1_row_filter_audit_19998_vs_19996.md",
        out_dir / "boundary_v1_row_filter_audit_19998_vs_19996.json",
    )
    overlap = create_overlap_matrix(
        out_dir / "boundary_v1_public_failed_overlap_matrix.csv",
        out_dir / "boundary_v1_public_failed_changed_row_summary.csv",
    )
    gate = create_adjusted_gate_policy(
        out_dir / "boundary_v1_diffband_precision_curve_transfer_adjusted.csv",
        out_dir / "boundary_v1_phase1_gate_policy.md",
    )
    scored_readiness = create_scored_eval_readiness(
        out_dir / "boundary_v1_scored_split20_crossfit_eval.md",
    )

    ledger = {
        "artifact": "boundary_v1_calibration_ledger",
        "role": "negative-control calibration ledger for boundary_specialist_v1_rowflip_constrained",
        "validation_only": True,
        "no_kaggle_submit": True,
        "public_lb_feedback_used": True,
        "candidate_csv_written": False,
        "new_full_test_scoring_performed": False,
        "external_metadata_used": False,
        "phase0_inputs": {
            "calibration_csv": str(PHASE0_CALIBRATION.relative_to(ROOT)),
            "diffband_curve_csv": str(PHASE0_CURVE.relative_to(ROOT)),
        },
        "phase1_prep_outputs": {
            "row_filter_audit_md": "reports/boundary_v1_row_filter_audit_19998_vs_19996.md",
            "row_filter_audit_json": "reports/boundary_v1_row_filter_audit_19998_vs_19996.json",
            "public_failed_overlap_matrix_csv": "reports/boundary_v1_public_failed_overlap_matrix.csv",
            "public_failed_changed_row_summary_csv": "reports/boundary_v1_public_failed_changed_row_summary.csv",
            "transfer_adjusted_curve_csv": "reports/boundary_v1_diffband_precision_curve_transfer_adjusted.csv",
            "phase1_gate_policy_md": "reports/boundary_v1_phase1_gate_policy.md",
            "scored_split20_readiness_md": "reports/boundary_v1_scored_split20_crossfit_eval.md",
        },
        "row_filter_interpretation": row_filter["interpretation"],
        "overlap_summary": overlap,
        "gate_summary": gate,
        "scored_eval_readiness": scored_readiness,
        "submission_readiness": "FAIL__scored_boundary_eval_not_ready",
        "next_action": "Generate/connect split-panel score coverage, then run ridge/pairwise cross-fit no-submit evaluation.",
    }
    write_json(out_dir / "boundary_v1_calibration_ledger.json", ledger)
    print(json.dumps(clean(ledger), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
