#!/usr/bin/env python3
"""Retrospective similarity/post-transfer analysis for autorun submissions.

This is a report-only utility. It never submits to Kaggle. It updates the runner state with
family quarantines when submitted variants show negative public transfer, so future
watchdog restarts cannot immediately spend quota on nearby tuned siblings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aggressive_quota_runner as runner

ROOT = runner.ROOT


def validation_by_variant() -> dict[str, dict[str, Any]]:
    data = runner.load_validation_cache()
    return {v["variant"]: v for v in data.get("all_variants", [])}


def result_path(result: dict[str, Any]) -> Path | None:
    if result.get("candidate_file"):
        p = Path(result["candidate_file"])
        if p.exists():
            return p
    return runner.local_submission_path(result.get("fileName"))


def main() -> None:
    ts = runner.stamp()
    state = runner.load_state()
    validations = validation_by_variant()
    submissions = state.get("submission_results", [])
    _, rows = runner.refresh_submissions(ts + "_postmortem")
    q = runner.quota_and_best(rows)

    analyses: list[dict[str, Any]] = []
    sha_to_results: dict[str, list[str]] = {}
    for r in submissions:
        if r.get("sha256"):
            sha_to_results.setdefault(r["sha256"], []).append(r.get("variant") or r.get("fileName") or "unknown")

    paths = [(r, result_path(r)) for r in submissions]
    for r, p in paths:
        variant_name = r.get("variant") or "unknown"
        validation = validations.get(variant_name, {})
        validation_delta = r.get("validation_mean_delta_vs_rankblend")
        if validation_delta is None:
            validation_delta = validation.get("mean_delta_vs_rankblend")
        public_delta = r.get("delta_vs_previous_best")
        transfer_ratio = None
        if public_delta is not None and validation_delta:
            transfer_ratio = float(public_delta) / float(validation_delta)
        family = r.get("family") or runner.variant_family(variant_name)
        row_diffs = []
        if p and p.exists():
            for other, op in paths:
                if other is r or not op or not op.exists():
                    continue
                try:
                    diff = runner.row_diff_count(p, op)
                except Exception as exc:
                    row_diffs.append({"other": other.get("variant") or other.get("fileName"), "error": repr(exc)})
                    continue
                row_diffs.append({
                    "other": other.get("variant") or other.get("fileName"),
                    "other_sha256": other.get("sha256"),
                    "row_diff": diff,
                })
            for ref in runner.reference_paths(state, q, exclude_sha=r.get("sha256")):
                try:
                    diff = runner.row_diff_count(p, ref["path"])
                except Exception as exc:
                    row_diffs.append({"other": ref["label"], "error": repr(exc)})
                    continue
                row_diffs.append({"other": ref["label"], "other_sha256": ref.get("sha256"), "row_diff": diff})
        exact_duplicate_group = sha_to_results.get(r.get("sha256"), []) if r.get("sha256") else []
        quarantine = bool(public_delta is not None and public_delta <= 0)
        if transfer_ratio is not None and transfer_ratio <= 0:
            quarantine = True
        analysis = {
            "variant": variant_name,
            "family": family,
            "fileName": r.get("fileName"),
            "sha256": r.get("sha256"),
            "publicScore": r.get("publicScore"),
            "previous_public_best": r.get("previous_public_best"),
            "public_delta_vs_previous_best": public_delta,
            "validation_mean_delta_vs_rankblend": validation_delta,
            "transfer_ratio": transfer_ratio,
            "exact_duplicate_variants_with_same_sha": exact_duplicate_group,
            "row_diffs": row_diffs,
            "quarantine_family": quarantine,
        }
        analyses.append(analysis)
        if quarantine:
            state.setdefault("quarantined_families", {})[family] = {
                "time_kst": runner.now_kst().isoformat(),
                "reason": "retrospective_negative_or_non_improving_public_transfer",
                "source_variant": variant_name,
                "public_delta_vs_previous_best": public_delta,
                "validation_mean_delta_vs_rankblend": validation_delta,
                "transfer_ratio": transfer_ratio,
                "sha256": r.get("sha256"),
            }

    duplicate_sha_groups = {sha: variants for sha, variants in sha_to_results.items() if len(variants) > 1}
    payload = {
        "kind": "autorun_batch_similarity_postmortem",
        "timestamp_kst": ts,
        "competition": runner.COMP,
        "current_live_best": q.get("best_score"),
        "current_live_best_row": q.get("best_row"),
        "submitted_count_in_state": len(submissions),
        "duplicate_sha_groups": duplicate_sha_groups,
        "min_autonomous_row_diff": runner.MIN_AUTONOMOUS_ROW_DIFF,
        "hard_duplicate_row_diff": runner.HARD_DUPLICATE_ROW_DIFF,
        "analyses": analyses,
        "policy_revision": {
            "rapid_batch_quota_burn_allowed": False,
            "exact_sha_duplicate_allowed": False,
            "near_duplicate_under_500_row_diff_allowed": False,
            "same_family_after_non_improving_public_transfer_allowed": False,
            "sleep_after_submit_seconds": runner.DEFAULT_SLEEP_AFTER_SUBMIT,
        },
    }

    state["last_similarity_postmortem_json"] = f"reports/{ts}_autorun_batch_similarity_postmortem.json"
    state["operating_policy_revision"] = payload["policy_revision"]
    state.setdefault("notes", []).append(
        f"{ts}: user correction applied — no rapid five-submit batches; block exact/near-duplicate CSVs and quarantine failed families after post-analysis."
    )
    runner.save_state(state)

    js = ROOT / f"reports/{ts}_autorun_batch_similarity_postmortem.json"
    md = ROOT / f"reports/{ts}_autorun_batch_similarity_postmortem.md"
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Autorun batch similarity postmortem",
        "",
        f"- timestamp: `{ts}`",
        f"- current live best: `{q.get('best_score')}`",
        f"- submissions analyzed: `{len(submissions)}`",
        f"- duplicate SHA groups: `{len(duplicate_sha_groups)}`",
        f"- new autonomous min row diff: `{runner.MIN_AUTONOMOUS_ROW_DIFF}`",
        f"- rapid batch quota burn allowed: `False`",
        "",
        "## Duplicate SHA groups",
    ]
    if duplicate_sha_groups:
        for sha, variants in duplicate_sha_groups.items():
            lines.append(f"- `{sha[:12]}...`: " + ", ".join(f"`{v}`" for v in variants))
    else:
        lines.append("- none")
    lines += ["", "## Per-submission transfer", ""]
    for a in analyses:
        lines.append(
            f"- `{a['variant']}`: public={a.get('publicScore')}, publicΔ={a.get('public_delta_vs_previous_best')}, "
            f"valΔ={a.get('validation_mean_delta_vs_rankblend')}, transfer={a.get('transfer_ratio')}, "
            f"quarantine={a.get('quarantine_family')}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(js), "md": str(md), "duplicate_sha_groups": duplicate_sha_groups, "quarantined_families": state.get("quarantined_families", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
