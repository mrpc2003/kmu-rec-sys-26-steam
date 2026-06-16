Verdict: NO_SAFE_INTERNAL_AXIS

Rationale bullets
- Pseudo-label transduction is now closed: top1 margin0 had high pseudo precision 0.8442 but zero mean lift, and margin filtering raised precision to 0.9109/0.9444 while worsening mean validation accuracy.
- This indicates pseudo labels are reinforcing teacher bias or shifting the exact top-half decision boundary, not adding useful signal.
- Checkpoint averaging/SWA-like smoothing is also closed: best mean delta was only +0.000067 with 1/3 positive splits, far below a meaningful gate.
- Existing internal axes already cover model capacity, seed expansion, CF variants, graph filters, sequence models, stackers, exact-K losses, and text/date/hours/popularity residuals.
- Given the known public gap to #1 is about +0.00970, the remaining plausible lift is unlikely to come from another small internal validation-only tweak.

If NO_SAFE_INTERNAL_AXIS: what is closed and what remains externally/rule-dependent
- Closed internally: pseudo-label transduction, margin-filtered pseudo labels, checkpoint averaging, LightGCN capacity/seed scaling, broad CF/model-family search, stackers, sequence objective variants, and weak internal side-feature residuals.
- Remaining path: professor approval for external Steam metadata, explicitly approved before collection/use.
- If external metadata is not approved, correct next move is final packaging: freeze the best validated rank_blend/emb128-family configuration, document validation evidence, safety constraints, and avoid public-LB-driven tuning.

Safety flags summary
- No Kaggle submit.
- No hidden labels.
- No public-LB threshold tuning.
- No full-test candidate CSV or submission materialization.
- External Steam metadata only with professor approval before use.

OPENCODE_POST_PSEUDO_MARGIN_AXIS_DONE