NO_SAFE_INTERNAL_AXIS

- The forced boundary v1 probe underperformed materially: public 0.77705 is below both current best 0.77825 and stable emb128 backup 0.77745.
- Its implied public flip precision was weak (~0.432), confirming the validation gate did not transfer.
- The panel20 gate already showed instability: failed top2/top1, worst split -14, and only marginal mean flip precision.
- Remaining internal ideas appear same-family retunes of already closed axes: boundary flips, residual gates, stackers, pseudolabel/transduction, or model-capacity/seed variants.
- With final candidates required to be reproducible, further speculative probes risk public overfit more than they offer independent signal.

Final-packaging items to verify:

- Keep `candidate_rank_blend_emb128_emb192.csv` as final slot1 unless a rules-compliant human decision says otherwise.
- Keep emb128 LightGCN 4-seed public 0.77745 as documented stable backup.
- Record the boundary v1 result and failed gate in the experiment ledger so it is not retried.
- Verify exact training/inference commands, seeds, checkpoints, and config hashes for the selected final candidate.
- Verify final CSV format, row count/order, binary labels, filename, and no accidental regenerated/candidate-risk file substitution.

BOUNDARY_AFTERCARE_ADVISORY_DONE
