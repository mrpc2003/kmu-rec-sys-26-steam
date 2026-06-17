# Final 5 Non-Duplicate Kaggle Submission Run

User-approved execution of the remaining five KMURecSys26 Steam submissions. This run does not create new model predictions; it only materializes and submits the approved precomputed label files.

## Safety

- user_approved_remaining_5_execution: `True`
- new_model_predictions_created: `False`
- external_metadata_used: `False`
- hidden_label_access: `False`
- submit_now: `True`

## Results

| slot | upload | SHA | status | public | Δ vs 0.77825 | min diff vs prior live |
|---|---|---:|---|---:|---:|---:|
| slot1 | `final5_slot1_tagcf_smoke_LABEL.csv` | `9a8d9bb1` | `SubmissionStatus.COMPLETE` | 0.72674 | -0.05151 | 3144 |
| slot2 | `final5_slot2_pure_als_popa2_LABEL.csv` | `dbeba05b` | `SubmissionStatus.COMPLETE` | 0.73304 | -0.04521 | 2604 |
| slot3 | `final5_slot3_itemknn_bm25_max_LABEL.csv` | `16519ee1` | `SubmissionStatus.COMPLETE` | 0.69333 | -0.08492 | 3556 |
| slot4 | `final5_slot4_rrf_pop_itemknn_ease_LABEL.csv` | `644bc71b` | `SubmissionStatus.COMPLETE` | 0.73704 | -0.04121 | 2796 |
| slot5 | `final5_slot5_stage2_median_z_LABEL.csv` | `af5362b4` | `SubmissionStatus.COMPLETE` | 0.74174 | -0.03651 | 906 |

## Notes

- Exact filename and exact local upload SHA duplicates were blocked before each submit.
- The row-diff guard was checked against locally available live-submission predictions before each submit.
- Final pair remains emb128 4-seed + rankblend unless a result is separately promoted after review.
