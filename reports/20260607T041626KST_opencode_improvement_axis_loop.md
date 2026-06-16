# OpenCode Improvement Axis Loop — 20260607T041626KST

## Verdict

`NO_SAFE_AXIS` — I did not launch a new probe. The current evidence does not support a fresh independent, bounded, validation-only axis that can plausibly clear the strict gate without repeating closed/stalled/quarantined families.

## Safety flags

- `validation_only`: `true`
- `candidate_csv_written`: `false`
- `kaggle_submit_executed`: `false`
- `hidden_labels_used`: `false`
- `external_steam_scraping_used`: `false`
- `git_stage_commit_push_executed`: `false`

## Stalled/completed classifications

- **userknn_gated_residual_fine**: `STALLED_INCOMPLETE`
  - reason: Prior tick reports Hermes terminated process group 18483 after about 14h51m; current inspection still finds no JSON/MD reports and a tail of repeated RuntimeWarning lines from userknn_residual_probe.py with no split completion artifact. Broad fine-grid must not be relaunched as-is.
- **jackknife_uncertainty_boundary_expanded**: `FAILED_INCOMPLETE_NO_METRIC_REPORT`
  - reason: Dead process, missing JSON/MD, and only a short partial log stopping mid val_random_uniform_seed123; no metric report exists for escalation.
- **jackknife_uncertainty_boundary_smoke**: `WEAK_SIGNAL_STRICT_GATE_FAIL`
  - `mean_delta_vs_base`: `0.0003667400146696309`
  - `min_delta_vs_base`: `-0.0012002400480095599`
  - `positive_splits`: `2/3`
  - `fixes`: `252`
  - `breaks`: `230`
  - `pooled_p_exact`: `0.33881500709211204`
- **dns_pool1_panel**: `CLOSED_NO_CANDIDATE`
  - reason: Single-split/panel evidence remains below strict gate and was already rejected as split-specific/public-noise risk.
  - `pooled_p_exact`: `0.2509927551833466`
- **userknn_gated_residual_smoke**: `WEAK_SIGNAL_STRICT_GATE_FAIL`
  - reason: Best mean delta is below +0.0015 and p is not <0.05; narrower p<0.05 variants have mean delta about +0.0008, still below strict gate.
  - `top_mean_delta_vs_base`: `0.0009001800360071884`
  - `top_min_delta_vs_base`: `0.0002000400080015563`
  - `top_positive_splits`: `3/3`
  - `top_fixes`: `406`
  - `top_breaks`: `352`
  - `top_pooled_p_exact`: `0.05415183075028983`

## New axis decision

NO_SAFE_AXIS: after inspecting current reports, logs, process state, quarantine, and trusted validation scripts, I did not find a fresh independent validation-only axis that is credible, bounded, materially different from closed axes, and capable of satisfying the strict gate in this run. Launching a probe now would either repeat UserKNN/jackknife/boundary/rankblend/capacity families already stalled, weak, quarantined, or public-negative, or would be an underpowered one-split blip explicitly disallowed by the objective.

## New probe

No probe launched. No command, PID file, log, probe report, or artifact directory was created for a new probe in this run.

## Ranked next-axis hypotheses and blockers

1. **Bounded UserKNN diagnostic after fixing NaN standardization only if the implementation bug is isolated first**
   - why not launched now: The broad fine-grid just stalled for ~15h with repeated invalid-divide warnings and no reports; relaunching it violates the prompt. A materially different future probe would need a tiny deterministic diagnostic proving finite features and a hard per-split row/sample cap before any 3-split run.
   - blockers: stalled broad grid; implementation warning source unresolved; prior smoke mean delta below +0.0015
2. **Out-of-fold residual calibration constrained to already available validation artifacts only**
   - why not launched now: Closest historical boundary/rankblend residual variants produced public-negative transfer or quarantine despite positive validation deltas, so a new residual scan lacks a safe independence argument without a new non-leaky validation design.
   - blockers: rankblend and boundary residual families quarantined/public-negative; near-duplicate risk; strict gate/public transfer mismatch
3. **New backbone/loss family beyond LightGCN capacity frontier**
   - why not launched now: SASRec, TagCF, SGL/XSimGCL, Hyperbolic, MultiVAE, AlphaRec/text/semantic families were already weak/redundant/negative; a genuinely new model would exceed the bounded 20-minute validation-only budget and lacks a current credible clue.
   - blockers: closed structural families; bounded runtime; no fresh evidence exceeding strict gate
4. **DNS/capacity micro-ensemble refinement**
   - why not launched now: DNS pool=1 and capacity/frontier refinements are closed as marginal or public-noise, with observed deltas around +0.0010 or less and non-significant paired tests; below strict threshold.
   - blockers: mean delta below +0.0015; non-significant p-values; capacity frontier/public-tested variants failed to beat current best

## Strict gate retained

- mean Δ threshold: `0.0015`
- min Δ nonnegative: `true`
- required positive splits: `3/3`
- fixes > breaks required: `true`
- pooled exact/McNemar p < `0.05`

## Report outputs

- Markdown: `reports/20260607T041626KST_opencode_improvement_axis_loop.md`
- JSON: `reports/20260607T041626KST_opencode_improvement_axis_loop.json`

## Manual QA / validation-only surface

Reports were written only under `reports/`. No CSV was written under `submissions/`, no full-test candidate was materialized, no Kaggle submit command was invoked, and no git staging/commit/push was performed.
