I detect a validation-only experiment planning request with strict no-submit/no-file constraints. I’m giving the prioritized 6-12 hour plan only.

## Ranked plan

### 1) Launch first: pseudo-label transduction on validation folds
**Why first:** the gap is too large for packaging or seed averaging. The most plausible remaining +0.5% to +1.0% axis is exploiting the per-user exact top-half candidate structure without external data.

**Core idea:** train from stable LightGCN, infer on validation/test-like candidate groups, add only high-confidence pseudo positives/negatives, retrain or fine-tune, and validate using the same held-out fold structure.

**Validation-only outline:**
- Use existing LightGCN emb128 L4 reg1e-3 4-seed setup as teacher.
- For each validation fold:
  - Train teacher without that fold.
  - Score candidates for held-out users.
  - Convert scores to per-user ranks/probabilities.
  - Pseudo-label only extreme positions:
    - top `k_pos`: confident positives
    - bottom `k_neg`: confident negatives
    - avoid boundary/middle candidates.
  - Sweep conservative settings:
    1. `k_pos=1, k_neg=1`
    2. `k_pos=2, k_neg=2`
    3. `k_pos=3, k_neg=3`
    4. confidence threshold by score margin, not just rank.
  - Retrain student from scratch or fine-tune last 20-40 epochs with pseudo edges weighted low.
- Evaluate only on validation fold accuracy.
- Compare against teacher under identical folds/seeds.

**Success gate:**
- Mean validation gain ≥ `+0.0040` absolute over current LightGCN fold baseline.
- No fold loses more than `-0.0015`.
- Gain survives at least 2 seed groups.
- Pseudo-label count sensitivity is smooth; not one magic threshold.

**Escalation gate to full-test materialization later:**
- Only if validation gain ≥ `+0.0050` and survives a stricter pseudo-label sweep.
- Then and only then prepare a full-test candidate, but not in this run.

---

### 2) SWA / checkpoint averaging on stable LightGCN
**Why second:** cheap, low-risk, may recover calibration/ranking stability. Unlikely alone to close +0.00970, but can compound with pseudo-labeling.

**Validation-only outline:**
- Use existing emb128 L4 reg1e-3 training.
- Save checkpoints from late epochs around best validation window.
- Average model weights or average predictions across checkpoints:
  - last 5 checkpoints
  - best 5 validation checkpoints
  - checkpoints every N epochs after convergence
- Evaluate fold accuracy, not public.
- Test both:
  - weight SWA
  - prediction averaging

**Success gate:**
- Mean validation gain ≥ `+0.0015`.
- No material fold regression.
- Works across emb128 and at least one emb192 run.

**Escalation gate:**
- Only combine with pseudo-labeling if independent validation gains are additive or non-destructive.

---

### 3) Pseudo-label + SWA combined
**Why third:** if pseudo-labeling shifts representation and SWA stabilizes the final ranking, the combination may be the first realistic route toward +0.007 to +0.010.

**Validation-only outline:**
- Take best pseudo-label recipe from experiment 1.
- Train student with late checkpoints.
- Compare:
  - pseudo-label only
  - SWA only
  - pseudo-label + SWA
  - prediction average of pseudo-students across seeds

**Success gate:**
- Combined validation gain ≥ max(single gains) + `0.0010`.
- Stable across folds.
- No evidence that gains come only from a single validation split.

---

### 4) Different internal backbone: NGCF/UltraGCN-style only if already scaffoldable
**Why fourth:** could provide orthogonal signal, but risk/time is higher. Do not spend six hours building infrastructure from scratch.

**Validation-only outline:**
- Only use if repo already has reusable sparse user-item graph loaders/training loops.
- Candidate backbones:
  - NGCF-style message passing
  - UltraGCN-style constraint loss
  - BPR-MF with stronger item/user bias calibration
- Keep candidate scoring identical to current LightGCN pipeline.
- Run small validation sweeps:
  - embedding 64/128
  - reg around `1e-4` to `1e-3`
  - 2 seeds first

**Success gate:**
- Must beat LightGCN validation by ≥ `+0.0030` within first 2-3 hours.
- Otherwise stop immediately.

**Escalation gate:**
- Only blend with LightGCN if validation correlation/error overlap shows complementary wins, not just similar predictions.

---

### 5) User-level adaptive rank correction from validation residuals
**Why fifth:** prior stackers/residuals were weak, but a very constrained correction may help if tied to candidate structure.

**Validation-only outline:**
- Analyze fold errors by:
  - user history length
  - candidate score margin
  - teacher entropy
  - item popularity
- Apply only monotonic, interpretable corrections:
  - popularity penalty/boost by user activity bucket
  - score temperature by user history length
  - margin-based abstention from pseudo-labeling
- Do not train a flexible stacker.

**Success gate:**
- Gain ≥ `+0.0010` and improves pseudo-label selection quality.
- Reject if it only improves one fold or one activity bucket while hurting others.

---

## What to launch first

Launch **validation-only pseudo-label transduction** first.

Reason: current best is already heavily optimized LightGCN/blending, and the public gap is about 194 rows. Seed expansion or final packaging cannot realistically close that. Pseudo-labeling is the only remaining internal axis with plausible step-change upside because it can use the exact candidate-group structure while staying within validation-only development.

Recommended first sweep:
1. Teacher: stable LightGCN emb128 L4 reg1e-3.
2. Pseudo labels: per-user top/bottom extremes only.
3. Student: same architecture, pseudo edges low weight.
4. Compare fold accuracy against teacher.
5. Stop if conservative pseudo labels do not improve validation.

---

## Strict gates before any full-test materialization

Do not create full-test predictions unless all are true:

1. Validation mean improves by at least `+0.0050` absolute over stable baseline.
2. Improvement appears in most folds, not one lucky fold.
3. Worst fold regression is smaller than `-0.0015`.
4. At least two seeds confirm the direction.
5. Hyperparameter sensitivity is broad:
   - nearby pseudo-label thresholds also work.
   - nearby epochs/checkpoints also work.
6. The method does not depend on public leaderboard feedback.
7. No external Steam metadata, scraping, or hidden test labels are used.
8. The final materialization step is explicitly approved later.

---

## Reject immediately

Reject these for the next 6-12 hours:

- More simple seed expansion of the same emb128/emb192 blend.
- Public-LB-driven threshold tuning.
- Reopening EASE/ItemKNN/BM25/TFIDF/EASE-HTR/Turbo-CF/GF-CF/ALS/WMF axes.
- Flexible logreg/stacker approaches that already showed public negative transfer.
- More xSimGCL/SGL/DirectAU/DNS/exact-K variants unless a new validation hypothesis exists.
- External Steam metadata or scraping.
- Any experiment that requires creating `submissions/` files or full-test CSVs now.
- Any approach whose validation lift is smaller than normal seed variance.

---

## Risks

- **Public overfit:** high if decisions are based on the known 0.77825 public score. Keep all decisions fold-based until a later approved materialization.
- **Transduction risk:** pseudo-labeling is allowed only if it uses provided candidate structure and model predictions, not external labels or metadata. Keep pseudo labels conservative and validation-proven.
- **Leaderboard mismatch:** validation gains may not transfer if folds do not mimic test user/item distribution. Require multi-fold stability.
- **False confidence from exact top-half structure:** exploiting candidate grouping is central to the task, but boundary examples are dangerous. Avoid middle-ranked pseudo labels.
- **Time risk:** new backbones can consume the window. Kill them fast unless they beat validation early.

OPENCODE_FIRST_PLACE_AXIS_DONE