I detect validation-only methodology advice: you need a cheap, falsifiable backup probe design, not repo actions. Here is the implementable SL@K-lite probe.

## 1. Objective formulation

Use existing LightGCN embeddings and sampled candidates per user. Continue from the same checkpoint used by old BPR/LightGCN runs.

For each training user `u`, sample:

- positives `P_u`: observed train positives
- negatives `N_u`: unobserved items sampled from the same sampler as the old-loss control
- candidate set `C_u = P_batch ∪ N_batch`
- scores `s_ui = dot(e_u, e_i)`

Define a smooth top-K/listwise surrogate over the candidate set.

### SL@K-lite objective

Let `K_train` be small, e.g. `20` or `50`.

For each user:

```python
scores = model.score_user_items(u, C_u)          # [m]
labels = is_positive(C_u)                       # [m], 0/1

# Soft top-K weights
tau = 0.05 to 0.2
soft_rank_weight = torch.softmax(scores / tau, dim=0)

# Option A: soft precision@K-style
loss_u = -torch.log((soft_rank_weight * labels).sum() + eps)
```

Better, use a pairwise-listwise hybrid that is more stable:

```python
pos_scores = scores[labels == 1]
neg_scores = scores[labels == 0]

# Smooth top negatives only
neg_weights = torch.softmax(neg_scores / tau, dim=0).detach()
hard_neg_score = (neg_weights * neg_scores).sum()

# Top-K aligned margin
loss_u = F.softplus(hard_neg_score - pos_scores.mean())
```

Recommended probe objective:

```python
loss = mean_u(
    alpha * BPR_loss(u)
    + (1 - alpha) * SLK_lite_loss(u)
)
```

Use `alpha = 0.5` for the new-loss continuation. Do not remove BPR entirely in the first cheap probe; pure listwise continuation can destabilize rankings quickly and create noisy false negatives.

## 2. Training/control protocol

Run only if LightGCN++ layer-mixture is below gate or borderline.

Use a strict paired continuation design:

### Starting point

Same checkpoint for both arms:

- `old-loss continuation`: continue with current BPR objective
- `SL@K-lite continuation`: continue with `0.5 * BPR + 0.5 * SLK-lite`

Everything else identical:

- same split
- same seed
- same checkpoint
- same sampler
- same user batches
- same learning rate
- same number of continuation epochs
- same evaluation code

### Minimal panel

Use the public surrogate panel already trusted:

- `val_random_uniform_seed42`
- `val_random_uniform_seed7`
- `val_random_uniform_seed123`

Do not tune on one split and claim transfer. Treat the 3-split paired delta as the unit of evidence.

### Runtime-minimized schedule

Per split:

1. Load best existing LightGCN checkpoint.
2. Continue old-loss control for very short budget:
   - 1 epoch
   - 2 epochs
   - optionally 4 epochs only if 1/2 epoch results are non-degrading
3. Continue SL@K-lite arm for same budgets.
4. Evaluate after each budget.

Suggested cheap grid:

```text
tau:       0.10 only
alpha:     0.50 only
K/cands:   existing batch positives + 128 or 256 negatives/user
epochs:    1, 2, optional 4
splits:    42, 7, 123
```

Only expand if paired deltas are clearly positive.

Do not run a large hyperparameter search. This is a falsification probe, not a final optimizer sweep.

## 3. Validation gates and kill conditions

Because single-split MDE is about `0.00355`, do not trust a one-split win.

### Pass gate

Proceed only if all are true:

```text
mean paired delta over 3 splits >= +0.0015
median paired delta > 0
at least 2/3 splits positive
SL@K-lite beats old-loss continuation, not just the original checkpoint
no split worse than -0.0015
```

Stronger submit-consideration gate:

```text
mean paired delta >= +0.0025
3/3 or very strong 2/3 positive
best epoch chosen by predeclared rule, not cherry-picked per split
```

Predeclared epoch rule:

- choose the earliest epoch where 3-split mean delta peaks among `{1,2,4}`
- if epoch 1 wins but epoch 2 collapses, kill as unstable

### Kill conditions

Stop immediately if:

```text
old-loss continuation also improves similarly
SL@K-lite gain over old-loss control < +0.0010 mean
only one split carries the gain
any split drops below -0.0020
pure popularity/top-head items increase while tail/user-personalized hits fall
validation gain appears only after choosing per-split best epoch
```

Also kill if the metric lift is mostly from users with very high train-degree unless the hidden metric is known to share that bias.

## 4. Why this is distinct from DNS/hard-negative mining

DNS changes which negatives are sampled or emphasized. It is still usually optimizing pairwise separation:

```text
s(u, positive) > s(u, selected negative)
```

SL@K-lite changes the shape of the objective to optimize the top of the ranked candidate set directly.

Key distinction:

- DNS: “pick harder negatives for BPR”
- SL@K-lite: “make the mass of the top-ranked set contain positives”

The proposed objective couples all candidate scores for a user through softmax/top-K weighting. A negative’s effect depends on its relative rank among other candidates, not just an independent pairwise margin. That makes it listwise/top-K aligned rather than merely hard-negative pairwise training.

The old-loss continuation control is essential because otherwise any improvement could be from extra training time, lower LR continuation, or checkpoint restart effects.

## 5. Traps likely to create false positives

Main traps:

1. **Validation-label leakage**
   - Do not construct top-K candidates using validation positives.
   - Do not sample negatives conditioned on validation labels.
   - Do not tune epoch/objective per validation split and claim test transfer.

2. **Per-split cherry-picking**
   - Choosing best epoch separately for seed42/7/123 will overstate lift.
   - Use one predeclared epoch-selection rule across the panel.

3. **Uncontrolled continuation**
   - Comparing SL@K-lite continuation to the frozen original checkpoint is invalid.
   - Must compare against same-budget old-loss continuation.

4. **Sampler confounding**
   - If SL@K-lite uses harder/more negatives than control, improvement may be DNS-like.
   - Keep candidate sampling identical between arms.

5. **Top-half metric mismatch**
   - If the classroom metric rewards per-user 50/50 top-half ordering, extreme global-popularity sharpening may help validation but fail hidden.
   - Track degree buckets and popularity buckets; kill if lift is only head-item concentration.

6. **Softmax temperature instability**
   - Too-low `tau` turns into noisy max-negative training.
   - Start with `tau=0.10`; do not sweep aggressively unless the controlled probe passes.

7. **Batch candidate illusion**
   - Optimizing within tiny candidate batches may not transfer to full-ranking evaluation.
   - Use at least 128 negatives/user if feasible; 256 is better if still cheap.

The cheapest falsifiable version is therefore: same checkpoint, same sampler, same batches, 3 validation splits, old BPR continuation versus `0.5 BPR + 0.5 soft top-negative/listwise SL@K-lite`, evaluated at fixed short epochs with paired deltas. If it cannot beat old-loss continuation on the panel, do not spend the last slot on it.

SLK_LITE_ADVISORY_DONE