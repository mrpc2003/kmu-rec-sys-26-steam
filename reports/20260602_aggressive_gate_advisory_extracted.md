I detect advisory mode: answer only, no tools, no delegation, no file changes.

Next cheapest probe: **boundary-only pairwise agreement re-ranker** over existing emb64/emb128/emb192 validation score files.

### Probe

Use emb128 as the anchor ranking. For each user with cutoff `K = n_items / 2`, only allow changes inside a symmetric boundary band around the cutoff.

For each user:

1. Compute within-user standardized scores:
   - `z64`, `z128`, `z192`
2. Anchor order:
   - `rank128`
3. Boundary sets:
   - `A = items ranked K-B ... K-1` currently predicted played
   - `C = items ranked K ... K+B-1` currently predicted not played
4. For each pair `(a in A, c in C)`, compute pairwise gaps:
   - `d64 = z64[c] - z64[a]`
   - `d128 = z128[c] - z128[a]`
   - `d192 = z192[c] - z192[a]`
5. Promote `c` above `a` only if a fixed rule fires.

### Fixed grid

Boundary width:

```text
B ∈ {8, 16, 32, 64}
```

Vote rule:

```text
vote2:  at least 2 of {d64, d128, d192} > τ
vote192: d192 > τ and d128 > -τ_guard
vote64_192: d64 > τ and d192 > τ
```

Thresholds:

```text
τ ∈ {0.00, 0.05, 0.10, 0.20, 0.35}
τ_guard ∈ {0.00, 0.05, 0.10}
```

Operator:

```text
boundary_bonus[item] =
  count of won pairwise comparisons inside boundary band

final_key =
  rank128 outside boundary unchanged
  inside boundary sorted by:
    played-side-preserving group unless promotion fires,
    then boundary_bonus desc,
    then z128 desc
```

Hard cap:

```text
max promoted bottom items per user ∈ {1, 2, 4, 8}
```

Full fixed grid size is still cheap:

```text
4 B values × 3 vote rules × thresholds × caps
```

No hidden/test candidate materialization. Run only on validation splits.

### Why this is structurally different

This is not another global z/rank blend. It is:

- anchor-preserving
- cutoff-local
- pairwise
- disagreement-targeted
- only changes predictions where top-half accuracy can change

It directly attacks the metric boundary instead of improving full-list ranking.

### Gate criteria

Compare against emb128 4-seed anchor across the same 3 validation splits.

Accept only if all are true:

```text
mean Δ accuracy ≥ +0.00355 MDE
split deltas positive on 3/3 splits
exact paired sign/binomial p ≤ 0.01
fixes > breaks by at least 1.20×
no split Δ < +0.0010
```

If it beats the previous manual-risk z128+z192 blend but remains below MDE, mark **manual-risk only**, do not generate candidate.

### Kill condition

Kill this axis if the best fixed-grid variant has any of:

```text
mean Δ < +0.00355
not positive on all 3 splits
p > 0.01
fix/break ratio ≤ 1.20
best gains mostly from one split
```

If killed, record: “boundary-local pairwise agreement did not clear validation MDE; no submission candidate generated.”

AGGRESSIVE_GATE_ADVISORY_DONE