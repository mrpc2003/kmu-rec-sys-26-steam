# emb192 vs emb128 — PAIRED (McNemar) on uniform split

- split `val_random_uniform_seed42` rows=19996 | decode-row acc: emb128 0.76505, emb192 0.76615
- disagreement rows: 660 (3.30%)

## McNemar 2x2 (per-row decode correctness)

| | emb128 right | emb128 wrong |
|---|---:|---:|
| **emb192 right** | 14979 | 341 |
| **emb192 wrong** | 319 | 4357 |

- discordant pairs: 660 | net (emb192−emb128): **+22** | McNemar two-sided p = **0.4137**
- **tier: POSITIVE_NOT_SIGNIFICANT**

emb192 wins 341 vs 319 on discordant rows (net +22), but McNemar p=0.4137 >= 0.05 -> directionally positive yet NOT statistically significant. The +0.0011 edge is within paired noise; a submission is a low-confidence bet (real downside risk it lands <= 0.77745).

## Why paired

The +0.0011 aggregate uniform gain is 1.6x the 0.0007 noise band — borderline. Since the two candidates differ on only ~3.4% of rows, a paired McNemar test on the same rows de-noises the comparison far better than comparing two aggregate accuracies, and directly informs whether a submission is justified.
