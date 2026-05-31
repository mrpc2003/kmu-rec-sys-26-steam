# MiniLM Semantic Axis — UNIFORM gate (last orthogonal signal)

- split `val_random_uniform_seed42` rows=19996 | text known rate 0.9995
- **text solo uniform: 0.63853** (pop floor 0.684, cf solo 0.76205)
- corr(raw)=0.2616  corr(within-user z)=**0.4606**
- **50/50 z-blend uniform: 0.73085** vs emb128 ref 0.76505 → **-0.03420**
- diagnostic grid (w_text→acc, stacker-trap, not trusted): {0.05: 0.76465, 0.1: 0.76375, 0.2: 0.76165, 0.3: 0.75705}
- **tier: REJECT_WEAK** — text solo 0.63853 < popularity floor 0.684 -> too weak. corr_z=0.461. 50/50 z-blend 0.73085 vs emb128 ref 0.76505 (-0.03420). REJECT as axis (orthogonal but too weak, same as TF-IDF/DirectAU).

