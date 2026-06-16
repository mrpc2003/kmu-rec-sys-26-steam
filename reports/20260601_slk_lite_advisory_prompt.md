CRITICAL EXECUTION RULES:
- Answer entirely yourself in this single response.
- Do NOT delegate to sub-agents.
- Do NOT wait for parallel agents.
- Do NOT edit files, run commands, write candidate CSVs, call Kaggle, or inspect hidden test labels.
- This is validation-only advice for an already saturated implicit-feedback Kaggle/classroom Steam played prediction task.
- End with the exact line: SLK_LITE_ADVISORY_DONE

Context:
- Last submission slot remains; no submission unless the human explicitly approves exact file path.
- Primary public surrogate is `val_random_uniform_seed{42,7,123}`; MDE about 0.00355 single-split, panel required.
- Existing axes are closed: popularity/itemKNN/EASE/ALS/BPR/LightGCN capacity/seed ensemble/SGL/DNS/stacking/hyperbolic/sequence/residual-popularity/multi-sampler priors.
- TAG-CF test-time aggregation panel just failed: best fixed mean delta only +0.000767, below MDE.
- LightGCN++ layer-mixture 3-split probe is already running.

Task:
Design the cheapest falsifiable `SL@K-lite` / top-K metric-aligned objective probe to run only if layer-mixture fails or is borderline. It must be implementable in this repo with existing LightGCN code and validation splits, avoid validation-label leakage for any test-transferable claim, and include confound-controlled old-loss continuation vs new top-K/listwise continuation. Provide:
1. Objective formulation precise enough to implement in PyTorch.
2. Training/control protocol with runtimes minimized.
3. Validation gates and kill conditions.
4. Reasons this is genuinely distinct from already closed DNS/hard-negative mining.
5. Any traps likely to create false positives on this per-user 50/50 top-half metric.
