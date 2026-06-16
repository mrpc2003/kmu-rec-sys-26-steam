# Aggressive user-gated z-blend probe

- verdict: **MANUAL_RISK_SIGNAL**
- variants: `200`
- strict pass count: `0`
- safety: validation-only; no hidden/test read; no candidate CSV; no Kaggle submit.

| rank | variant | mean Δ | min~max Δ | splits+ | fixes | breaks | ratio | p | changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `full_user_zblend__all_users` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 2 | `full_user_zblend__margin_low_q0.6` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 3 | `full_user_zblend__margin_low_q0.8` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 4 | `full_user_zblend__diff192_high_q0.2` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 5 | `full_user_zblend__diff192_high_q0.4` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 6 | `full_user_zblend__changed_high_q0.2` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 7 | `full_user_zblend__changed_high_q0.4` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 8 | `full_user_zblend__changed_high_q0.6` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 9 | `full_user_zblend__changed_high_q0.8` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 10 | `full_user_zblend__candn_high_q0.2` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 11 | `full_user_zblend__candn_high_q0.4` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 12 | `full_user_zblend__margin_low_q0.4_or_diff192_high` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 13 | `full_user_zblend__margin_low_q0.6_and_changed_high` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 14 | `full_user_zblend__margin_low_q0.6_or_diff192_high` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 15 | `full_user_zblend__margin_low_q0.8_and_changed_high` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 16 | `full_user_zblend__margin_low_q0.8_or_diff192_high` | +0.001500 | +0.001100~+0.002200 | 3/3 | 506 | 416 | 1.216 | 0.003356 | 922 |
| 17 | `full_user_zblend__margin_low_q0.4` | +0.001467 | +0.001100~+0.002100 | 3/3 | 504 | 416 | 1.212 | 0.004102 | 920 |
| 18 | `full_user_zblend__margin_low_q0.4_and_changed_high` | +0.001467 | +0.001100~+0.002100 | 3/3 | 504 | 416 | 1.212 | 0.004102 | 920 |
| 19 | `boundary_user_zblend__all_users__B4` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 20 | `boundary_user_zblend__all_users__B8` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 21 | `boundary_user_zblend__all_users__B16` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 22 | `boundary_user_zblend__margin_low_q0.4__B4` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 23 | `boundary_user_zblend__margin_low_q0.4__B8` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 24 | `boundary_user_zblend__margin_low_q0.4__B16` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
| 25 | `boundary_user_zblend__margin_low_q0.6__B4` | +0.001367 | +0.001000~+0.001800 | 3/3 | 504 | 422 | 1.194 | 0.007738 | 926 |
