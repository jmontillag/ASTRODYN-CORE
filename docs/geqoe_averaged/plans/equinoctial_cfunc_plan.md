# Plan: Equinoctial SP cfunc (heyoka compiled fast path)

## Goal

Compile the equinoctial short-period corrections (dp1, dp2, dq1, dq2)
into a heyoka cfunc for SIMD batch evaluation, replacing the lambdified
Python loop in `mean_to_osculating_state_equinoctial_batch()`.

## Architecture

**Revised approach**: The equinoctial cfunc outputs only `[dp1, dp2, dq1, dq2]`
(4 outputs, no M channel). The L reconstruction continues to use the polar
cfunc for `dPsi + dM`. This keeps the equinoctial cfunc small (52 terms vs
~150 for polar+M) and avoids duplicating the M channel.

## Variable Layout (8 inputs)

```
Index  Variable      Source
  0    cos_f         Re(F) = cos(true anomaly)
  1    sin_f         Im(F) = sin(true anomaly)
  2    q             eccentricity parameter
  3    Q             inclination parameter
  4    cos_omega     (p2*q2 + p1*q1) / (g*Q)  [no atan2]
  5    sin_omega     (p1*q2 - p2*q1) / (g*Q)  [no atan2]
  6    cos_Omega     q2 / Q                    [no atan2]
  7    sin_Omega     q1 / Q                    [no atan2]
```

Parameters: `[s2, s3, s4, s5]` (same as polar cfunc).

Outputs: `[dp1, dp2, dq1, dq2]`.

## Output Assembly

For each channel (zeta, eta), accumulate Re and Im of `sum_m(c_m * w^m)`:

```
dzeta_red_re = sum s_n * [c_re*cos(mω) - c_im*sin(mω)]
dzeta_red_im = sum s_n * [c_re*sin(mω) + c_im*cos(mω)]
```

Then rotate by exp(iΩ):

```
dp2 = dzeta_red_re * cos_Ω - dzeta_red_im * sin_Ω
dp1 = dzeta_red_re * sin_Ω + dzeta_red_im * cos_Ω
```

Same for eta → (dq2, dq1).

## Data Sources

| Source | Entries | For |
|--------|---------|-----|
| EQNOC_SHORT_DATA (zeta, eta) | 32 | Rational SP corrections |
| EQNOC_LOG_DATA (zeta, eta) | 20 | Log-term corrections |
| **Total** | **52** | |

M channel handled by existing polar cfunc (18 rational + 9 log = 27 terms).

## Files to Change

### `heyoka_compiled.py` — add:
1. `_load_eqnoc_short_data()` — AST loader for EQNOC_SHORT_DATA
2. `_load_eqnoc_log_data()` — AST loader for EQNOC_LOG_DATA
3. `build_eqnoc_sp_cfunc()` — builds the 8-input, 4-output cfunc
4. `get_eqnoc_sp_cfunc()` — lazy-caching wrapper
5. Cache globals: `_CACHED_EQNOC_SP_CFUNC`, `_CACHED_EQNOC_SP_CFUNC_INFO`

### `short_period.py` — add/modify:
1. Import: `from .heyoka_compiled import get_eqnoc_sp_cfunc as _get_eqnoc_sp_cfunc`
2. Feature flag: `_HAS_EQNOC_SP_CFUNC`
3. `_evaluate_eqnoc_sp_batch_cfunc()` — batch wrapper for the cfunc
4. Modify `mean_to_osculating_state_equinoctial_batch()` — add cfunc fast path

## Key Details

### Chebyshev tables
- cos/sin(k*f): max_k = 12 (same as polar)
- cos/sin(m*ω): max_m = 6 (EQNOC data has |m| up to 6, vs 5 for polar)

### Log terms
- φ = atan2(q·sin(f), 1+q·cos(f)) — same as polar
- EQNOC_LOG_DATA coefficients are complex (contain I) → split via sp.re/sp.im
  and convert each part separately with `_sympy_to_heyoka` (not the complex variant)
- Multiplied by φ before accumulating into dzeta/deta re/im

### L reconstruction in batch
The cfunc fast path in `mean_to_osculating_state_equinoctial_batch`:
1. Call equinoctial cfunc → dp1, dp2, dq1, dq2
2. Call `evaluate_truncated_short_period_batch` → dPsi, dM (polar cfunc, already fast)
3. L_osc = L_mean + dPsi + dM
4. K_osc = solve_kepler_gen(L_osc, p1_osc, p2_osc)

### Degenerate cases (g→0, Q→0)
- cos/sin(ω) computed from Cartesian: `(p2*q2+p1*q1)/(g*Q)` etc.
- Guard with `g_safe = where(g>0, g, 1)`, `Q_safe = where(Q>0, Q, 1)`
- Set cos_ω=1, sin_ω=0 when g=0 or Q=0

## Performance Expectations

- **Build**: ~10-30s (52 terms, SymPy parse + heyoka conversion)
- **Compile**: ~2-8s (smaller tree than polar cfunc)
- **Per-call**: SIMD native, ~100x faster than lambdified Python loop
- **Overall batch speedup**: ~2-5x (polar cfunc call for L still needed)

## Testing

1. Numerical equivalence: cfunc vs lambdified at multiple test orbits (1e-14 tolerance)
2. Edge cases: g=0.001, Q=0.01, g=1e-6
3. Full regression: extended_validation.py with equinoctial route
