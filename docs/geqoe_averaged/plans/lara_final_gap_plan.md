# Plan: Close the Lara {1+:2:1} Gap (52 m → 20 m)

## Current state

The Lara-Brouwer propagator achieves **52 m RSS at 30 days** for the Topex orbit
(J2-only, a=7707.27 km, e=0.0001, i=66.04°).  Lara (2021) reports **~20 m** for the
same {1+:2:1} configuration.  The gap is ~32 m, consistent with an O(J₂²) initialization
error (~22 m theoretical estimate).

### What's working correctly

| Component | Implementation | Status |
|---|---|---|
| Generating function W₁ | Lara (2021) Eq. 6 + C₁ from Eq. 13 | Exact |
| SP corrections (6 elements) | heyoka AD Poisson brackets of W₁ | Exact to O(J₂) |
| Lyddane non-singular form | d(ecosω), d(esinω), d(M+ω) | No 1/e singularity |
| phi 2π-periodicity | atan2(sin(f-ℓ), cos(f-ℓ)) | Fixed (was 8.7 km bug) |
| Secular rates ∂K/∂(L,G,H) | heyoka AD of K = H₀,₀+J₂H₀,₁+J₂²/2·H₀,₂ | Exact |
| BV calibration | Lara (2021) Eq. 23: solve for L̂ | Correct formula |
| Forward map (propagation) | Lyddane SP → osc Keplerian → Cartesian | Exact to O(J₂) |

### What's wrong: the initialization

The osculating-to-mean conversion (`osculating_to_mean_w1` in `short_period.py` line 670)
uses **Cartesian-space subtraction**:

```python
for _ in range(max_iter):
    r_unpert, v_unpert = keplerian_to_cartesian(mean_kep)
    r_fwd, v_fwd = mean_to_cartesian_heyoka(mean_kep)  # forward map with SP
    sp_r = r_fwd - r_unpert                              # Cartesian SP displacement
    r_mean = r_target - sp_r                              # subtract in Cartesian
    mean_kep = cartesian_to_keplerian(r_mean, v_mean)    # convert back ← ERROR HERE
```

The `cartesian_to_keplerian(r_mean, v_mean)` step introduces O(J₂²) errors because the
Cartesian→Keplerian conversion is nonlinear: a small Cartesian perturbation maps to
element changes that depend on the orbital position (especially for e→0 where ω is
poorly defined).  This O(J₂²) initialization bias causes a secular drift of ~32 m over
30 days.

## The fix: Lyddane-space initialization

Instead of subtracting SP corrections in Cartesian space, subtract them directly in
the non-singular Lyddane element space.  The heyoka cfunc already outputs corrections
in Lyddane form: `[da, d(ecosω), d(esinω), dI, dΩ, d(M+ω)]`.

### Algorithm

Given osculating Keplerian elements (a, e, i, Ω, ω, M):

1. **Convert to Lyddane representation**:
   ```
   osc_ecosw = e * cos(ω)
   osc_esinw = e * sin(ω)
   osc_Mpw   = M + ω
   osc_a     = a
   osc_I     = i
   osc_Om    = Ω
   ```

2. **Compute SP corrections at the osculating state** using the heyoka cfunc:
   ```
   E = solve_kepler(M, e)
   L = sqrt(μ·a), G = L·sqrt(1-e²), H = G·cos(i)
   [da, d_ecosw, d_esinw, dI, dOm, d_Mpw] = cfunc([E, ω, L, G, H])
   ```

3. **First-order inverse in Lyddane space** (no Cartesian, no 1/e):
   ```
   mean_a     = osc_a     - da
   mean_ecosw = osc_ecosw - d_ecosw
   mean_esinw = osc_esinw - d_esinw
   mean_I     = osc_I     - dI
   mean_Om    = osc_Om    - dOm
   mean_Mpw   = osc_Mpw   - d_Mpw
   ```

4. **Recover mean Keplerian elements**:
   ```
   mean_e  = sqrt(mean_ecosw² + mean_esinw²)
   mean_ω  = atan2(mean_esinw, mean_ecosw)
   mean_M  = mean_Mpw - mean_ω
   ```

5. **Iterate** (optional, for O(J₂²) accuracy):
   ```
   Recompute SP corrections at the new mean state.
   mean_lyddane = osc_lyddane - sp_corrections(mean_lyddane)
   Repeat until convergence.
   ```

### Why this works

- The Lyddane combinations (ecosω, esinω, M+ω) are **non-singular at e=0**: no 1/e, no
  undefined ω.  The subtraction is a simple linear operation in a well-conditioned space.
- The conversion to/from Keplerian (step 4) uses atan2 which is exact — no nonlinear
  amplification of errors.
- The iteration converges in 2-3 steps because the SP corrections are O(J₂) ≈ 1e-3.

### Implementation location

Replace the body of `osculating_to_mean_w1()` in
`docs/geqoe_averaged/lara_theory/short_period.py` (line 670).  Keep the same function
signature: `osculating_to_mean_w1(osc_kep, mu, Re, J2, max_iter=20, tol=1e-12)`.

The heyoka cfunc is obtained via `_get_sp_heyoka_cfunc(mu, Re, J2)`.

## Point 2: BV calibration (already correct)

The BV calibration in `propagator.py` `_compute_bv_correction()` implements Lara (2021)
Eq. 23 correctly:

```
E₀ = exact osculating energy (including zonal potential)
L̂ = μ / sqrt(2·(-E₀ + J₂·H₀,₁(L',G',H') + J₂²/2·H₀,₂(L',G',H')))
δn = μ²/L̂³ - μ²/L'³
```

The perturbation terms H₀,₁, H₀,₂ are evaluated at L' (first-order mean), not L̂.
This is explicitly stated in the paper (page 14): "If now L' is replaced in Eq. (22)
by the calibrated value L̂..."  Only the Keplerian term -μ²/(2L²) uses L̂.

The BV correction is added to dl/dt in `propagate_mean_delaunay()`:
```
total dl/dt = ∂K/∂L(L') + δn = μ²/L̂³ + perturbation_terms(L')
```

**This is correct.  No change needed.**

However, once the initialization (point 1) is fixed, the mean L' will be slightly
different (by O(J₂²)).  The BV correction will automatically adjust since it depends
on L'.  So fixing point 1 will also implicitly improve the BV accuracy.

## Expected improvement

The theoretical O(J₂²) initialization error is:
- δa ≈ J₂² × a ≈ 1.17e-6 × 7707 ≈ 9 m
- δn/n ≈ 1.5 × J₂² ≈ 1.8e-6
- In-track drift over 30 days: Δx ≈ a × δn × t ≈ 7707 × 1.8e-6 × 2.59e6 ≈ 36 m

Current gap: 52 - 20 = 32 m (consistent with this estimate).

After fix: the initialization error drops to O(J₂³) ≈ 1.3e-9, giving:
- In-track drift: ≈ 0.03 m over 30 days (negligible)
- Expected residual: dominated by second-order SP effects, ~20 m

## Validation

1. **t=0 round-trip**: `osc → mean → osc` position error should be < 0.01 m
2. **Topex 30 days**: RSS at 30 days should be < 30 m (approaching paper's ~20 m)
3. **Cross-check at moderate e**: results should match Cartesian-space initialization
   to O(J₂²) ~ 10 m for orbits with e > 0.01
4. **All 437 existing tests must pass**

## Files involved

| File | Role |
|---|---|
| `lara_theory/short_period.py` | `osculating_to_mean_w1()` — rewrite body |
| `lara_theory/propagator.py` | No changes needed (calls `osculating_to_mean_w1` already) |
| `tests/test_lara_theory.py` | Update Topex 30-day threshold to < 30 m |

## Key context for the session

- The heyoka cfunc is cached globally as `_SP_HEYOKA_CFUNC`.  It takes
  `[E, g, L, G, H]` and returns `[da, d(ecosw), d(esinw), dI, dOm, d(M+w)]`.
  Build with `_get_sp_heyoka_cfunc(mu, Re, J2)`.

- The cfunc requires the eccentric anomaly E, not the mean anomaly M.  Use
  `solve_kepler(M, e)` to get E.

- The phi 2π-periodicity fix (atan2(sin(f-ℓ), cos(f-ℓ))) is critical — without it
  the cfunc gives wrong results at E = 2πk for k ≥ 1.

- Constants: `MU = 398600.4354360959`, `RE = 6378.1366`, `J2 = 1.08262617385222e-3`
  from `astrodyn_core.geqoe_taylor.constants`.

- Run tests: `conda run -n astrodyn-core-env pytest tests/ -q --tb=short`

- Run Topex validation:
  ```bash
  conda run -n astrodyn-core-env python3 -c "
  import sys; sys.path.insert(0, 'docs/geqoe_averaged'); sys.path.insert(0, '.')
  import numpy as np
  from lara_theory.propagator import LaraBrouwerPropagator
  from geqoe_mean.coordinates import kepler_to_rv
  from astrodyn_core.geqoe_taylor.constants import MU, RE, J2
  from astrodyn_core.geqoe_taylor import ZonalPerturbation
  from astrodyn_core.geqoe_taylor.cowell import (
      _build_cowell_heyoka_general_system, _build_par_values)
  import heyoka as hy
  r0, v0 = kepler_to_rv(7707.270, 0.0001, 66.04, 180.001, 270, 180)
  prop = LaraBrouwerPropagator(MU, RE, {2: J2}, use_w1_sp=True)
  prop.initialize(r0, v0, 0.0)
  pert = ZonalPerturbation({2: J2}, mu=MU, re=RE)
  sys_cow, _, pm = _build_cowell_heyoka_general_system(
      pert, mu_val=MU, use_par=True, time_origin=0.0)
  ta = hy.taylor_adaptive(sys_cow, list(r0)+list(v0), tol=1e-15,
      compact_mode=True, pars=_build_par_values(pert, pm))
  t_grid = np.linspace(0, 30*86400, 1000)
  truth = np.empty((len(t_grid), 3))
  for i, t in enumerate(t_grid): ta.propagate_until(t); truth[i] = ta.state[:3]
  lp, _ = prop.propagate(t_grid)
  err = np.linalg.norm(lp - truth, axis=1)
  print(f'RSS(30d)={err[-1]*1000:.1f} m, RMS={np.sqrt(np.mean(err**2))*1000:.1f} m')
  "
  ```

## Estimated effort

The change is ~30 lines of code in `osculating_to_mean_w1()`.  The algorithm is
straightforward (subtract Lyddane corrections, recover Keplerian, iterate).  The
main risk is numerical edge cases at e=0 or critical inclination, which should be
tested explicitly.
