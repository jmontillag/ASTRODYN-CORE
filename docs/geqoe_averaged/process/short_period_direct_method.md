# Direct Residue Method for GEqOE Short-Period Kernels

## Problem Statement

The first-order near-identity averaging map for GEqOE with truncated zonal
harmonics requires solving, for each variable (g, Q, Psi, Omega, M) and each
zonal degree n, the homological equation:

    du_1/dM = f(F) - <f>_M

where F = exp(i f) is the complex true anomaly exponential, M = L - Psi is the
uniformly advancing fast phase, and the forcing f(F) is a finite Laurent
polynomial in F for each fixed w-harmonic m (where w = exp(i(Psi - Omega))).

## Previous Approach (Blocked)

The previous implementation (`zonal_short_period_general.py`) solved this by
converting to the F variable:

    u_1 = integral of [f(F) - <f>] * (dM/dF) dF

and calling SymPy's `ratint` on each F^k * dM/dF term individually. This had
two fatal problems:

1. **Log term crash**: Individual primitives contain log(F), log(F+q), and
   log(1+qF) terms that cancel in the sum but cause `_mean_of_f_expression`
   to fail with `NotImplementedError` when computing residues.

2. **Expression swell**: ~20 separate `ratint` calls per harmonic, each
   producing large rational expressions that are then combined and simplified.
   For Psi and Omega at J2, this never completed.

## New Approach: Direct Residue Decomposition

### Key Insight

The Jacobian dM/dF has known, fixed pole structure:

    dM/dF = -i(1-q^2)^3 / (1+q^2) * F / [(F+q)^2 (1+qF)^2]

So the full integrand for each w-harmonic m is:

    scale * N(F) / [F^s * (F+q)^2 * (1+qF)^2]

where N(F) is a polynomial in F with coefficients in q, Q, and s accounts
for negative F-powers in the forcing.

### Algorithm

1. **Combine numerator**: Build N(F) = sum_k a_{m,k} F^{k+1} - mean_m * F
   as a single polynomial (shift by F^s to clear negative powers).

2. **Polynomial long division**: N(F) = Q(F) * denom + N_rem(F) using
   `sp.div`. Gives polynomial quotient Q(F) of small degree.

3. **Double-pole residues by evaluation**:
   - B = N_rem(-q) / [(-q)^s * (1-q^2)^2]  (at F = -q)
   - D = N_rem(-1/q) / [(-1/q)^s * ((-1/q+q))^2]  (at F = -1/q)

4. **F=0 residues by Taylor recurrence**: For shift >= 2, compute Taylor
   coefficients of chi(F) = N_rem / [(F+q)^2(1+qF)^2] at F=0 via the
   convolution identity P_k = sum Q_j chi_{k-j}, solved iteratively.

5. **Assemble periodic antiderivative** (all log terms cancel by periodicity):
   - Polynomial: integral of Q(F)
   - Double poles: -B/(F+q), -D/(q(1+qF))
   - Higher F-poles: -E_j/((j-1)F^{j-1}) for j >= 2

6. **Zero-mean gauge**: Compute <u_1>_M structurally (term-by-term using
   cached formulas for <1/(F+q)>_M and <1/(1+qF)>_M) instead of substituting
   F -> F(z) into the full expression.

### Why Log Terms Cancel

On the unit circle |F| = 1 (physical orbits), F traces a closed contour as M
advances by 2pi. The function [f(F) - <f>] has zero M-average by construction,
so its integral over one period is zero. Since log(F) = if is not single-valued
on this contour, its coefficient must be exactly zero. Similarly for log(F+q)
and log(1+qF), since -q and -1/q lie inside/outside the unit circle
respectively.

### Structural Mean Formulas

The M-averages of the pole terms are computed by direct residue calculus
(substituting F -> (z-q)/(1-qz) and evaluating at z=0):

    <1/(F+q)>_M = -q(2+q^2) / ((1-q^2)(1+q^2))
    <1/(1+qF)>_M = (1+2q^2) / ((1-q^2)(1+q^2))
    <F^k>_M = mean_f_power(k)  [existing cached function]

This avoids the expression-swell catastrophe of substituting the full rational
expression.

## Performance

### Initial direct method (v1)

| Variable | Old (ratint) | Direct v1  | Speedup |
|----------|-------------|------------|---------|
| J2 g     | ~8s         | 3.3s       | 2.4x    |
| J2 Q     | ~10s        | 1.3s       | 7.7x    |
| J2 M     | ~4s         | 1.4s       | 2.9x    |
| J2 Psi   | BLOCKED     | 43s        | inf     |
| J2 Omega | BLOCKED     | 3.1s       | inf     |

The old method failed entirely for Psi and Omega due to the log-term crash.

### Optimizations for higher zonal degrees

Scaling to J3-J5 exposed three additional SymPy bottlenecks:

1. **`sp.cancel(scale * Q_int)`**: Multiplying the scale factor by the
   integrated polynomial quotient triggers catastrophic multivariate GCD
   in SymPy. Fix: distribute scale into each F-coefficient individually
   (`_scale_polynomial_coeffwise`). Speedup: 591s → 0.7s for J4 Psi m=0.

2. **`sp.cancel` on rational coefficients of q, Q**: The general-purpose
   `sp.cancel` is much slower than `sp.Poly`-based GCD for expressions in
   the specific polynomial ring Q(q, Q). Fix: `_fast_cancel` uses
   `sp.Poly(q, Q)` + `sp.gcd` instead. Speedup: 2-10x per call.

3. **`_clean` on loaded cached expressions**: Calling `sp.cancel(sp.together(...))`
   on already-canonical serialized expressions is redundant and catastrophically
   slow for large expressions. Fix: skip `_clean` when loading from cache.
   Speedup: >10 min → <1s for J4 data loading.

### Final timings (one-time code generation)

| Degree | g    | Q    | Psi  | Omega | M    | Total  |
|--------|------|------|------|-------|------|--------|
| J2     | 4.6s | 2.8s | 9.1s | 3.9s  | 2.4s | ~23s   |
| J3     | <10s | <10s | <30s | <10s  | <10s | ~1min  |
| J4     | <30s | <30s | ~2m  | <30s  | <30s | ~5min  |
| J5     | <1m  | <1m  | ~30m | ~40m  | ~5m  | ~80min |

J5 generation completed in ~80 minutes of one-time computation. Psi and Omega are the
bottlenecks due to 6 w-harmonics each with high-degree (10+) polynomials in q, Q.
The computation is CPU-bound (100% single-core, ~320MB RSS) and completes cleanly.
All J2-J5 data is now generated and validated.

## Scaling to Higher Degrees

The method scales to J3-J5 because:
- The forcing is still a finite Laurent polynomial (support grows linearly with n)
- The denominator structure (F+q)^2(1+qF)^2 is independent of n
- Only the shift s and polynomial degree grow with n
- All operations (long division, evaluation, Taylor recurrence) are polynomial-time
- The canonical single-fraction output enables efficient serialization/deserialization

## Numerical Validation

### J2-only, multiple initial anomalies

| Case   | M0 [deg] | K RMS [rad] | pos RMS [km] | pos max [km] |
|--------|----------|-------------|--------------|--------------|
| low-e  | 20       | 9.52e-6     | 0.155        | 0.297        |
| low-e  | 80       | 7.76e-6     | 0.190        | 0.426        |
| low-e  | 140      | 8.94e-6     | 0.164        | 0.345        |
| low-e  | 220      | 1.58e-5     | 0.255        | 0.517        |
| low-e  | 300      | 1.80e-5     | 0.299        | 0.581        |
| high-e | 35       | 2.41e-4     | 2.905        | 4.563        |
| high-e | 90       | 1.91e-4     | 2.232        | 3.836        |
| high-e | 170      | 1.35e-4     | 1.821        | 3.139        |
| high-e | 260      | 1.83e-4     | 3.022        | 5.655        |
| high-e | 340      | 2.31e-4     | 3.815        | 6.845        |

### J2-J5, full multi-anomaly validation

| Case   | M0  | K RMS [rad] | pos RMS [km] | pos max [km] | Psi RMS [rad] | Omega RMS [rad] |
|--------|-----|-------------|--------------|--------------|---------------|-----------------|
| low-e  | 20  | 9.51e-6     | 0.155        | 0.297        | 1.22e-5       | 2.32e-5         |
| low-e  | 80  | 7.74e-6     | 0.189        | 0.426        | 1.23e-5       | 3.43e-5         |
| low-e  | 140 | 8.92e-6     | 0.164        | 0.344        | 1.21e-5       | 2.69e-5         |
| low-e  | 220 | 1.58e-5     | 0.255        | 0.517        | 1.64e-5       | 3.30e-5         |
| low-e  | 300 | 1.79e-5     | 0.299        | 0.581        | 2.67e-5       | 4.00e-5         |
| high-e | 35  | 2.41e-4     | 2.909        | 4.568        | 2.42e-4       | 2.49e-4         |
| high-e | 90  | 1.91e-4     | 2.235        | 3.840        | 1.91e-4       | 1.97e-4         |
| high-e | 170 | 1.35e-4     | 1.823        | 3.143        | 1.35e-4       | 1.39e-4         |
| high-e | 260 | 1.83e-4     | 3.025        | 5.661        | 1.83e-4       | 1.91e-4         |
| high-e | 340 | 2.32e-4     | 3.820        | 6.852        | 2.31e-4       | 2.41e-4         |

Inverse round-trip (osc -> mean -> osc): low-e 7.2 m, high-e 1.6 m Cartesian error.
J2-J5 results are indistinguishable from J2-J4, confirming that J5 contributes
negligibly to the first-order short-period corrections.

### Zonal scaling

Scaling test (J2-J5, high-e): log-log slope = 0.998. The error scales linearly
with the zonal scale factor, consistent with the O(epsilon) magnitude of the
short-period correction itself. The reconstruction error is dominated by the
first-order periodic signature, not by second-order truncation artifacts.

## Hypotheses Confirmed

**Hypothesis A (CONFIRMED)**: The blocker was SymPy expression swell from
integrating combined rational expressions. The coefficient-space approach
(combined partial fractions with structural assembly) removes it.

**Hypothesis B (CONFIRMED)**: The short-period kernels admit a finite structural
form: polynomial + simple poles at F=-q and F=-1/q + higher poles at F=0.
The operator coupling is sparse (pentadiagonal in F-powers).

**Hypothesis C (CONFIRMED)**: Psi and Omega are expensive only because of poor
symbolic realization (term-by-term ratint with log terms), not because the
exact first-order kernels are intrinsically impractical.

## Conclusion

The exact first-order mixed-zonal short-period closure for GEqOE is now
computationally viable through J5. The direct residue decomposition method,
combined with three targeted SymPy optimizations (coefficient-wise scaling,
Poly-based GCD, and skip-clean on deserialization), reduces the symbolic
computation from "blocked/infinite" to minutes of one-time code generation
(~80 min for J5). The generated expressions enable sub-km osculating
reconstruction for low-eccentricity orbits and few-km reconstruction for
high-eccentricity (e=0.65) orbits over 10+ orbital
periods, with no anomaly-dependent pathologies.
