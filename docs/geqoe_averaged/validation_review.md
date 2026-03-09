# Validation Review: Averaged J2 GEqOE Formulation

**Date:** 2026-03-09  
**Reviewed file:** `docs/geqoe_averaged/geqoe_averaged_j2_zonal_note.tex` (current post-`edc4c05` state)  
**Validation script:** `docs/geqoe_averaged/scripts/validation_check.py`

## Summary

The core formulation is sound. The first-order secular closure is correct, the
inverse-map strategy works, and the numerical results are consistent across
multiple orbit regimes and initial anomalies. The previously identified
`\Psi`-map algebra issue is already fixed in the current note and script, so
there is no live blocking correctness bug in the present averaged `J2`
formulation.

## Findings (by severity)

### F1. Closed-form Brouwer rates are exact at first order; quadrature gap is second-order

**Severity: INFORMATIONAL**

The linearization `h ≈ c`, `d ≈ -U/c` used to derive the closed-form Brouwer
rates is exact at first order in `J2`. Corrections from using exact `h` enter
the RHS at `O(J2^2)`. Comparing the analytical Brouwer rates against the
numerical `K`-average of the exact frozen-state integrand gives relative gaps of
order `1.6e-4` to `1.8e-4`, which is consistent with omitted second-order
terms.

Reported values from `CHECK 2`:

```text
<p1_dot>: rel err = 1.60e-04
<p2_dot>: rel err = 1.60e-04
<q1_dot>: rel err = 1.84e-04
<q2_dot>: rel err = 1.84e-04
<Psi_dot>: rel err = 1.60e-04
<Omega_dot>: rel err = 1.84e-04
```

The averaged magnitude rates vanish to machine precision:

```text
<g_dot> = 3.18e-26
<Q_dot> = 3.57e-18
```

This supports the current interpretation: `g` and `Q` are exact secular
constants at first order, and the remaining rate mismatch is second-order.

### F2. J2 scaling: residuals are broadly O(J2^2) but not perfectly clean

**Severity: LOW**

The `J2` scaling study indicates that the full-map residuals are consistent with
first-order theory. For `J2` scale factors `>= 0.5`, the component RMS scales
approximately as `J2^2`. At small `J2`, orbit-mean quadrature noise becomes
visible and spoils a clean asymptotic ratio.

Reported values from `CHECK 4`:

```text
scale=0.10  RMS=2.18e-07   actual/J2^2 ratio: 4.52
scale=0.25  RMS=5.74e-07   actual/J2^2 ratio: 1.90
scale=0.50  RMS=1.54e-06   actual/J2^2 ratio: 1.28
scale=1.00  RMS=4.83e-06   actual/J2^2 ratio: 1.00
scale=2.00  RMS=1.69e-05   actual/J2^2 ratio: 0.88
scale=5.00  RMS=9.90e-05   actual/J2^2 ratio: 0.82
```

Interpretation: no evidence of a structural error in the averaged formulation;
the residuals are compatible with omitted `O(J2^2)` terms.

### F3. Near-circular orbit conditioning affects the angular diagnostics

**Severity: INFORMATIONAL**

For the low-eccentricity LEO test case, the component RMS remains good while the
`\Psi` RMS appears noticeably larger:

```text
LEO (e=0.01, i=51.6 deg):
  comp RMS = 1.07e-05
  Psi RMS  = 5.06e-04
  Om RMS   = 6.00e-05
```

This is a conditioning issue, not a formulation bug: when `g` is small,
extracting `\Psi = atan2(p1, p2)` amplifies small component errors by roughly
`1/g`. The note should mention this if near-circular use cases are expected to
matter.

### F4. Averaging-variable notation is fine at first order but deserves a caveat

**Severity: INFORMATIONAL**

The note defines the orbit average as a time average written in `K`. In the
first-order frozen-state derivation, switching between `K` and `G = K - \Psi`
is immaterial because `dK = dG` at that order. That said, the distinction will
matter for any future second-order extension, so a brief warning in the note is
appropriate.

### F5. Earlier Psi-map algebra bug is already resolved in the current note

**Severity: RESOLVED**

The prior review pass flagged a missing factor of `1/g` in the `dPsi/dG`
equation. That was a real issue during the intermediate derivation stage, but it
is already corrected in the current note and the numerical script.

Current note:

```text
dPsi/dG = (r/w) [ d - w_h - (U/c)(1 + cos G / g) ]
```

Current script:

```text
psi_dot = (p2 * p1_dot - p1 * p2_dot) / g^2
```

So there is no outstanding live bug here.

## Checks Passed

### Multi-anomaly validation (`CHECK 3`)

Tested `M0 = {0, 20, 45, 90, 135} deg` on the Molniya reference orbit. The full
inverse map consistently improves the secular model match against per-orbit
means by about `12x` to `47x`:

```text
M0=  0°  naive RMS=5.74e-05  full RMS=4.85e-06  (12x)
M0= 20°  naive RMS=2.26e-04  full RMS=4.82e-06  (47x)
M0= 45°  naive RMS=1.78e-04  full RMS=4.83e-06  (37x)
M0= 90°  naive RMS=1.24e-04  full RMS=4.86e-06  (25x)
M0=135°  naive RMS=6.15e-05  full RMS=4.89e-06  (13x)
```

As expected, the improvement is strongest away from symmetry points where the
short-period correction is largest.

### Multi-regime validation (`CHECK 3b`)

```text
LEO (e=0.01, i=51.6°)      comp RMS=1.07e-05
MEO GPS (e=0.01, i=55°)    comp RMS=1.04e-07
Molniya (e=0.74, i=63.4°)  comp RMS=4.83e-06
GTO (e=0.73, i=28.5°)      comp RMS=7.89e-06
```

All tested regimes show the secular model plus inverse map tracking per-orbit
means to better than about `2e-5` in component RMS.

### Angle/magnitude meanings (`CHECK 1`)

```text
g ≈ e
Q = tan(i/2)
Psi ≈ omega + RAAN
Omega = RAAN
```

The small `g` and `\Psi` offsets are `O(J2)` and consistent with the use of the
generalized GEqOE variables rather than pure Keplerian ones.

### Code-path consistency (`CHECK 6`)

The `J2` fast path and the zonal path give identical `w_h` values for the same
`J2` forcing, which supports the implementation consistency of the current RHS.

## Commands Run

```bash
conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/validation_check.py
```

## Residual Risks / Open Checks

1. **Second-order drift horizon**: the `~1.6e-4` relative gap in secular rates
   means the first-order model will accumulate drift on long arcs. The practical
   time horizon for maneuver-detection use is still unquantified.
2. **Critical-inclination and extreme-inclination coverage**: the current checks
   are encouraging near Molniya conditions, but no dedicated tests were run at
   the exact critical inclination or near equatorial / retrograde singular
   limits.
3. **General-zonal extension**: the present validation is strictly `J2`. The
   quadrature machinery is plausible for higher zonals, but unvalidated.
4. **Mean fast-phase reconstruction**: the slow-state formulation is validated,
   but full `K` reconstruction still deserves its own targeted check before it
   is used operationally.
