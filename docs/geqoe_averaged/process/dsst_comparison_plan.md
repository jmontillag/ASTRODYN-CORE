# Plan: DSST Comparison for GEqOE Averaged Theory

## Goal

Add a fair, controlled comparison of the first-order GEqOE averaged theory
against Orekit's DSST semi-analytical propagator (zonal-only), both
theoretically and numerically. This fills the gap in the current document
which only compares against Brouwer–Lyddane (a purely analytical theory of
the same class as ours).

---

## Part 1 — Theoretical Comparison Section

**Location**: New Section 11 in `geqoe_averaged_zonal_theory.tex`, between
the current Extended Validation (Section 10) and the bibliography.

**Title**: *Comparison with Semi-Analytical Theory (DSST)*

### 1.1 Structural comparison table

Side-by-side table covering:

| Aspect | GEqOE Averaged (this work) | DSST (Orekit / Danielson) |
|---|---|---|
| Element set | GEqOE — non-canonical, K-based | Classical equinoctial (a, h, k, p, q, λ) |
| Averaging variable | Generalized eccentric longitude K | Mean longitude λ |
| Perturbation order — mean drift | First-order O(ε) for all zonals | First-order for all zonals + J₂² secular correction |
| Short-period corrections | Closed-form (residue decomposition) | Analytical Fourier series truncated in ecc. power (max 4) |
| Mean-element propagation | Closed-form rotation (J₂) / smooth ODE (J₂–J₅) | Numerical integration of mean-element ODE |
| Singularities | None for prograde non-rectilinear | None for non-rectilinear |
| Critical inclination | No 1−5cos²i divisor — clean by construction | Historically problematic; handled in modern implementations |
| Eccentricity truncation | Exact in eccentricity (rational functions of q) | Power series in e, truncated at maxEccPow (default ≤ 4) |
| Runtime cost per epoch | Formula evaluation — O(μs) | Quadrature + ODE step — O(ms) |
| Extensibility | New residue derivation needed per perturbation class | Modular: DSSTTesseral, DSSTThirdBody, DSSTDrag built-in |

### 1.2 Theoretical error budget

Explain **why** DSST should generally be more accurate:

1. **J₂² secular terms**: The standard DSST formulation (Danielson et al.
   1995) includes second-order J₂×J₂ secular and long-period corrections in
   the mean-element equations. Our theory is strictly first-order.
   - Order of the missing term: J₂² ≈ 10⁻⁶ rad/orbit in mean drift
   - This accumulates linearly with time → dominant over long arcs
   - For high-altitude orbits (large a), ε ∝ (Rₑ/a)² shrinks, so the
     gap should narrow

2. **Keplerian Jacobian**: Both theories use the Keplerian dt/dK (or dt/dλ)
   at first order. The correction from the perturbed Jacobian is O(ε²) and
   identical for both. DSST's numerical quadrature could in principle absorb
   this, but in practice both theories treat it the same way.

3. **Eccentricity truncation tradeoff**: DSST truncates the short-period
   Fourier series at eccentricity power ≤ 4 (default). Our theory is exact
   in eccentricity (rational functions of q = e/(1+β)). For high-e orbits
   (Molniya, GTO), our closed-form expressions may actually be more accurate
   in the short-period reconstruction than DSST's truncated series.

### 1.3 Expected accuracy hierarchy

State the prediction before showing data:

```
Cowell (truth)  ≫  DSST-osculating  >  GEqOE mean+SP  >  Brouwer–Lyddane
```

With the caveat that for high-eccentricity orbits, the eccentricity
truncation in DSST may partially close or even reverse the gap in
short-period reconstruction.

### 1.4 Cost–accuracy tradeoff framing

The comparison is not about which theory is "better" in isolation, but about
the Pareto frontier of cost vs. accuracy. Our theory trades ~1 order of
magnitude in accuracy for ~2–3 orders of magnitude in evaluation speed.
For catalog-scale conjunction screening, this is the right tradeoff.

---

## Part 2 — Numerical Comparison

### 2.1 Add DSST propagation functions to `extended_validation.py`

Use the `astrodyn_core` high-level API. Three DSST configurations:

```python
from astrodyn_core import (
    AstrodynClient, BuildContext, GravitySpec,
    IntegratorSpec, PropagatorKind, PropagatorSpec, SpacecraftSpec,
)

def _build_dsst_propagator(orbit, state_type="OSCULATING"):
    """Build a DSST (zonal-only J2–J5) propagator via astrodyn_core API."""
    app = AstrodynClient()
    dsst_spec = PropagatorSpec(
        kind=PropagatorKind.DSST,
        spacecraft=SpacecraftSpec(mass=1.0),
        integrator=IntegratorSpec(
            kind="dp853",
            min_step=1e-3,
            max_step=300.0,
            position_tolerance=1e-6,
        ),
        dsst_propagation_type="MEAN",
        dsst_state_type=state_type,
        force_specs=[GravitySpec(degree=5, order=0)],
    )
    ctx = BuildContext(initial_orbit=orbit)
    builder = app.propagation.build_builder(dsst_spec, ctx)
    return builder.buildPropagator(
        builder.getSelectedNormalizedParameters()
    )

def _propagate_dsst_grid(propagator, epoch, frame, t_grid):
    """Propagate DSST to a time grid, return positions [N, 3] in km."""
    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        state = propagator.propagate(epoch.shiftedBy(float(t)))
        pv = state.getPVCoordinates(frame)
        pos = pv.getPosition()
        positions[i] = np.array([pos.getX(), pos.getY(), pos.getZ()]) / 1e3
    return positions

def run_dsst_zonal_osc(orbit, epoch, frame, t_grid):
    """DSST zonal, osculating output — primary comparison."""
    prop = _build_dsst_propagator(orbit, state_type="OSCULATING")
    return _propagate_dsst_grid(prop, epoch, frame, t_grid)

def run_dsst_zonal_mean(orbit, epoch, frame, t_grid):
    """DSST zonal, mean-only output — isolates mean-drift accuracy."""
    prop = _build_dsst_propagator(orbit, state_type="MEAN")
    return _propagate_dsst_grid(prop, epoch, frame, t_grid)
```

For the **high-eccentricity power variant**, we need lower-level access to
set `maxEccPowShortPeriodics > 4`. The `astrodyn_core` high-level API
doesn't expose this parameter, so for the high-e cases we build the
`DSSTZonal` force model manually:

```python
def run_dsst_zonal_high_ecc(orbit, epoch, frame, t_grid, max_ecc_pow=6):
    """DSST zonal with elevated eccentricity power for high-e orbits."""
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTZonal
    from org.orekit.propagation.conversion import DSSTPropagatorBuilder
    # ... manual construction with DSSTZonal(frame, provider,
    #     maxDegreeShortPeriodics=5, maxEccPowShortPeriodics=max_ecc_pow,
    #     maxFrequencyShortPeriodics=11) ...
```

Run this variant only for Molniya (e=0.74), GTO (e=0.73), and HEO (e=0.4).
Test `maxEccPowShortPeriodics` ∈ {4 (default), 6, 8} to see if the
short-period reconstruction improves — and whether our exact closed-form
expressions still beat DSST's truncated series at high eccentricity.

**Key design choices:**
- `degree=5, order=0` → zonal-only through J₅, matching our theory
- Constants come from Orekit's gravity field provider (EGM2008). The ~0.1%
  difference vs. our hardcoded J₂–J₅ is noted in the document but accepted.
- Orekit orbit construction: reuse existing `_init_orekit()` + refactored
  shared `_build_orekit_orbit(case)` helper.

### 2.2 Add DSST to the main validation loop

In `run_single_case()`, add new blocks after the Brouwer section:

```python
# --- 5. DSST zonal (osculating) — primary comparison ---
orbit_ok, orekit_orbit, epoch, frame = _build_orekit_orbit(case)
t0 = time.time()
dsst_osc_cart = run_dsst_zonal_osc(orekit_orbit, epoch, frame, t_grid)
err_dsst_osc = compute_errors(truth_cart, dsst_osc_cart, "DSST-osc")
result["dsst_osc"] = err_dsst_osc
result["dsst_osc_time"] = time.time() - t0

# --- 6. DSST zonal (mean-only) — isolate mean-drift accuracy ---
t0 = time.time()
dsst_mean_cart = run_dsst_zonal_mean(orekit_orbit, epoch, frame, t_grid)
err_dsst_mean = compute_errors(truth_cart, dsst_mean_cart, "DSST-mean")
result["dsst_mean"] = err_dsst_mean

# --- 7. DSST high-ecc-power (only for high-e cases) ---
if case.e >= 0.35:
    for max_ecc in [6, 8]:
        dsst_he = run_dsst_zonal_high_ecc(
            orekit_orbit, epoch, frame, t_grid, max_ecc_pow=max_ecc)
        result[f"dsst_osc_ecc{max_ecc}"] = compute_errors(
            truth_cart, dsst_he, f"DSST-osc-ecc{max_ecc}")
```

Also need a **GEqOE mean-only propagation** (no short-period map) for the
mean-drift comparison against DSST mean-only. This is just `mean_hist`
converted to Cartesian without applying `mean_to_osculating_state` — the
mean GEqOE elements converted directly via `geqoe2cart`. Store as
`result["geqoe_mean_only"]`.

Refactor the Orekit orbit construction (currently inside `run_brouwer`) into
a shared helper `_build_orekit_orbit(case)` that `run_brouwer`,
`run_dsst_zonal_osc`, and `run_dsst_zonal_mean` all call.

### 2.3 Run across the same 12-case test matrix

No changes to `CASES` — use the existing 12 orbital regimes.

### 2.4 Extend output

**LaTeX tables**:

Table A (primary) — osculating comparison against Cowell truth:
```
Case | a | e | i | Days | GEqOE mean+SP (pos/rad) | DSST osc (pos/rad) | Brouwer (pos/rad)
```

Table B (mean-drift decomposition) — mean-only comparison:
```
Case | GEqOE mean-only pos RMS | DSST mean-only pos RMS | Ratio
```
This isolates the secular/long-period accuracy gap (J₂² terms) from
short-period reconstruction differences.

Table C (high-e eccentricity power study) — only Molniya/GTO/HEO cases:
```
Case | GEqOE mean+SP | DSST ecc=4 | DSST ecc=6 | DSST ecc=8
```
Shows whether increasing eccentricity power improves DSST's high-e accuracy
and whether our closed-form expressions have an advantage.

**Figures** (update existing + add new):

1. **Updated bar chart** (`extended_validation_comparison_bar.png`):
   Three-theory side-by-side bars per case (GEqOE, DSST, Brouwer).

2. **Updated position error time series**
   (`extended_validation_pos_errors.png`): Add DSST curve (dotted) alongside
   GEqOE (solid) and Brouwer (dashed).

3. **New: cost–accuracy scatter** (`cost_accuracy_scatter.png`):
   Log-log plot with wall-clock time (x) vs. position RMS (y). One point per
   (case, theory) combination. Visualizes the Pareto frontier.

4. **New: mean-drift comparison** (`mean_drift_comparison.png`):
   Position error time series for selected cases showing GEqOE mean-only vs
   DSST mean-only. This makes the J₂² secular drift gap visually clear.

5. **New: eccentricity power convergence** (`ecc_power_convergence.png`):
   For the 3 high-e cases, bar chart of DSST accuracy vs. maxEccPow,
   with GEqOE horizontal reference line.

### 2.5 Timing

Add `time.time()` measurements around each propagator call (already done for
Cowell/GEqOE/Brouwer). Report wall-clock time per case in a separate table
or as annotations on the scatter plot.

---

## Part 3 — Analysis for the Document

### 3.1 Where DSST should win (large ε, long arcs)

- **LEO orbits**: ε ∝ J₂(Rₑ/a)² is largest → J₂² secular drift dominates
  over long arcs. Expect DSST to show noticeably smaller error growth.
- **Critical inclination**: DSST's J₂² long-period terms better capture the
  near-resonant ω dynamics. Though our theory handles critical inclination
  without singularity, the second-order secular accuracy should still favor
  DSST.
- **Long propagation windows** (50–100 orbits): The O(J₂²) drift
  accumulates, so the gap widens with time.

### 3.2 Where the gap should be small

- **MEO/GPS** (a = 26560 km): ε² ∝ (Rₑ/a)⁴ ≈ 10⁻⁸, so J₂² correction
  is negligible. Expect near-parity or even GEqOE advantage if DSST's
  eccentricity truncation matters.
- **High-altitude, short arcs**: Both theories converge to similar accuracy
  when the perturbation is small.

### 3.3 Where GEqOE might win (high eccentricity)

- **Molniya / GTO** (e = 0.73–0.74): DSST truncates its short-period
  Fourier series at eccentricity power ≤ 4. Our closed-form rational
  expressions are exact in eccentricity. The short-period reconstruction
  error for DSST may be larger than ours in these regimes, partially or
  fully offsetting its mean-drift advantage.
- This would be a notable result if confirmed — it demonstrates a concrete
  advantage of the closed-form approach.

### 3.4 Cost–accuracy narrative

Frame the results as a Pareto frontier:
- Brouwer–Lyddane: fast but least accurate
- GEqOE mean+SP: very fast, significantly more accurate than Brouwer
- DSST: slower (ms vs. μs per epoch), more accurate (J₂² secular)
- Cowell: much slower, exact

The GEqOE theory occupies a useful niche: Brouwer-class speed with accuracy
approaching DSST, making it suitable for catalog-scale applications where
DSST's per-epoch cost is prohibitive.

---

## Implementation Steps

| # | Task | Files | Depends on |
|---|------|-------|------------|
| 1 | Refactor Orekit orbit construction into shared helper | `extended_validation.py` | — |
| 2 | Implement `run_dsst_zonal()` with osculating + mean output modes | `extended_validation.py` | 1 |
| 3 | Implement high-ecc-power DSST variant for Molniya/GTO cases | `extended_validation.py` | 2 |
| 4 | Add all DSST variants to `run_single_case()` loop | `extended_validation.py` | 2, 3 |
| 5 | Extend `write_latex_table()` with DSST columns | `extended_validation.py` | 4 |
| 6 | Update figure generation (bar chart, time series) with DSST | `extended_validation.py` | 4 |
| 7 | Add cost–accuracy scatter plot | `extended_validation.py` | 6 |
| 8 | Run full 12-case comparison, collect results | — | 7 |
| 9 | Write theoretical comparison section in .tex | `geqoe_averaged_zonal_theory.tex` | 8 (uses results) |
| 10 | Update abstract and conclusion to mention DSST comparison | `geqoe_averaged_zonal_theory.tex` | 9 |

---

## Resolved Decisions

1. **Constant matching** → **Accept mismatch.** Use `astrodyn_core`
   high-level API with Orekit's gravity field provider. Note the ~0.1%
   J₂–J₅ constant difference in the document. Only build a custom provider
   if results are ambiguous.

2. **DSST mean-only mode** → **Yes, run both modes.** Two DSST
   configurations per case:
   - `dsst_state_type="OSCULATING"` — full osculating output for direct
     comparison against GEqOE mean+SP and Brouwer (the primary comparison)
   - `dsst_state_type="MEAN"` — mean-only output, compared against GEqOE
     mean propagation (no short-period map applied) to isolate mean-drift
     accuracy from short-period reconstruction accuracy.
   This decomposition will clearly show whether the gap comes from the J₂²
   secular terms (mean drift) or from short-period differences.

3. **DSST eccentricity power** → **Yes, test higher values.** For the
   high-eccentricity cases (Molniya e=0.74, GTO e=0.73, HEO e=0.4), also
   run DSST with `maxEccPowShortPeriodics` increased beyond the default 4
   (try 6 or 8 if computationally feasible). This isolates the eccentricity
   truncation effect and tests whether our closed-form rational expressions
   have an advantage in the high-e regime.
