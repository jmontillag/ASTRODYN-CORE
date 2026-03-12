# General Zonal Short-Period Plan

## Current Status

The folder now contains a validated first-order averaged GEqOE **slow** model
for truncated zonal sets `J2 + ... + JN`.

What is already solid:

- `J2` mean-state initialization and forward reconstruction were validated
  against the full osculating GEqOE flow.
- The truncated zonal slow drift was derived in exact symbolic form and checked
  against numerical one-revolution averages and long-arc orbit means.
- The remaining missing piece for a complete averaged formulation is the
  **general zonal short-period map**, especially for the fast phase and full
  osculating-state reconstruction.

## Goal

Derive and validate the first-order near-identity zonal map

`z_osc = z_bar + eps * u1(z_bar, K_bar)`

for the mixed truncated zonal problem, with explicit formulas or exact symbolic
generators for:

- `g, Q, Psi, Omega`
- the fast phase `K` or an equivalent mean longitude variable
- inverse initialization `osc -> mean`
- forward reconstruction `mean -> osc`

The end state should support direct comparison against full osculating GEqOE
and Cartesian propagation, not only against per-orbit means.

## Why This Step Is Nontrivial

- The slow zonal model already closes in finite harmonics of
  `omega = Psi - Omega`, but the short-period map depends on the fast anomaly
  and must remain gauge-consistent with the chosen averaged variables.
- Mixed odd/even zonals introduce different harmonic families in the slow
  angles, so the short-period generator must preserve the same parity
  structure.
- The `J2` case showed that small coordinate mistakes in the fast-phase map can
  dominate Cartesian reconstruction error even when the slow drift is correct.
- Validation must distinguish between:
  - slow-flow agreement with orbit means,
  - osculating GEqOE reconstruction,
  - Cartesian reconstruction accuracy over long arcs.

## Proposed Work Sequence

### 1. Formalize the mixed-zonal near-identity map

- Write the first-order homological equations directly in GEqOE variables for a
  truncated zonal potential.
- Fix the averaging variable and gauge explicitly so the fast-phase correction
  is unambiguous.
- Decide whether the fast variable should be handled as `K`, `L`, or an
  auxiliary mean-longitude variable with a clean inverse map.

### 2. Derive exact degree-wise short-period kernels

- Reuse the current residue framework used for the slow drift.
- For each isolated degree `n`, derive the periodic correction kernels for
  `g, Q, Psi, Omega, K`.
- Identify the exact harmonic support in `(K, omega)` and whether the highest
  surviving short-period harmonic also truncates degree-wise.

### 3. Build the truncated mixed-zonal generator

- Sum the isolated-degree kernels into an exact truncated `J2 + ... + JN`
  short-period map.
- Expose numerical evaluators from the symbolic generator so the map can be
  applied consistently in validation scripts and future code.

### 4. Validate inverse initialization

- Starting from full osculating GEqOE states, apply the numerical inverse map
  and propagate the mean system.
- Check that the resulting mean trajectory matches per-orbit means across:
  - multiple initial anomalies,
  - low- and high-eccentricity cases,
  - several zonal truncations,
  - zonal scale-factor sweeps.

### 5. Validate forward osculating reconstruction

- Reconstruct the full osculating GEqOE state from the propagated mean state.
- Compare directly against full numerical propagation for:
  - GEqOE component errors,
  - fast-phase errors,
  - Cartesian position/velocity errors.
- Confirm the residuals scale as first-order theory predicts.

## Deliverables

- An updated derivation note covering the mixed-zonal short-period map.
- Exact symbolic generators for the degree-wise and truncated short-period
  coefficients.
- A validation script focused on full osculating reconstruction.
- Regression tests that lock the first-order structure and reconstruction
  accuracy.

## Validation Targets

The next stage should be considered successful only if all of the following are
true:

- the inverse map improves mean initialization consistently across orbit
  families and initial anomalies;
- the forward map reconstructs the fast phase without the kind of hidden gauge
  error previously found in the `J2` case;
- Cartesian reconstruction errors stay small and scale correctly under zonal
  strength rescaling;
- the symbolic short-period generator agrees with direct numerical quadrature of
  the frozen-state exact zonal RHS.

## Likely Files To Touch Next

- `docs/geqoe_averaged/scripts/zonal_symbolic_general.py`
- `docs/geqoe_averaged/scripts/zonal_mean_validation.py`
- `docs/geqoe_averaged/geqoe_averaged_j2_zonal_note.tex`
- a new mixed-zonal short-period validation note and script under
  `docs/geqoe_averaged/`
