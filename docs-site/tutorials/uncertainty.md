# Tutorial: Uncertainty Workflows (Covariance + STM)

This tutorial walks through the covariance/STM workflow demonstrated in
`examples/uncertainty.py`.

It is an advanced example, but it is one of the best learning tools in the repo
because it shows not only *how* to run covariance propagation, but also how to
validate the results.

## Learning goals

By the end of this tutorial, you should understand how to:

- propagate a covariance with the STM method
- compare Cartesian and Keplerian covariance representations
- extract raw STM matrices without an initial covariance
- validate covariance results (PSD checks, representation consistency, I/O round-trip)
- save uncertainty outputs to YAML (and optionally HDF5)

## Source script (executable truth)

- `examples/uncertainty.py`

Run it from the repo root:

```bash
conda run -n astrodyn-core-env python examples/uncertainty.py
```

## Before you run it

Prerequisites:

- project installed in `astrodyn-core-env`
- `orekit` available in that environment
- `numpy` available (installed via project dependencies)
- optional: `h5py` if you want HDF5 covariance output

This script writes outputs under:

- `examples/generated/`

## Mental model (what this example teaches)

The script is structured as a sequence of increasingly strict checks:

1. **Propagate Cartesian covariance**
2. **Propagate equivalent Keplerian covariance**
3. **Extract raw STM (Φ)**
4. **Check representation consistency**
5. **Check PSD validity**
6. **Check YAML round-trip serialization**

That makes it both a tutorial and a validation harness.

## Inputs and setup

The script defines:

- a Keplerian `OrbitStateRecord` (`INITIAL_STATE`)
- a Cartesian 6×6 covariance (`INITIAL_COV_CART`)
- a 3-day `OutputEpochSpec` sampled every 30 minutes
- a helper `_build_propagator()` that returns a fresh numerical propagator

Why the "fresh propagator" helper matters:

- numerical propagators and STM-enabled wrappers may carry internal state
- the script intentionally builds a new propagator per workflow to avoid
  cross-contamination between runs

## Part 1: Cartesian covariance propagation

The first run propagates a Cartesian covariance using:

- `UncertaintySpec(method="stm", orbit_type="CARTESIAN")`
- `app.uncertainty.propagate_with_covariance(...)`

Outputs:

- `examples/generated/cov_cart_trajectory.yaml`
- `examples/generated/cov_cart_series.yaml`

What to look for in the printed output:

- 1σ position and velocity growth over time
- growth factors over 12-hour intervals

Interpretation:

- this shows how uncertainty grows under the chosen dynamics model
- the script uses a numerical propagator with a low-order gravity model (J2 2x2)

## Part 2: Keplerian covariance propagation

The script then converts the same physical initial covariance from Cartesian to
Keplerian coordinates and runs a second propagation:

- `orbit_type="KEPLERIAN"`
- `position_angle="MEAN"`

Outputs:

- `examples/generated/cov_kep_trajectory.yaml`
- `examples/generated/cov_kep_series.yaml`
- `examples/generated/cov_kep_series.h5` (optional; only if `h5py` is installed)

What to look for:

- `σ_a` (semimajor axis uncertainty) often stays comparatively stable
- `σ_M` (mean anomaly uncertainty) tends to grow, capturing along-track divergence

This is a core reason Keplerian covariance is useful:

- it often aligns more directly with how orbital uncertainty is discussed in OD
  and mission analysis workflows

## Part 3: STM-only extraction (no covariance required)

The script uses:

- `setup_stm_propagator(...)`
- `propagate_with_stm(epoch)`

to obtain raw STM matrices `Φ(t, t0)` at selected epochs.

This is useful when you want:

- direct access to the state transition matrix
- custom covariance propagation logic
- diagnostics/conditioning analysis independent of the higher-level client flow

The script prints:

- `|det(Φ) - 1|`
- Frobenius norm `||Φ||_F`
- condition number
- maximum absolute entry

These are practical diagnostics for understanding numerical behavior over time.

## Part 4: Representation consistency check

This is the most instructive part of the script.

At daily checkpoints it compares:

- covariance propagated directly in Cartesian space
- covariance propagated in Keplerian space then converted back to Cartesian
- covariance reconstructed manually as `Φ P0 Φᵀ`

What the script demonstrates:

- Keplerian-vs-Cartesian representation changes are not introducing meaningful
  extra error beyond numerical integration path differences

If you are developing your own uncertainty workflows, this is the pattern to
copy for trust-building and regression checks.

## Part 5: PSD verification

Covariance matrices should be positive semi-definite (PSD).

The script checks eigenvalues of every covariance record in both series and
reports any violations.

Why this matters:

- a covariance that is not PSD can break downstream estimation/analysis code
- PSD checks are a cheap, high-value validation step for saved outputs

## Part 6: YAML round-trip verification

The script reloads the saved covariance series and compares matrix arrays with
the in-memory results.

This verifies:

- serialization format integrity
- I/O helpers in the `uncertainty` client/module
- no accidental precision loss beyond the expected tolerance

## Output files (what they are for)

- `cov_cart_trajectory.yaml`: state series used alongside Cartesian covariance run
- `cov_cart_series.yaml`: Cartesian covariance records over time
- `cov_kep_trajectory.yaml`: state series used alongside Keplerian covariance run
- `cov_kep_series.yaml`: Keplerian covariance records over time
- `cov_kep_series.h5`: optional HDF5 form of Keplerian covariance series

## Common pitfalls and interpretation tips

- **Different representations describe the same physical uncertainty**
  The numbers do not look similar component-by-component, especially between
  Cartesian and Keplerian forms. Compare them only after converting to the same
  representation.

- **Fresh propagators matter**
  Reusing the exact same propagator object across multiple runs can complicate
  reproducibility and interpretation in advanced workflows.

- **PSD checks are not optional in production**
  Keep the PSD verification pattern when adapting this example to your own data.

- **HDF5 output may be skipped**
  If `h5py` is not installed, the script still runs and prints a note.

## Where to go next

After running this tutorial:

1. Adapt `INITIAL_STATE` and `INITIAL_COV_CART` to your mission case
2. Reduce the time span to create a faster development loop
3. Add your own representation-consistency checkpoints
4. Export results and compare with downstream mission/estimation tooling

## Related docs

- [Propagation Quickstart](propagation-quickstart.md)
- [Scenario + Mission Workflows](scenario-missions.md)
- [Common Setup Issues](../troubleshooting/common-setup-issues.md)
- [API: `astrodyn_core`](../reference/api/astrodyn_core.md)
