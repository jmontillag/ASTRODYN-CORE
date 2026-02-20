# `uncertainty` module

Covariance propagation and covariance-series persistence.

## Purpose

The `uncertainty` module propagates uncertainty alongside a nominal trajectory.

Current main capability:
- **STM covariance propagation** via Orekit matrix harvester APIs.

Planned/partial capability:
- **Unscented Transform** scaffolding (`method='unscented'` path exists but is not the primary production flow yet).

## Core components

- `spec.py`: `UncertaintySpec` (method, state dimension, orbit type, angle convention).
- `propagator.py`: `STMCovariancePropagator` and factory helpers.
- `models.py`: `CovarianceRecord` / `CovarianceSeries` typed containers.
- `io.py`: YAML/HDF5 save/load with extension-based auto-dispatch.

## Internal flow (STM)

1. Validate/normalize `UncertaintySpec`.
2. Configure propagator in Cartesian integration basis for STM consistency.
3. Initialize matrix harvesting (`setupMatricesComputation`).
4. Propagate to requested epochs.
5. Convert/store covariance records (6x6 or 7x7 with mass).
6. Optionally map to requested orbit/angle representations for output.

## Intended use cases

1. **Trajectory uncertainty envelopes**
   - produce covariance time histories for mission analysis.
2. **Persistence and handoff**
   - write covariance products to YAML/HDF5 for downstream tooling.
3. **Integrated state + covariance workflows**
   - use through `StateFileClient.propagate_with_covariance(...)`.

See `examples/uncertainty.py` and generated files in `examples/generated/`.

## Boundaries

- Requires an already-configured Orekit propagator (typically numerical).
- Does not replace mission event logic; combine with `mission` when needed.
