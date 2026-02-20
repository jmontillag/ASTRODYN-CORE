# Architecture Snapshot (Phase 1+)

Last updated: 2026-02-20

The repository started as a propagation-only Phase 1 core and now includes state-file, mission, uncertainty, and façade-governed API lanes.

## 1) Propagation lane (stable)

Core flow:

1. User defines `PropagatorSpec`.
2. `PropagatorFactory` routes by `PropagatorKind` via `ProviderRegistry`.
3. Provider returns Orekit-native objects (`PropagatorBuilder` / `Propagator`).

Main modules:

- `astrodyn_core.propagation.specs`
- `astrodyn_core.propagation.interfaces`
- `astrodyn_core.propagation.registry`
- `astrodyn_core.propagation.factory`
- `astrodyn_core.propagation.providers`
- `astrodyn_core.propagation.assembly` (force/attitude/spacecraft assembly)

Default provider mapping:

- `numerical` -> `NumericalPropagatorBuilder`
- `keplerian` -> `KeplerianPropagatorBuilder`
- `dsst` -> `DSSTPropagatorBuilder`
- `tle` -> `TLEPropagatorBuilder` or `TLEPropagator`

## 2) State I/O lane (stable)

Purpose:

- Serialize/deserialize mission state data independently from Orekit class instances.
- Convert file records to Orekit objects at runtime.

Main modules:

- `astrodyn_core.states.models`
- `astrodyn_core.states.io`
- `astrodyn_core.states.orekit`
- `astrodyn_core.states.validation`
- `astrodyn_core.states.client`

Formats:

- YAML/JSON (scenario + compact series schema)
- HDF5 (compressed columnar state-series storage)

## 3) Mission helper lane (stable)

Purpose:

- Translate scenario timeline/maneuver definitions into executable propagation workflows.

Main modules:

- `astrodyn_core.mission.maneuvers` (compatibility façade)
- `astrodyn_core.mission.models`
- `astrodyn_core.mission.timeline`
- `astrodyn_core.mission.intents`
- `astrodyn_core.mission.kinematics`
- `astrodyn_core.mission.simulation`
- `astrodyn_core.mission.plotting`

Current behavior:

- timeline/event-driven maneuver scheduling
- fast Keplerian intent solving
- impulse application during propagation replay
- detector-driven closed-loop execution (`ScenarioExecutor`)
- orbital-element plot export

## 4) Uncertainty lane (stable)

Main modules:

- `astrodyn_core.uncertainty.propagator` (compatibility façade)
- `astrodyn_core.uncertainty.matrix_io`
- `astrodyn_core.uncertainty.transforms`
- `astrodyn_core.uncertainty.records`
- `astrodyn_core.uncertainty.stm`
- `astrodyn_core.uncertainty.factory`

Current behavior:

- STM covariance propagation with YAML/HDF5 persistence
- Unscented path scaffold retained as planned future work

## 5) Public API strategy

- Keep common symbols at package root (`astrodyn_core`).
- Expose an app-level unified façade (`AstrodynClient`) for most users.
- Keep domain façades available for focused workflows (`PropagationClient`, `StateFileClient`, `MissionClient`, `UncertaintyClient`, `TLEClient`).
- Preserve direct access to Orekit-native builders/propagators for advanced users.

## 6) Next architectural step

Complete Phase C API governance hardening: maintain stable façade-first user paths, enforce private-import boundaries, and document deprecation/stability policy.
