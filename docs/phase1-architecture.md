# Architecture Snapshot (Phase 1+)

Last updated: 2026-02-19

The repository started as a propagation-only Phase 1 core and now includes state-file and mission-profile lanes.

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

## 3) Mission helper lane (in progress)

Purpose:

- Translate scenario timeline/maneuver definitions into executable propagation workflows.

Main modules:

- `astrodyn_core.mission.maneuvers`
- `astrodyn_core.mission.plotting`

Current behavior:

- timeline/event-driven maneuver scheduling
- fast Keplerian intent solving
- impulse application during propagation replay
- orbital-element plot export

## 4) Public API strategy

- Keep common symbols at package root (`astrodyn_core`).
- Expose a single user-friendly state/mission facade (`StateFileClient`).
- Preserve direct access to Orekit-native builders/propagators for advanced users.

## 5) Next architectural step

Add a detector-driven mission execution path that binds scenario triggers to Orekit event detectors for closed-loop maneuver execution in numerical propagation.
