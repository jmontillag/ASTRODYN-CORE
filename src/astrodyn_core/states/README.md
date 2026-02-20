# `states` module

State-file schemas, validation, I/O, and Orekit conversion wrappers.

## Purpose

The `states` module defines a typed contract for mission/scenario files and provides serialization + conversion helpers.

Main layers:
- `models.py`: typed dataclasses (`OrbitStateRecord`, `StateSeries`, `ScenarioStateFile`, etc.).
- `validation.py`: normalization and strict validation of epochs, frames, representations, and elements.
- `io.py`: YAML/JSON/HDF5 load/save utilities.
- `orekit.py`: compatibility façade for Orekit conversion helpers.
- `orekit_dates.py`, `orekit_resolvers.py`, `orekit_convert.py`, `orekit_ephemeris.py`, `orekit_export.py`: decomposed Orekit helper layers.
- `client.py`: high-level façade (`StateFileClient`) used by examples and most user code.

## Key model concepts

- **`OrbitStateRecord`**: one state at one epoch.
- **`StateSeries`**: ordered sequence of state records plus interpolation metadata.
- **`ScenarioStateFile`**: top-level container for universe/spacecraft/initial state/timeline/maneuvers/series.
- **`OutputEpochSpec`**: explicit or generated output epoch sets.

## `StateFileClient` role

`StateFileClient` is the integration hub that:
- loads/saves state/scenario files,
- exports trajectories from propagators,
- converts series/scenarios into Orekit ephemerides,
- bridges to mission and uncertainty workflows.

## Intended use cases

1. **Scenario authoring + validation**
   - maintain reproducible YAML mission inputs.
2. **Trajectory export pipelines**
   - export to compact YAML or HDF5.
3. **Replay/interpolation**
   - convert serialized trajectories back into Orekit ephemerides.
4. **Unified API for mission + uncertainty features**
   - call mission and covariance workflows through one client.

See `examples/scenario_missions.py` (`io`, `inspect`, `intent`, `detector`) and `examples/quickstart.py` (`plot`).

## Boundaries

- State schemas and persistence are here.
- Dynamics construction is in `propagation`.
- Maneuver logic is in `mission`.
