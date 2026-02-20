# State I/O and Scenario Design

Last updated: 2026-02-20

This document describes the implemented state-file architecture and the planned extensions.

## 1) Goals

- Define mission/scenario inputs in files (not hardcoded scripts).
- Keep serialization independent from Orekit runtime classes.
- Support:
  - single state
  - state time series
  - scenario timeline and maneuver directives.
- Keep a single high-level API entrypoint for users.

## 2) Implemented Architecture

## Modules

- `src/astrodyn_core/states/models.py`
  - typed dataclasses (`OrbitStateRecord`, `StateSeries`, `ScenarioStateFile`, timeline/maneuver records)
- `src/astrodyn_core/states/io.py`
  - YAML/JSON/HDF5 read/write
- `src/astrodyn_core/states/orekit.py`
  - compatibility façade for Orekit conversion and trajectory helpers
- `src/astrodyn_core/states/orekit_dates.py`, `orekit_resolvers.py`, `orekit_convert.py`, `orekit_ephemeris.py`, `orekit_export.py`
  - decomposed Orekit helper layers (Phase B)
- `src/astrodyn_core/states/validation.py`
  - schema/date parsing validation helpers
- `src/astrodyn_core/states/client.py`
  - `StateFileClient` facade for end-to-end workflows

## User entrypoint

```python
from astrodyn_core import StateFileClient

client = StateFileClient()
```

This class centralizes loading/saving, Orekit conversions, trajectory export, scenario export, and plotting hooks.

## 3) Schema Coverage (Current)

## Core records

- `OrbitStateRecord`
  - `epoch`, `frame`, `representation`, orbit values, `mu_m3_s2`, optional `mass_kg`
- `StateSeries`
  - ordered states with optional interpolation metadata
- `ManeuverRecord`
  - trigger + model mapping (impulsive and intent forms)
- `AttitudeRecord`
  - stored for timeline/schema completeness (execution support is still limited)
- `TimelineEventRecord`
  - named event references used by maneuvers
- `ScenarioStateFile`
  - top-level scenario container

## Timeline support currently implemented

Timeline `point.type` supports:

- `epoch`
- `elapsed` (relative to another event)
- `apogee`
- `perigee`
- `ascending_node`
- `descending_node`

Maneuver triggers can directly reference timeline events (`trigger.type: event`).

## 4) State-Series Formats

## YAML/JSON compact format

Implemented compact schema:

- `defaults` block for invariant series fields
- `columns` list
- `rows` matrix

This avoids repeating full keys for every sample while staying readable for onboarding.

## HDF5 format

Implemented with `h5py`:

- columnar datasets
- compressed numeric arrays
- schema/interpolation/defaults metadata in file attributes

This is the preferred path for large trajectory datasets.

## 5) Orekit Integration

Implemented:

- state record -> Orekit `Orbit`
- state series -> Orekit ephemeris
- scenario series export from propagator
- scenario maneuver compilation and mission-series export hooks

Date conversion policy:

- use Orekit wrapper helpers (`orekit.pyhelpers`) in the conversion layer
- keep file schema dates in UTC ISO-8601 strings.

## 6) Mission Execution Scope (Current vs Next)

## Current

- Fast Keplerian approximation for intent maneuver solving.
- Scenario export path that applies compiled impulses through propagation replay.
- Detector-driven maneuver execution integrated with numerical propagation (`ScenarioExecutor`).
- Good for rapid mission design iteration and file-generation workflows.

## Next

- Trigger recurrence/window semantics expansion in schema.
- Continued execution reporting/traceability refinements.
- API-governance hardening so façade-tier workflows remain the default user path.

## 7) Near-Term Change Plan

1. Extend scenario schema with recurrence/window constraints.
2. Add/expand integration tests for recurring closed-loop maintenance scenarios.
3. Document stability/deprecation expectations for state-file and Orekit conversion APIs.
4. Document recommended file choices:
   - YAML compact for learning/debugging
   - HDF5 for large production trajectories.

## 8) Example Files

Reference scenarios in `examples/state_files/`:

- `leo_initial_state.yaml`
- `leo_state_series.yaml`
- `leo_mission_timeline.yaml`
- `leo_intent_mission.yaml`
- `leo_sma_maintenance_timeline.yaml`
- `generated_scenario_enriched.yaml`
