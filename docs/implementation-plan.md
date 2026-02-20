# ASTRODYN-CORE Implementation Plan

Last updated: 2026-02-20 (rev 3)

This document tracks current implementation status and the forward plan for propagation, state I/O, and mission-profile capabilities.

## 1) Product Direction

- Keep Orekit-native builder/propagator access first-class.
- Keep declarative specs as orchestration and validation layers, not as a replacement for Orekit APIs.
- Provide a practical mission workflow from:
  - scenario/state files
  - propagation
  - trajectory export
  - basic analysis plots.

## 2) Current Stage

The project is past Phase 1/2 feature delivery and is now in API-governance hardening (Phase C) after a full Phase B module decomposition.

### Completed baseline

- Propagation core:
  - `PropagatorSpec`, `BuildContext`, `ProviderRegistry`, `PropagatorFactory`
  - `PropagationClient` façade for ergonomic builder/propagator workflows
  - Orekit-native providers: numerical, keplerian, dsst, tle
- Assembly/config:
  - force-model specs and assembly
  - spacecraft specs and assembly
  - attitude specs and assembly
  - YAML config loading and packaged presets
- Detector-driven mission execution (Phase 2 prototype):
  - `mission/detectors.py`: Orekit EventDetector factory (apogee, perigee, node, epoch triggers)
  - `mission/executor.py`: `ScenarioExecutor`, `MissionExecutionReport`, `ManeuverFiredEvent`
  - Occurrence policies: `first`, `every`, `nth`, `limited`
  - Guard conditions: `sma_above_m`, `sma_below_m`, `altitude_above_m`, `altitude_below_m`
  - Active window constraints: `active_window.start` / `active_window.end`
- Uncertainty propagation (Lane D):
  - `uncertainty/spec.py`: `UncertaintySpec` (method: stm | unscented-future)
  - `uncertainty/models.py`: `CovarianceRecord`, `CovarianceSeries`
  - `uncertainty/io.py`: YAML and HDF5 covariance I/O
  - split implementation modules (`matrix_io.py`, `transforms.py`, `records.py`, `stm.py`, `factory.py`) with compatibility façade in `uncertainty/propagator.py`
  - `UnscentedCovariancePropagator` stub (raises NotImplementedError, planned)
  - `StateFileClient.propagate_with_covariance()` and covariance save/load methods
- DSST builder integration:
  - explicit `dsst_propagation_type` and `dsst_state_type`
- State I/O:
  - typed state/scenario models
  - YAML/JSON load/save
  - compact state-series format (`defaults` + `columns` + `rows`)
  - HDF5 columnar series I/O with compression
  - split Orekit helper modules (`orekit_dates.py`, `orekit_resolvers.py`, `orekit_convert.py`, `orekit_ephemeris.py`, `orekit_export.py`) with compatibility façade in `states/orekit.py`
  - unified facade: `StateFileClient`
- Mission profile helpers:
  - timeline events in scenario files
  - event-referenced maneuver triggers
  - intent maneuvers with fast Keplerian approximation
  - increment and absolute target support for raise intents
  - orbital-element plot export to PNG
- Maintainability decomposition (Phase B complete):
  - mission split via compatibility façade (`mission/maneuvers.py`)
  - uncertainty split via compatibility façade (`uncertainty/propagator.py`)
  - states orekit split via compatibility façade (`states/orekit.py`)
  - propagation config split via compatibility façade (`propagation/config.py`)
- Tests:
  - focused regression suites and representative examples passing for all decomposition slices

### Current limitation (intentional for now)

- Maneuver planning is Keplerian (fast approximation), then impulses are applied during propagation replay.
- Detector-driven closed-loop execution is available via `mission.executor.ScenarioExecutor`; direct in-builder maneuver injection remains out of scope.

## 3) Architecture Snapshot

### Lane A: Propagation core

- `src/astrodyn_core/propagation/*`
- Responsibility: builder/propagator construction and Orekit-native provider access.

### Lane B: State I/O and conversions

- `src/astrodyn_core/states/*`
- Responsibility: scenario/state schema, persistence formats, date/orbit conversion helpers, ephemeris conversion.

### Lane C: Mission helpers

- `src/astrodyn_core/mission/*`
- Responsibility: scenario maneuver compilation/execution helpers and analysis plotting.

### Public ergonomic entrypoint

- `AstrodynClient` is the façade-first app entrypoint.
- Domain clients remain available for focused usage:
  - `PropagationClient`
  - `StateFileClient`
  - `MissionClient`
  - `UncertaintyClient`
  - `TLEClient`

## 4) Roadmap

## Phase 1 (complete) - Orekit-native propagation foundation

- Stable provider registry/factory and typed specs
- Numerical/Keplerian/DSST/TLE providers

## Phase 1.1 (complete) - Declarative force/spacecraft/attitude assembly

- Practical dynamics setup through typed specs
- YAML-driven dynamics model loading

## Phase 1.2 (complete) - State I/O and ephemeris bridge

- Unified state-file subsystem
- Compact and HDF5 series formats
- Orekit ephemeris conversion paths

## Phase 1.3 (complete) - Scenario timeline and maneuver intents

Done:

- timeline events (`epoch`, `elapsed`, `apogee/perigee`, nodes)
- event-triggered maneuver references
- intent maneuver support:
  - `raise_perigee` (absolute or increment)
  - `raise_semimajor_axis` / `maintain_semimajor_axis_above` (absolute or increment)
  - `change_inclination`
- mission-profile plotting

## Phase 2 (complete) - Closed-loop detector-driven maneuver execution

Goal: drive mission actions from propagation events directly, not only precompiled Keplerian times.

Delivered:

- Detector integration layer (`mission/detectors.py`):
  - `ApsideDetector` for apogee/perigee triggers
  - `NodeDetector` for ascending/descending node triggers
  - `DateDetector` for epoch/elapsed/event-reference triggers
  - Python `EventHandler` subclass applied as Orekit handler via JPype
- Occurrence policies in trigger dict (`occurrence: first | every | nth | limited`)
- Guard conditions in trigger dict (`guard: {sma_above_m, sma_below_m, altitude_above_m, altitude_below_m}`)
- Active window constraints (`active_window: {start, end}`)
- `ScenarioExecutor` class: configure + run + sample trajectory
- `MissionExecutionReport`: fired events, applied/skipped summary, total Δv
- `StateFileClient.run_scenario_detector_mode()` entrypoint
- Example: `leo_detector_mission.yaml` via `examples/scenario_missions.py --mode detector`

## Phase 2.5 (complete) - Uncertainty / Covariance Propagation

Goal: propagate orbital state covariance alongside trajectories using the STM method.

Delivered:

- `uncertainty/` module (new Lane D):
  - `UncertaintySpec`: configuration (method, stm_name, orbit_type, include_mass)
  - `CovarianceRecord` / `CovarianceSeries`: typed frozen dataclasses with numpy bridge
  - `STMCovariancePropagator`: wraps `setupMatricesComputation` + `MatricesHarvester`
  - `UnscentedCovariancePropagator`: stub (planned, raises `NotImplementedError`)
  - `create_covariance_propagator()` factory
  - YAML and HDF5 I/O (`save_covariance_series_*`, `load_covariance_series_*`)
- `StateFileClient` additions: `propagate_with_covariance()`, `save_covariance_series()`, `load_covariance_series()`
- Example: `examples/uncertainty.py`

## Phase B (complete) - Module decomposition and compatibility façades

Goal: reduce large-module coupling while preserving all existing public import paths.

Delivered:

- mission split: `models.py`, `timeline.py`, `intents.py`, `kinematics.py`, `simulation.py`
- uncertainty split: `matrix_io.py`, `transforms.py`, `records.py`, `stm.py`, `factory.py`
- states split: `orekit_dates.py`, `orekit_resolvers.py`, `orekit_convert.py`, `orekit_ephemeris.py`, `orekit_export.py`
- propagation split: `universe.py`, `parsers/dynamics.py`, `parsers/forces.py`, `parsers/spacecraft.py`
- legacy compatibility façades maintained in original module paths

## Phase C (in progress) - API governance and boundary hardening

Goal: keep refactoring gains user-visible and stable.

In progress:

- public/internal boundary policy (`docs/phasec-api-governance.md`)
- import hygiene test gate (`tests/test_api_boundary_hygiene.py`)
- façade-first examples with reduced boilerplate (`app.propagation`, `app.state`, etc.)

## Phase 3 (future) - Source-spec lane and interoperability

- External ephemeris/source specs (OEM/OCM/other bridges)
- bounded propagator composition and caching strategy

## Phase 4 (future) - Multi-satellite orchestration and advanced derivatives

- Parallel propagation orchestration
- optional field-based/derivative lanes where practical

## 5) Immediate Backlog (next sessions)

1. Finalize root export policy language for stable façade vs advanced low-level tiers.
2. Add API stability/deprecation notes to release process docs.
3. Implement Unscented Transform covariance propagation (`UnscentedCovariancePropagator`).
4. Add recurrence / `every-Nth-orbit` timeline semantics using Orekit event occurrence filtering.
5. Add CI pipeline for lint + tests.
6. Expand docs:
   - detector vs. Keplerian mode trade-offs
   - covariance interpretation guide (orbit type, frame conventions)
   - maintenance mission profiles with guard conditions.

## 6) Risks and Mitigations

### Risk: Overpromising physical fidelity of intent maneuvers

- Mitigation:
  - keep Keplerian intent solver explicitly documented as approximation
  - provide detector-driven execution option for operational behavior.

### Risk: Timeline schema complexity

- Mitigation:
  - define a minimal stable core first
  - add recurrence/window features incrementally with strict validation.

### Risk: Orekit wrapper differences across environments

- Mitigation:
  - keep converters/test coverage around date/orbit conversions
  - favor explicit helper paths and integration tests.

## 7) Session Bootstrap Checklist

When resuming development:

1. Read this file and `docs/phase1-architecture.md`.
2. Inspect root exports:
   - `src/astrodyn_core/__init__.py`
   - `src/astrodyn_core/states/__init__.py`
3. Inspect mission helpers:
   - `src/astrodyn_core/mission/maneuvers.py`
   - `src/astrodyn_core/mission/plotting.py`
4. Run tests:
   - `conda run -n astrodyn-core-env pytest -q`
5. Execute one mission scenario example:
   - `python examples/scenario_missions.py --mode intent`

## 8) Decision Log

- 2026-02-13: Single-repo architecture selected.
- 2026-02-13: Orekit-native and builder-first design locked as core policy.
- 2026-02-13: Phase 1.1 (force/spacecraft/attitude assembly) completed.
- 2026-02-19: State I/O subsystem and `StateFileClient` consolidated.
- 2026-02-19: Scenario timeline events and event-triggered maneuvers introduced.
- 2026-02-19: Intent maneuvers support both absolute targets and increments for raise cases.
- 2026-02-19: Semimajor-axis maintenance example/tests added using timeline-driven events.
- 2026-02-19: Detector-driven execution implemented via `mission/detectors.py` + `mission/executor.py`.
- 2026-02-19: Occurrence/guard/window trigger extensions kept backward-compatible (additive dict keys).
- 2026-02-19: Uncertainty lane (Lane D) added with STM covariance propagation.
- 2026-02-19: Unscented Transform covariance propagation reserved as planned-but-not-yet-implemented stub.
- 2026-02-20: Phase B module decomposition completed with compatibility façades preserved.
- 2026-02-20: `PropagationClient` added and composed in `AstrodynClient` as the preferred propagation façade path.
- 2026-02-20: Phase C API governance started with import hygiene checks and façade-tier policy docs.
