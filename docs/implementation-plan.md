# ASTRODYN-CORE Implementation Plan

Last updated: 2026-02-19 (rev 2)

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

The project is past the original "Phase 1 propagation foundation" and is now in an early mission-execution stage.

### Completed baseline

- Propagation core:
  - `PropagatorSpec`, `BuildContext`, `ProviderRegistry`, `PropagatorFactory`
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
  - `uncertainty/propagator.py`: `STMCovariancePropagator` via `setupMatricesComputation`
  - `UnscentedCovariancePropagator` stub (raises NotImplementedError, planned)
  - `StateFileClient.propagate_with_covariance()` and covariance save/load methods
- DSST builder integration:
  - explicit `dsst_propagation_type` and `dsst_state_type`
- State I/O:
  - typed state/scenario models
  - YAML/JSON load/save
  - compact state-series format (`defaults` + `columns` + `rows`)
  - HDF5 columnar series I/O with compression
  - Orekit conversion (`state -> orbit`, `state series -> ephemeris`)
  - unified facade: `StateFileClient`
- Mission profile helpers:
  - timeline events in scenario files
  - event-referenced maneuver triggers
  - intent maneuvers with fast Keplerian approximation
  - increment and absolute target support for raise intents
  - orbital-element plot export to PNG
- Tests:
  - full local suite passing (`38` tests currently)
  - scenario maneuver and timeline maintenance tests included

### Current limitation (intentional for now)

- Maneuver planning is Keplerian (fast approximation), then impulses are applied during propagation replay.
- No fully detector-driven closed-loop maneuver execution integrated in `NumericalPropagator` yet.

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

- `StateFileClient` centralizes file + conversion + scenario-export operations.

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
- Example: `leo_detector_mission.yaml`, `demo_detector_mission.py`

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
- Example: `demo_covariance_propagation.py`

## Phase 3 (future) - Source-spec lane and interoperability

- External ephemeris/source specs (OEM/OCM/other bridges)
- bounded propagator composition and caching strategy

## Phase 4 (future) - Multi-satellite orchestration and advanced derivatives

- Parallel propagation orchestration
- optional field-based/derivative lanes where practical

## 5) Immediate Backlog (next sessions)

1. Validate detector-driven execution against Keplerian-mode results for known maneuver scenarios.
2. Implement Unscented Transform covariance propagation (`UnscentedCovariancePropagator`).
3. Add recurrence / `every-Nth-orbit` timeline semantics using Orekit's event occurrence filtering.
4. Add CI pipeline for lint + tests.
5. Expand docs:
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
   - `python examples/demo_mission_maneuver_profile.py`

## 8) Decision Log

- 2026-02-13: Single-repo architecture selected.
- 2026-02-13: Orekit-native and builder-first design locked as core policy.
- 2026-02-13: Phase 1.1 (force/spacecraft/attitude assembly) completed.
- 2026-02-19: State I/O subsystem and `StateFileClient` consolidated as the default user entrypoint.
- 2026-02-19: Scenario timeline events and event-triggered maneuvers introduced.
- 2026-02-19: Intent maneuvers support both absolute targets and increments for raise cases.
- 2026-02-19: Semimajor-axis maintenance example/tests added using timeline-driven events.
- 2026-02-19: Detector-driven execution implemented via `mission/detectors.py` + `mission/executor.py`.
- 2026-02-19: Occurrence/guard/window trigger extensions kept backward-compatible (additive dict keys).
- 2026-02-19: Uncertainty lane (Lane D) added with STM covariance propagation.
- 2026-02-19: Unscented Transform covariance propagation reserved as planned-but-not-yet-implemented stub.
