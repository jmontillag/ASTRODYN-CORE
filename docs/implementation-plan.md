# ASTRODYN-CORE Implementation Plan

Last updated: 2026-02-19

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

## Phase 1.3 (in progress) - Scenario timeline and maneuver intents

Done:

- timeline events (`epoch`, `elapsed`, `apogee/perigee`, nodes)
- event-triggered maneuver references
- intent maneuver support:
  - `raise_perigee` (absolute or increment)
  - `raise_semimajor_axis` / `maintain_semimajor_axis_above` (absolute or increment)
  - `change_inclination`
- mission-profile plotting

Remaining for phase completion:

- detector-driven execution mode with Orekit event detectors in numerical propagation
- recurrence/window semantics in timeline (for robust mission operations)
- clearer mission schema docs and validation for edge cases

## Phase 2 (next) - Closed-loop maneuver execution

Goal: drive mission actions from propagation events directly, not only precompiled Keplerian times.

Planned deliverables:

- Detector integration layer for numerical propagation:
  - apside/node-based triggers
  - epoch/elapsed timeline triggers
  - trigger occurrence policies (`first`, `every`, `nth`, optional limits)
- Maneuver guard evaluation at trigger time:
  - example: maintain semimajor axis floor using on-trigger checks
- Scenario execution report:
  - which events fired, which maneuvers were skipped/applied, applied delta-v summary

## Phase 3 (future) - Source-spec lane and interoperability

- External ephemeris/source specs (OEM/OCM/other bridges)
- bounded propagator composition and caching strategy

## Phase 4 (future) - Multi-satellite orchestration and advanced derivatives

- Parallel propagation orchestration
- optional field-based/derivative lanes where practical

## 5) Immediate Backlog (next sessions)

1. Implement detector-driven mission execution prototype in `mission` module.
2. Add scenario schema fields for trigger recurrence/window constraints.
3. Add integration tests for detector mode (including guard-based maintenance cases).
4. Expand docs and examples for:
   - timeline reference patterns
   - maintenance mission profiles
   - expected approximation limits of intent solvers.
5. Add CI pipeline for lint + tests.

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
- 2026-02-19: Numerical detector-driven closed-loop mission execution kept as next major implementation target.
