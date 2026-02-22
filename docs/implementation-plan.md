# ASTRODYN-CORE Implementation Plan

Last updated: 2026-02-22 (rev 4)

This document tracks current implementation status and the forward plan for
propagation, state I/O, mission-profile, and extensibility capabilities.

## 1) Product Direction

- Keep Orekit-native builder/propagator access first-class.
- Keep declarative specs as orchestration and validation layers.
- Support extensibility for custom/analytical propagators (e.g. GEqOE J2 Taylor
  propagators from MOSAIC) via the same registry/factory pattern used by built-in
  Orekit providers.
- Provide a practical mission workflow from scenario/state files through
  propagation, trajectory export, and basic analysis.
- Physical constants (mu, J2, Re) must always be resolved from Orekit
  `Constants`, never hardcoded.

## 2) Current Stage

The project has completed all planned refactoring phases (1 through D) and an
architecture freeze revision (2026-02-22).  The architecture is considered
stable and ready for extension with custom propagators.

### Completed baseline

- Propagation core:
  - `PropagatorSpec`, `BuildContext`, `ProviderRegistry`, `PropagatorFactory`
  - `PropagationClient` facade for ergonomic builder/propagator workflows
  - Orekit-native providers: numerical, keplerian, dsst, tle
  - Extensible registry: accepts any string as propagator kind (not limited to
    the `PropagatorKind` enum)
  - `CapabilityDescriptor` with `is_analytical` and `supports_custom_output` flags
  - `BuildContext.body_constants` and `require_body_constants()` for analytical
    providers (resolved from Orekit `Constants`, never hardcoded)
- Assembly/config:
  - force-model specs and assembly
  - spacecraft specs and assembly
  - attitude specs and assembly
  - YAML config loading and packaged presets
- Detector-driven mission execution:
  - `mission/detectors.py`: Orekit EventDetector factory
  - `mission/executor.py`: `ScenarioExecutor`, `MissionExecutionReport`
  - Occurrence policies, guard conditions, active window constraints
- Uncertainty propagation:
  - `UncertaintySpec` (method: stm)
  - `CovarianceRecord`, `CovarianceSeries`
  - `STMCovariancePropagator`, YAML and HDF5 I/O
- DSST builder integration
- State I/O: typed models, YAML/JSON/HDF5, compact series format
- Ephemeris module: OEM, OCM, SP3, CPF parsing and propagator creation
- Tests: 128 tests passing, import hygiene enforcement

### Architecture freeze decisions (2026-02-22)

- Removed 4 backward-compatibility re-export facade files
  (`propagation/config.py`, `mission/maneuvers.py`, `uncertainty/propagator.py`,
  `states/orekit.py`).  All internal imports now use canonical module paths.
- Removed `UnscentedCovariancePropagator` stub.  "unscented" is no longer a
  valid `UncertaintySpec.method` value.  Will be re-added when implemented.
- Opened `PropagatorKind` for extensibility: `ProviderRegistry` accepts any
  string as a kind key; `PropagatorSpec.kind` accepts `PropagatorKind | str`.
- Added `body_constants` field to `BuildContext` for analytical providers.
  Constants are resolved from Orekit `Constants` (WGS84) at call time, never
  hardcoded.
- Added `is_analytical` and `supports_custom_output` flags to
  `CapabilityDescriptor` for distinguishing custom propagators from Orekit-native
  ones.
- Renamed `PropagationClient.build_numerical_propagator_from_state()` to
  `build_propagator_from_state()` (kind-agnostic).
- Consolidated docs from 6 files to 3: this file, `api-governance.md`, and
  `extending-propagators.md`.

## 3) Architecture Snapshot

### Lane A: Propagation core

- `src/astrodyn_core/propagation/*`
- Responsibility: builder/propagator construction, Orekit-native and custom
  provider access.

### Lane B: State I/O and conversions

- `src/astrodyn_core/states/*`
- Responsibility: scenario/state schema, persistence, date/orbit conversion,
  ephemeris conversion.

### Lane C: Mission helpers

- `src/astrodyn_core/mission/*`
- Responsibility: maneuver compilation/execution and analysis plotting.

### Lane D: Uncertainty propagation

- `src/astrodyn_core/uncertainty/*`
- Responsibility: STM covariance propagation, I/O, type transforms.

### Lane E: Ephemeris

- `src/astrodyn_core/ephemeris/*`
- Responsibility: ephemeris file parsing, downloading, propagator creation.

### Public ergonomic entrypoint

- `AstrodynClient` is the facade-first app entrypoint.
- Domain clients remain available for focused usage.

## 4) Roadmap

### Completed phases

- Phase 1: Orekit-native propagation foundation
- Phase 1.1: Declarative force/spacecraft/attitude assembly
- Phase 1.2: State I/O and ephemeris bridge
- Phase 1.3: Scenario timeline and maneuver intents
- Phase 2: Closed-loop detector-driven maneuver execution
- Phase 2.5: STM covariance propagation
- Phase B: Module decomposition
- Phase C: API governance and boundary hardening
- Phase D: Deprecated code removal + Ephemeris module
- Architecture freeze revision (2026-02-22)

### Phase 3 (next) -- Custom propagator extension

Goal: port GEqOE J2 Taylor propagator from MOSAIC as the first custom
analytical propagator, validating the extension architecture.

Steps:

1. Port `geqoe_utils` math core (conversions, Jacobians, Taylor propagator)
   into `src/astrodyn_core/propagation/providers/geqoe/`.
2. Port `extra_utils` math helpers (derivative combinatorics).
3. Implement `GEqOEProvider` as a JPype subclass of Orekit
   `AbstractPropagator` that wraps the numpy-based Taylor propagator.
4. Register provider with the `ProviderRegistry`.
5. Expose backend-specific output (raw GEqOE states, analytical STM) through
   `supports_custom_output` capability.

See `docs/extending-propagators.md` for the full extension pattern.

### Phase 4 (future) -- Multi-satellite and advanced derivatives

- Parallel propagation orchestration.
- Optional field-based derivative lanes.
- Unscented transform covariance propagation.

## 5) Immediate Backlog

1. Port GEqOE propagator from MOSAIC (Phase 3).
2. Add recurrence / `every-Nth-orbit` timeline semantics.
3. Add CI pipeline for lint + tests.
4. Implement Unscented Transform covariance propagation.

## 6) Risks and Mitigations

### Risk: Overpromising physical fidelity of intent maneuvers

- Mitigation: keep Keplerian intent solver explicitly documented as approximation;
  provide detector-driven execution option for operational behavior.

### Risk: Custom propagator integration friction

- Mitigation: extension guide documents exact steps; GEqOE port validates the
  pattern before architecture freeze is final.

### Risk: Orekit wrapper differences across environments

- Mitigation: keep converters/test coverage around date/orbit conversions;
  favor explicit helper paths and integration tests.

## 7) Session Bootstrap Checklist

When resuming development:

1. Read this file and `docs/extending-propagators.md`.
2. Inspect root exports: `src/astrodyn_core/__init__.py`.
3. Run tests: `conda run -n astrodyn-core-env pytest -q`.
4. Execute one example: `python examples/scenario_missions.py --mode intent`.

## 8) Decision Log

- 2026-02-13: Single-repo architecture selected.
- 2026-02-13: Orekit-native and builder-first design locked as core policy.
- 2026-02-13: Phase 1.1 (force/spacecraft/attitude assembly) completed.
- 2026-02-19: State I/O subsystem and `StateFileClient` consolidated.
- 2026-02-19: Timeline events, intent maneuvers, detector-driven execution.
- 2026-02-19: Uncertainty lane (Lane D) added with STM covariance propagation.
- 2026-02-20: Phase B module decomposition completed.
- 2026-02-20: Phase C API governance completed.
- 2026-02-21: Phase D: deprecated code removed, ephemeris module added.
- 2026-02-22: Architecture freeze revision:
  - Deleted 4 backward-compatibility re-export facade files.
  - Removed UnscentedCovariancePropagator stub.
  - Opened PropagatorKind/ProviderRegistry for custom propagator kinds.
  - Added body_constants to BuildContext (Orekit Constants-based, no hardcoding).
  - Added is_analytical and supports_custom_output to CapabilityDescriptor.
  - Renamed build_numerical_propagator_from_state -> build_propagator_from_state.
  - Consolidated docs from 6 files to 3.
  - Created extending-propagators.md guide for future contributors.
