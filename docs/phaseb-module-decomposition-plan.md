# Phase B Module Decomposition Plan

Last updated: 2026-02-20
Status: Complete (Slices 1-4 complete)
Depends on: docs/maintainability-cleanup-roadmap.md (Phase A complete)

Progress snapshot:

- Slice 1 (mission split): complete
- Slice 2 (uncertainty split): complete
- Slice 3 (states/orekit split): complete
- Slice 4 (propagation/config split): complete
- Next: Phase C (API governance and boundary hardening)

## 1) Goal

Decompose large multi-responsibility modules into cohesive submodules while preserving all existing public behavior and import compatibility.

Primary targets:

- src/astrodyn_core/mission/maneuvers.py
- src/astrodyn_core/uncertainty/propagator.py
- src/astrodyn_core/propagation/config.py
- src/astrodyn_core/states/orekit.py

## 2) Design Constraints

- No breaking API changes in this phase.
- Existing public imports continue to work.
- Existing tests remain the behavior oracle.
- Keep façade layer behavior unchanged.
- Internal private cross-imports should be reduced or eliminated.

## 3) Target Topology

## 3.1 mission lane

Current issue:

- maneuvers.py combines data models, timeline resolution, trigger logic, intent solvers, vector math, simulation/export orchestration, and state conversion.
- detectors.py and executor.py import private helpers from maneuvers.py.

Target modules:

- src/astrodyn_core/mission/models.py
  - CompiledManeuver
  - ResolvedTimelineEvent
- src/astrodyn_core/mission/timeline.py
  - resolve_timeline_events (public internal utility)
  - resolve_trigger_date
  - resolve_maneuver_trigger
- src/astrodyn_core/mission/intents.py
  - resolve_delta_v_vector
  - intent_raise_perigee
  - intent_raise_semimajor_axis
  - intent_change_inclination
- src/astrodyn_core/mission/kinematics.py
  - local_to_inertial_delta_v
  - local_basis_vectors
  - rotate_vector_about_axis
  - unit, tuple_to_vector, vector tuple helpers
- src/astrodyn_core/mission/simulation.py
  - compile_scenario_maneuvers
  - simulate_scenario_series
  - export_scenario_series
  - state_to_record and impulse application wrappers

Compatibility strategy:

- Keep src/astrodyn_core/mission/maneuvers.py as orchestration façade that re-exports moved symbols.
- Replace private underscore cross-import usage in detectors.py and executor.py with non-underscore internal APIs from timeline.py/intents.py.

## 3.2 uncertainty lane

Current issue:

- propagator.py contains low-level Java array glue, covariance transforms, record conversion, STM class, and factory functions in one file.

Target modules:

- src/astrodyn_core/uncertainty/matrix_io.py
  - realmatrix_to_numpy
  - numpy_to_realmatrix
  - java double[][] helpers
- src/astrodyn_core/uncertainty/transforms.py
  - change_covariance_type
  - orbit_jacobian
  - frame_jacobian
  - transform_covariance_with_jacobian
  - orekit orbit/angle adapters
- src/astrodyn_core/uncertainty/records.py
  - state_to_orbit_record
  - numpy_to_nested_tuple
- src/astrodyn_core/uncertainty/stm.py
  - STMCovariancePropagator implementation
  - setup logic helpers
- src/astrodyn_core/uncertainty/factory.py
  - setup_stm_propagator
  - create_covariance_propagator
  - Unscented stub entrypoint wiring

Compatibility strategy:

- Keep src/astrodyn_core/uncertainty/propagator.py as backward-compatible façade that imports and re-exports current public symbols.
- Keep existing tests importing uncertainty.propagator unchanged initially.

## 3.3 propagation lane

Current issue:

- config.py combines universe model loading, runtime resolver getters, and dynamics/spacecraft parser logic.

Target modules:

- src/astrodyn_core/propagation/universe.py
  - universe loading and normalization
  - get_iers_conventions/get_itrf_frame/get_mu/get_earth_shape
- src/astrodyn_core/propagation/parsers/dynamics.py
  - load_dynamics_config
  - load_dynamics_from_dict
  - parse_integrator/attitude/forces
- src/astrodyn_core/propagation/parsers/spacecraft.py
  - load_spacecraft_config
  - load_spacecraft_from_dict
  - parse_structured_spacecraft_v1
- src/astrodyn_core/propagation/parsers/forces.py
  - parse_gravity/drag/srp/third_body/relativity/solid_tides/ocean_tides

Compatibility strategy:

- Keep src/astrodyn_core/propagation/config.py as façade that re-exports previous function names.
- Preserve all currently exported names in propagation/__init__.py.

## 3.4 states lane

Current issue:

- orekit.py mixes conversion, resolver helpers, ephemeris generation, export sampling, and record serialization.

Target modules:

- src/astrodyn_core/states/orekit_dates.py
  - to_orekit_date
  - from_orekit_date
- src/astrodyn_core/states/orekit_resolvers.py
  - resolve_frame
  - resolve_mu
- src/astrodyn_core/states/orekit_convert.py
  - to_orekit_orbit
  - state_to_record
- src/astrodyn_core/states/orekit_ephemeris.py
  - state_series_to_ephemeris
  - scenario_to_ephemeris
  - interpolation sample resolution
- src/astrodyn_core/states/orekit_export.py
  - export_trajectory_from_propagator
  - sampling ephemeris resolution and bounds validation

Compatibility strategy:

- Keep src/astrodyn_core/states/orekit.py as façade and re-export existing public entrypoints.

## 4) Migration Sequence

## Slice 1 — mission split (highest coupling payoff)

1. Introduce new mission submodules.
2. Move logic with minimal edits.
3. Keep maneuvers.py façade exports.
4. Update detectors.py and executor.py imports to non-private APIs.
5. Run mission + detector + façade tests.

## Slice 2 — uncertainty split (largest file)

1. Introduce uncertainty submodules.
2. Move helper layers first, STM class second, factory third.
3. Keep propagator.py façade exports.
4. Run covariance + façade tests.

## Slice 3 — states orekit split

1. Introduce states orekit submodules.
2. Move date/resolver/convert first, export/ephemeris second.
3. Keep orekit.py façade exports.
4. Run state_orekit + series + trajectory wrapper tests.

## Slice 4 — propagation config split

1. Introduce universe + parser modules.
2. Keep config.py façade exports.
3. Run universe/spec/config/factory tests.

## 5) Verification Matrix

Minimum per-slice test requirements:

- Mission slice:
  - tests/test_mission_maneuvers.py
  - tests/test_mission_detector_execution.py
  - tests/test_client_facades.py

- Uncertainty slice:
  - tests/test_covariance_propagation.py
  - tests/test_client_facades.py

- States slice:
  - tests/test_state_orekit.py
  - tests/test_state_series_ephemeris.py
  - tests/test_trajectory_export_wrapper.py
  - tests/test_state_io.py

- Propagation config slice:
  - tests/test_universe_config.py
  - tests/test_specs.py
  - tests/test_registry_factory.py
  - tests/test_spacecraft_config.py

Cross-check after each slice:

- run examples/scenario_missions.py --mode detector
- run examples/uncertainty.py

## 6) API Compatibility and Deprecation

- No symbol removals during Phase B.
- Existing import paths stay valid via façade modules.
- If any symbol rename becomes unavoidable, add alias and deprecation note in docstrings and release notes.

## 7) Risks and Controls

Risk: Hidden circular imports after splitting.
Control: Move shared leaf utilities first; keep import direction top-down (helpers -> orchestration).

Risk: Behavior drift in vector/trigger math.
Control: Keep function bodies unchanged during first move; refactor only after parity tests pass.

Risk: Partial migration leaves mixed private imports.
Control: Add temporary checklist in each slice PR and block merge until private cross-imports are removed for that slice.

## 8) Done Criteria for Phase B

- Large target modules decomposed according to topology.
- Legacy façade files remain thin and mostly import/re-export layers.
- No private underscore cross-imports across mission modules.
- Full focused suite plus representative examples pass.
- Architecture docs updated with new internal module map.
