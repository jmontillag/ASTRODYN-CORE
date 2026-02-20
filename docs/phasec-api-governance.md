# Phase C API Governance and Boundary Policy

Last updated: 2026-02-20
Status: Complete
Depends on: docs/maintainability-cleanup-roadmap.md (Phase B complete)

## 1) Goal

Keep the API clean and predictable for users after refactors by enforcing a
clear public/internal contract, validating import hygiene, and establishing a
deprecation path for legacy patterns.

## 2) Public API Contract

### Tier 1 -- Facade clients (recommended for most users)

Preferred user entrypoint:

- `AstrodynClient`
  - `app.propagation` -> propagation construction workflows
  - `app.state` -> state I/O + Orekit conversion + trajectory export
  - `app.mission` -> mission planning/execution/plotting
  - `app.uncertainty` -> covariance propagation + I/O
  - `app.tle` -> TLE cache/resolve workflows

Specialized facades remain public for focused workflows:

- `PropagationClient`
- `StateFileClient`
- `MissionClient`
- `UncertaintyClient`
- `TLEClient`

### Tier 2 -- Data models and specs

All typed dataclasses and specs are public and stable:

- Propagation: `PropagatorSpec`, `PropagatorKind`, `IntegratorSpec`, `BuildContext`,
  force/spacecraft/attitude specs
- States: `OrbitStateRecord`, `StateSeries`, `ScenarioStateFile`, `OutputEpochSpec`,
  `ManeuverRecord`, `TimelineEventRecord`, `AttitudeRecord`, `parse_epoch_utc`
- Mission: `CompiledManeuver`, `MissionExecutionReport`, `ManeuverFiredEvent`, `ScenarioExecutor`
- Uncertainty: `UncertaintySpec`, `CovarianceRecord`, `CovarianceSeries`,
  `change_covariance_type`, `setup_stm_propagator`
- TLE: `TLEQuery`, `TLERecord`, `TLEDownloadResult`

### Tier 3 -- Advanced low-level helpers

For expert Orekit-native workflows:

- `PropagatorFactory`, `ProviderRegistry`, `CapabilityDescriptor`
- `register_default_orekit_providers`
- `assemble_force_models`, `assemble_attitude_provider`
- `load_dynamics_config`, `load_dynamics_from_dict`,
  `load_spacecraft_config`, `load_spacecraft_from_dict`
- `get_propagation_model`, `get_spacecraft_model`,
  `list_propagation_models`, `list_spacecraft_models`

## 3) Internal API Contract

- New cross-module usage must prefer non-underscore symbols.
- Private underscore helpers are allowed only inside compatibility facade modules.
- Compatibility facade modules (all deprecated, scheduled for removal):
  - `mission/maneuvers.py`
  - `uncertainty/propagator.py`
  - `states/orekit.py`
  - `propagation/config.py`
- These facades use `__getattr__` for lazy deprecation warnings on private aliases.

## 4) Deprecation Policy

### Active deprecations

1. **Compatibility facade private aliases**: Underscore-prefixed aliases in
   `maneuvers.py`, `propagator.py`, `orekit.py`, `config.py` emit
   `DeprecationWarning` when accessed. Import from the canonical module instead.

2. **StateFileClient cross-domain methods**: The following methods on
   `StateFileClient` are deprecated in favor of using the proper domain client
   (via `AstrodynClient` or directly):
   - `compile_scenario_maneuvers()` -> `MissionClient`
   - `export_trajectory_from_scenario()` -> `MissionClient`
   - `plot_orbital_elements()` -> `MissionClient.plot_orbital_elements_series()`
   - `run_scenario_detector_mode()` -> `MissionClient`
   - `create_covariance_propagator()` -> `UncertaintyClient`
   - `propagate_with_covariance()` -> `UncertaintyClient`
   - `save_covariance_series()` -> `UncertaintyClient`
   - `load_covariance_series()` -> `UncertaintyClient`

### Removal timeline

- Deprecated items will be removed in the next major cleanup cycle.
- Before removal: all tests and examples must already use the new paths (done).
- Removal will consist of deleting the deprecated methods and facade alias code.

## 5) Enforcement

Implemented checks in `tests/test_api_boundary_hygiene.py`:

- `test_no_private_cross_module_imports_outside_facades`: Fails if non-facade
  modules import private underscore symbols from other `astrodyn_core` modules.
- `test_root_all_consistency`: Verifies every name in root `__all__` exists
  in the module.
- `test_examples_do_not_import_private_symbols`: Ensures examples don't import
  underscore-prefixed symbols.
- `test_examples_prefer_public_subpackage_paths`: Ensures examples use public
  subpackage paths, not internal module paths.
- `test_facade_modules_use_getattr_for_deprecated_aliases`: Ensures facade
  modules use `__getattr__` for private aliases (not bare assignment).

Regression suites to run for API-affecting changes:

- `tests/test_client_facades.py`
- `tests/test_state_orekit.py`
- `tests/test_state_series_ephemeris.py`
- `tests/test_trajectory_export_wrapper.py`
- `tests/test_covariance_propagation.py`
- `tests/test_mission_maneuvers.py`
- `tests/test_mission_detector_execution.py`

## 6) Example Usability Rule

If a refactor changes architecture, examples must show a simpler or equally
simple user path.

Current rules for examples:

- Prefer `AstrodynClient` over manual registry/factory setup.
- Use `app.mission` for mission/plotting workflows (not `app.state`).
- Use `app.uncertainty` for covariance workflows (not `app.state`).
- Import from public package/subpackage paths only (no internal modules).
- Keep direct low-level examples only where they demonstrate advanced
  Orekit-native control.

## 7) Root Export Policy

Root `__all__` is organized into three tiers with clear comments:

- **Tier 1**: Facade clients (6 symbols)
- **Tier 2**: Data models and specs (~28 symbols)
- **Tier 3**: Advanced low-level helpers (~12 symbols)

Symbols removed from root `__all__` (still importable from subpackages):

- Individual TLE functions (use `TLEClient` methods instead)
- `compile_scenario_maneuvers`, `plot_orbital_elements_series` (use `MissionClient`)
- `STMCovariancePropagator`, `create_covariance_propagator`, `setup_stm_propagator`,
  `save_covariance_series`, `load_covariance_series` (use `UncertaintyClient`
  or import from `astrodyn_core.uncertainty`)

## 8) Phase C Completion Checklist

- [x] API tier policy published in root docs
- [x] Root `__all__` curated into tiered groups
- [x] Subpackage `__init__.py` files reduced to public API
- [x] Import-boundary hygiene test active and extended
- [x] Examples migrated to facade-first paths
- [x] Tests migrated to facade-first paths
- [x] Deprecation warnings on compatibility facades (with `__getattr__`)
- [x] Deprecation warnings on `StateFileClient` cross-domain methods
- [x] `change_covariance_type` and `parse_epoch_utc` promoted to public API
- [x] Documentation updated
