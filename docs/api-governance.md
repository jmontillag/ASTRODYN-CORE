# API Governance and Boundary Policy

Last updated: 2026-02-22

## 1) Goal

Keep the API clean and predictable for users and contributors by enforcing a
clear public/internal contract, validating import hygiene, and providing
stable import paths.

## 2) Public API Contract

### Tier 1 -- Facade clients (recommended for most users)

Preferred user entrypoint:

- `AstrodynClient`
  - `app.propagation` -> propagation construction workflows
  - `app.state` -> state I/O + Orekit conversion + trajectory export
  - `app.mission` -> mission planning/execution/plotting
  - `app.uncertainty` -> covariance propagation + I/O
  - `app.tle` -> TLE cache/resolve workflows
  - `app.ephemeris` -> ephemeris file parsing + propagator creation

Specialized facades remain public for focused workflows:

- `PropagationClient`
- `StateFileClient`
- `MissionClient`
- `UncertaintyClient`
- `TLEClient`
- `EphemerisClient`

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
- Ephemeris: `EphemerisSpec`, `EphemerisFormat`

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

- Cross-module usage must prefer non-underscore symbols.
- Private underscore helpers are allowed only within a module's own files.
- All imports use canonical module paths directly (no re-export facade modules).
- Internal module paths (e.g. `states.orekit_convert`, `propagation.parsers.dynamics`,
  `uncertainty.stm`, `uncertainty.factory`) are implementation details and must
  not be imported by examples.

## 4) Deprecation Policy

When removing public API in the future:

1. Add `DeprecationWarning` in the current release.
2. Migrate all tests and examples to use the new path.
3. Remove the deprecated code in the next cleanup cycle.

## 5) Enforcement

Implemented checks in `tests/test_api_boundary_hygiene.py`:

- `test_no_private_cross_module_imports`: Fails if any source module imports
  private underscore symbols from other `astrodyn_core` modules.
- `test_root_all_consistency`: Verifies every name in root `__all__` exists
  in the module.
- `test_examples_do_not_import_private_symbols`: Ensures examples don't import
  underscore-prefixed symbols.
- `test_examples_prefer_public_subpackage_paths`: Ensures examples use public
  subpackage paths, not internal module paths.

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

- **Tier 1**: Facade clients (7 symbols including EphemerisClient)
- **Tier 2**: Data models and specs (~30 symbols)
- **Tier 3**: Advanced low-level helpers (~12 symbols)
