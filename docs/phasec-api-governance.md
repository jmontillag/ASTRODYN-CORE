# Phase C API Governance and Boundary Policy

Last updated: 2026-02-20
Status: In progress
Depends on: docs/maintainability-cleanup-roadmap.md (Phase B complete)

## 1) Goal

Keep the API easier for users after refactors by enforcing a clear public/internal contract and validating import hygiene.

## 2) Public API Contract

Preferred user entrypoint:

- `AstrodynClient`
  - `app.propagation` → propagation construction workflows
  - `app.state` → state I/O + Orekit conversion + trajectory export
  - `app.mission` → mission planning/execution
  - `app.uncertainty` → covariance workflows
  - `app.tle` → TLE cache/resolve workflows

Specialized façades remain public for power users:

- `PropagationClient`
- `StateFileClient`
- `MissionClient`
- `UncertaintyClient`
- `TLEClient`

Low-level Orekit-native APIs remain supported:

- `PropagatorFactory`, `ProviderRegistry`, typed specs, and assembly helpers.

## 3) Internal API Contract

- New cross-module usage must prefer non-underscore symbols.
- Private underscore helpers are allowed only inside compatibility façade modules.
- Compatibility façade modules are currently:
  - `mission/maneuvers.py`
  - `uncertainty/propagator.py`
  - `states/orekit.py`
  - `propagation/config.py`

## 4) Enforcement

Implemented checks:

- `tests/test_api_boundary_hygiene.py`
  - Fails if non-façade modules import private underscore symbols from other `astrodyn_core` modules.

Regression suites to run for API-affecting changes:

- `tests/test_client_facades.py`
- `tests/test_state_orekit.py`
- `tests/test_state_series_ephemeris.py`
- `tests/test_trajectory_export_wrapper.py`
- `tests/test_covariance_propagation.py`
- `tests/test_mission_maneuvers.py`
- `tests/test_mission_detector_execution.py`

## 5) Example Usability Rule

If a refactor changes architecture, examples must show a simpler or equally simple user path.

Current rule for examples:

- Prefer `AstrodynClient` over manual registry/factory setup when equivalent behavior is possible.
- Keep direct low-level examples only where they demonstrate advanced Orekit-native control.

## 6) Next Phase C Tasks

1. Add a short “public API stability + deprecation” section to release notes template.
2. Add lint/type checks to CI if/when a project-wide lint baseline is adopted.
3. Evaluate whether root `__all__` should be grouped into “stable façade” vs “advanced low-level” exports.
