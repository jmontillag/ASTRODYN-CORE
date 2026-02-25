# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-02-25

### Added

- Shared Orekit environment package `astrodyn_core.orekit_env` for universe/frame/Earth configuration and resolver helpers.
- Canonical propagation assembly subpackages:
  - `astrodyn_core.propagation.assembly_parts`
  - `astrodyn_core.propagation.dsst_parts`
- Repo-local `AGENTS.md` with required execution environment policy (`astrodyn-core-env`).
- Developer task `Makefile` shortcuts for install/test/example/native build workflows.

### Changed

- Finalized architecture for v1.0 with canonical imports across source and tests.
- Updated README developer workflow to consistently use `conda run -n astrodyn-core-env ...`.
- Added/strengthened API boundary hygiene tests to enforce final import path policy.
- `pyproject.toml` package version bumped to `1.0.0`.

### Removed

- Deprecated propagation compatibility shim modules:
  - `astrodyn_core.propagation.universe`
  - `astrodyn_core.propagation.assembly`
  - `astrodyn_core.propagation.dsst_assembly`
- Transition-only shim compatibility tests used during migration.

### Migration Notes (Import Paths)

Use these canonical imports in all code:

- `astrodyn_core.propagation.universe` -> `astrodyn_core.orekit_env`
- `astrodyn_core.propagation.assembly` -> `astrodyn_core.propagation.assembly_parts`
- `astrodyn_core.propagation.dsst_assembly` -> `astrodyn_core.propagation.dsst_parts`

### Validation

- Full test suite passes in the project Conda env (`astrodyn-core-env`).
