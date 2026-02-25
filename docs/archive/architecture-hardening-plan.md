# Architecture Hardening Plan (Completed for v1.0)

Last updated: 2026-02-25

This document records the structural hardening work for `src/astrodyn_core`
through the final architecture selected for v1.0.

## 0) Current progress snapshot (2026-02-25)

Completed in code:

- Added shared `orekit_env` package:
  - `src/astrodyn_core/orekit_env/universe_config.py`
  - `src/astrodyn_core/orekit_env/frames.py`
  - `src/astrodyn_core/orekit_env/earth.py`
  - `src/astrodyn_core/orekit_env/__init__.py`
- Introduced canonical assembly module paths used internally:
  - `src/astrodyn_core/propagation/assembly_parts/`
  - `src/astrodyn_core/propagation/dsst_parts/`
- Implemented compatibility transition (Phase A + B) to migrate imports safely
- Finalized architecture for v1.0 (Phase C):
  - Removed compatibility shim modules
  - Removed shim compatibility tests
  - Tightened hygiene tests to reject shim imports and assert shim files stay removed

Immediate next milestone:

- Release packaging/docs/tagging (architecture migration is complete).

## 1) Scope and goals

Goals:

- Keep the public API stable for immediate team adoption.
- Reduce package boundary leakage and mixed-responsibility modules.
- Prepare a clean final structure where compatibility shims are removed in a
  planned major-version step.

Current architectural strengths:

- Clear façade-first user API (`AstrodynClient` + domain clients).
- Registry/factory/provider separation for propagation.
- Good coverage of API hygiene and provider behavior in tests.

Main structural concerns addressed in this plan:

- Shared Orekit environment concerns historically lived under
  `propagation/universe.py` but are used outside propagation.
- `propagation/assembly.py` and `propagation/dsst_assembly.py` aggregate many
  responsibilities, which increases maintenance cost.

## 2) Target structure (final)

### Shared Orekit environment boundary

Create a dedicated package for configuration and Earth/frame constants
resolution:

- `src/astrodyn_core/orekit_env/universe_config.py`
- `src/astrodyn_core/orekit_env/frames.py`
- `src/astrodyn_core/orekit_env/earth.py`
- `src/astrodyn_core/orekit_env/__init__.py`

Responsibilities:

- universe config load/validation/default policy
- IERS/ITRF frame resolution
- Earth shape and gravitational parameter resolution

### Propagation force assembly decomposition

Current canonical split (already implemented):

- `src/astrodyn_core/propagation/assembly_parts/orchestrator.py`
- `src/astrodyn_core/propagation/assembly_parts/__init__.py`
- `src/astrodyn_core/propagation/dsst_parts/assembly.py`
- `src/astrodyn_core/propagation/dsst_parts/__init__.py`

Optional final naming cleanup (major release candidate):

- rename `assembly_parts/` -> `assembly/`
- rename `dsst_parts/` -> `dsst/`

Optional deeper decomposition after naming cleanup:

- `src/astrodyn_core/propagation/assembly/orchestrator.py`
- `src/astrodyn_core/propagation/assembly/attitude.py`
- `src/astrodyn_core/propagation/assembly/celestial.py`
- `src/astrodyn_core/propagation/assembly/atmosphere.py`
- `src/astrodyn_core/propagation/assembly/spacecraft_shapes.py`
- `src/astrodyn_core/propagation/assembly/forces_gravity.py`
- `src/astrodyn_core/propagation/assembly/forces_drag_srp.py`
- `src/astrodyn_core/propagation/assembly/forces_tides.py`
- `src/astrodyn_core/propagation/assembly/__init__.py`

DSST split:

- `src/astrodyn_core/propagation/dsst/assembly.py`
- `src/astrodyn_core/propagation/dsst/forces_gravity.py`
- `src/astrodyn_core/propagation/dsst/forces_drag.py`
- `src/astrodyn_core/propagation/dsst/forces_srp.py`
- `src/astrodyn_core/propagation/dsst/forces_third_body.py`
- `src/astrodyn_core/propagation/dsst/unsupported.py`
- `src/astrodyn_core/propagation/dsst/__init__.py`

## 3) Migration strategy

### Phase A (now): compatibility-first refactor

1. Introduce `orekit_env` package.
2. Move universe/frame/earth/mu logic there.
3. Keep `propagation/universe.py` as a compatibility shim (re-exports only).
4. Split assembly files into submodules.
5. Keep `propagation/assembly.py` and `propagation/dsst_assembly.py` as
   compatibility shims (re-exports only).
6. Update internal imports to canonical module paths.

Status: **Completed** (2026-02-25).

Release policy in this phase:

- No public API removal.
- No example breakage.
- Prefer additive changes + deprecation warnings where needed.

### Phase B (transition phase): deprecation enforcement

1. Emit `DeprecationWarning` on shim module imports.
2. Add tests that fail on new internal usage of shim paths.
3. Update docs/examples to canonical imports only.

Execution checklist (historical):

1. Add `DeprecationWarning` in:
  - `propagation/universe.py`
  - `propagation/assembly.py`
  - `propagation/dsst_assembly.py`
2. Add boundary-hygiene assertions preventing internal source files from
  importing these shim modules.
3. Keep examples on public package paths only (`astrodyn_core.propagation`),
  but avoid direct imports from shim modules.
4. Add docs migration table: old path -> canonical path.
5. Run transition test gate in project env:
  - `conda run -n astrodyn-core-env pytest -q -rs tests/test_universe_config.py tests/test_dsst_assembly.py tests/test_registry_factory.py tests/test_api_boundary_hygiene.py`

### Phase B migration table (for docs/release notes)

Use these canonical imports for all code. The deprecated paths listed below were
available only during the transition window and have now been removed in v1.0.

| Deprecated path | Canonical path | Notes |
|---|---|---|
| `astrodyn_core.propagation.universe` | `astrodyn_core.orekit_env` | Shared universe/frame/earth config helpers |
| `astrodyn_core.propagation.assembly` | `astrodyn_core.propagation.assembly_parts` | Force/attitude assembly helpers |
| `astrodyn_core.propagation.dsst_assembly` | `astrodyn_core.propagation.dsst_parts` | DSST force assembly helpers |

### Phase C (executed for v1.0): finalization

1. Remove compatibility shims:
   - `propagation/universe.py`
   - `propagation/assembly.py`
   - `propagation/dsst_assembly.py`
2. Keep only canonical package paths.
3. Tighten boundary-hygiene tests to reject shim imports entirely.

Status: **Completed for v1.0** (2026-02-25).

Execution checklist (completed):

1. Remove shim files and update all references to canonical modules.
2. If naming cleanup is accepted, fold `assembly_parts`/`dsst_parts` into
  `assembly`/`dsst` package names before or during shim removal.
3. Publish migration note in release docs with exact import replacements.
4. Run full suite in project env:
  - `conda run -n astrodyn-core-env pytest -q`

## 4) Risk controls

- Migrate internal imports first, then examples/docs, then remove shims.
- Execute targeted test batches before full suite:
  - `tests/test_universe_config.py`
  - `tests/test_dsst_assembly.py`
  - `tests/test_registry_factory.py`
  - `tests/test_api_boundary_hygiene.py`
- During the transition window, shim compatibility tests were used and then removed
  after finalization.

## 5) Ownership boundaries (steady state)

- `orekit_env`: shared environment config + frame/earth/mu resolution
- `propagation`: provider/factory/spec + force assembly orchestration
- `states`: state-file models, conversion, and I/O (consuming shared `orekit_env`)
- `mission`: timeline compilation/execution workflows
- `uncertainty`: covariance propagation methods + transforms

## 6) Exit criteria

The architecture is considered fully formalized when all are true:

1. No internal module imports rely on compatibility shims.
2. Examples use only public package paths.
3. Boundary hygiene tests include and enforce final path policy.
4. Shims removed in major release with migration notes published.

Status: **Met for v1.0** (shim modules removed early before public adoption).

## 7) Recommended next action now

Prepare release artifacts/documentation:

1. Publish release notes including the canonical import paths (migration table above).
2. Tag and release v1.0.
3. Optionally add CI to re-run `conda run -n astrodyn-core-env pytest -q` on pushes/PRs.
