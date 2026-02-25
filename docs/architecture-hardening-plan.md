# Architecture Hardening Plan (Pre-release + Finalization)

Last updated: 2026-02-25

This document defines the structural hardening plan for `src/astrodyn_core` so
the repository can be released now with low migration risk, while converging to
a strict long-term package architecture.

## 0) Current progress snapshot (2026-02-25)

Completed in code:

- Added shared `orekit_env` package:
  - `src/astrodyn_core/orekit_env/universe_config.py`
  - `src/astrodyn_core/orekit_env/frames.py`
  - `src/astrodyn_core/orekit_env/earth.py`
  - `src/astrodyn_core/orekit_env/__init__.py`
- Converted to compatibility shims:
  - `src/astrodyn_core/propagation/universe.py`
  - `src/astrodyn_core/propagation/assembly.py`
  - `src/astrodyn_core/propagation/dsst_assembly.py`
- Introduced canonical assembly module paths used internally:
  - `src/astrodyn_core/propagation/assembly_parts/`
  - `src/astrodyn_core/propagation/dsst_parts/`
- Added transition tests:
  - `tests/test_universe_shim_compat.py`
  - `tests/test_assembly_shim_compat.py`

Immediate next milestone:

- Phase B (deprecation enforcement + canonical import hygiene).

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

Main structural concerns to address:

- Shared Orekit environment concerns currently live under
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

### Phase B (next minor): deprecation enforcement

1. Emit `DeprecationWarning` on shim module imports.
2. Add tests that fail on new internal usage of shim paths.
3. Update docs/examples to canonical imports only.

Execution checklist (next steps):

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
  - `conda run -n astrodyn-core-env pytest -q -rs tests/test_universe_config.py tests/test_universe_shim_compat.py tests/test_assembly_shim_compat.py tests/test_dsst_assembly.py tests/test_registry_factory.py tests/test_api_boundary_hygiene.py`

### Phase C (next major): finalization

1. Remove compatibility shims:
   - `propagation/universe.py`
   - `propagation/assembly.py`
   - `propagation/dsst_assembly.py`
2. Keep only canonical package paths.
3. Tighten boundary-hygiene tests to reject shim imports entirely.

Execution checklist (major release prep):

1. Remove shim files and update all references to canonical modules.
2. If naming cleanup is accepted, fold `assembly_parts`/`dsst_parts` into
  `assembly`/`dsst` package names before or during shim removal.
3. Publish migration note in release docs with exact import replacements.
4. Run full suite in project env:
  - `conda run -n astrodyn-core-env pytest -q`

## 4) Risk controls

- Keep function/symbol names stable through shims during Phase A.
- Migrate internal imports first, then examples/docs, then enforce warnings.
- Execute targeted test batches before full suite:
  - `tests/test_universe_config.py`
  - `tests/test_dsst_assembly.py`
  - `tests/test_registry_factory.py`
  - `tests/test_api_boundary_hygiene.py`
- Add shim compatibility tests during transition:
  - `tests/test_universe_shim_compat.py`
  - `tests/test_assembly_shim_compat.py`

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

## 7) Recommended next action now

Implement Phase B.1 and B.2 in one PR:

1. Add deprecation warnings to shim modules.
2. Enforce internal no-shim-import policy in hygiene tests.
3. Keep compatibility tests until major release.
