# Architecture Hardening Plan (Pre-release + Finalization)

Last updated: 2026-02-24

This document defines the structural hardening plan for `src/astrodyn_core` so
the repository can be released now with low migration risk, while converging to
a strict long-term package architecture.

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

Split monoliths into cohesive modules:

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

Release policy in this phase:

- No public API removal.
- No example breakage.
- Prefer additive changes + deprecation warnings where needed.

### Phase B (next minor): deprecation enforcement

1. Emit `DeprecationWarning` on shim module imports.
2. Add tests that fail on new internal usage of shim paths.
3. Update docs/examples to canonical imports only.

### Phase C (next major): finalization

1. Remove compatibility shims:
   - `propagation/universe.py`
   - `propagation/assembly.py`
   - `propagation/dsst_assembly.py`
2. Keep only canonical package paths.
3. Tighten boundary-hygiene tests to reject shim imports entirely.

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
