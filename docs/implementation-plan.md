# ASTRODYN-CORE Detailed Implementation Plan

This document is the long-term execution plan for building `astrodyn-core` as a single repository with Orekit-native semantics and a builder-first architecture.

It is written to let a new agent or developer continue work in a fresh session without losing context.

---

## 1) Product Direction and Non-Goals

### Product direction

- Build a modern, extensible astrodynamics propagation package that preserves direct Orekit usage.
- Keep `PropagatorBuilder` as a first-class concept, not hidden behind heavy wrappers.
- Support declarative configuration while still allowing direct Orekit objects when needed.
- Keep one repo (`ASTRODYN-CORE`) for now; split later only if necessary.

### Non-goals (for now)

- No adapter layer for old MOSAIC APIs.
- No full backward compatibility with legacy config schema.
- No immediate multi-repo split between propagation and ephemeris parsing.
- No production-grade remote data client in Phase 1.

---

## 2) Current Status (Already Implemented)

Implemented in this repo:

- Packaging and project scaffolding:
  - `pyproject.toml`
  - `README.md`
  - `.gitignore`
- Propagation core primitives:
  - `src/astrodyn_core/propagation/specs.py`
  - `src/astrodyn_core/propagation/interfaces.py`
  - `src/astrodyn_core/propagation/capabilities.py`
  - `src/astrodyn_core/propagation/registry.py`
  - `src/astrodyn_core/propagation/factory.py`
- Orekit-native default providers:
  - `src/astrodyn_core/propagation/providers/integrators.py`
  - `src/astrodyn_core/propagation/providers/orekit_native.py`
- Phase 1.1 force/attitude/spacecraft specs and assembly:
  - `src/astrodyn_core/propagation/forces.py`
  - `src/astrodyn_core/propagation/spacecraft.py`
  - `src/astrodyn_core/propagation/attitude.py`
  - `src/astrodyn_core/propagation/assembly.py`
- Public exports:
  - `src/astrodyn_core/__init__.py`
  - `src/astrodyn_core/propagation/__init__.py`
- Tests scaffold:
  - `tests/test_specs.py`
  - `tests/test_registry_factory.py`
- Architecture notes:
  - `docs/phase1-architecture.md`

Top-level import pattern is already supported via root exports.

---

## 3) Guiding Architecture

### 3.1 Dual-lane model

- **Orekit-native lane**
  - Providers return real Orekit `PropagatorBuilder` and/or `Propagator` objects.
  - No abstraction should remove access to native Orekit methods.
- **Extension lane**
  - Plugin providers for custom analytical models and experimental propagators.
  - Plugins should still publish Orekit-like capability contracts.

### 3.2 Core building blocks

- `PropagatorSpec`: declarative requested behavior.
- `BuildContext`: runtime input objects (initial orbit, force models, attitude, metadata).
- `ProviderRegistry`: registration and discovery by `PropagatorKind`.
- `PropagatorFactory`: orchestration for builder/progator construction.
- `CapabilityDescriptor`: feature support metadata.

### 3.3 Public API strategy

- Keep common APIs at package root (`astrodyn_core`).
- Keep advanced internals in namespaced modules.
- Add new symbols to root exports only when stable.

---

## 4) External Context Sources (Reference Map)

Use these files as migration and design references.

### MOSAIC references

- Propagation core and config:
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/config.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/core.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/auxiliary.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/output.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/orbit_utils.py`
- GEQOE analytical logic:
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/propagator.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/conversion.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/jacobians.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/utils.py`
- Estimation integration patterns:
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/estimation/bls/strategies/propagation.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/estimation/bls/interfaces.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/estimation/bls/problem.py`
  - `/home/astror/Projects/MOSAIC/src/mosaicpy/estimation/bls_factory.py`

### ASTROR references

- Declarative source-based propagation:
  - `/home/astror/Projects/ASTROR/src/astror/propagation/spec.py`
  - `/home/astror/Projects/ASTROR/src/astror/propagation/factory.py`
  - `/home/astror/Projects/ASTROR/src/astror/propagation/README.md`
- TLE subsystem:
  - `/home/astror/Projects/ASTROR/src/astror/tle/models.py`
  - `/home/astror/Projects/ASTROR/src/astror/tle/downloader.py`
  - `/home/astror/Projects/ASTROR/src/astror/tle/parser.py`
  - `/home/astror/Projects/ASTROR/src/astror/tle/propagator.py`
- Ephemeris parsing wrappers:
  - `/home/astror/Projects/ASTROR/src/astror/parsing/ephemeris/readers.py`

### Orekit references

- `https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/propagation/package-summary.html`
- `https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/propagation/conversion/package-summary.html`
- `https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/propagation/analytical/package-summary.html`
- `https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/propagation/numerical/package-summary.html`
- `https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/propagation/semianalytical/dsst/package-summary.html`

---

## 5) Phased Roadmap

## Phase 1 (in progress) - Core Orekit-native propagation

Goal: stable builder-first foundation.

### Deliverables

- [x] spec + context + registry + factory
- [x] default providers for numerical/keplerian/dsst/tle
- [x] root package exports
- [ ] robust test environment and CI
- [ ] concrete Orekit example scripts
- [ ] improve error messages around missing Orekit runtime/JVM state

### Phase 1 completion criteria

- Can instantiate all four provider kinds through factory.
- Can import package root in external project (`from astrodyn_core import ...`).
- Documentation includes usage examples and constraints.
- Unit tests pass in CI for non-Orekit logic.

## Phase 1.1 (complete) - Declarative force-model and attitude assembly

Goal: move practical setup logic from ad-hoc context into typed specs.

### Tasks

- [x] Add `forces.py` specs (`GravitySpec`, `DragSpec`, `SRPSpec`, `ThirdBodySpec`, `RelativitySpec`, `SolidTidesSpec`, `OceanTidesSpec`).
- [x] Add `spacecraft.py` spec (`SpacecraftSpec` with isotropic + box-wing models).
- [x] Add `attitude.py` specs (LOF/Nadir/Inertial/custom provider hook via `AttitudeSpec`).
- [x] Add `assembly.py` Orekit translation module (`assemble_force_models`, `assemble_attitude_provider`).
- [x] Wire assembly into providers (`orekit_native.py`) with backward-compatible fallback.
- [x] Add `force_specs`, `spacecraft`, `attitude` fields to `PropagatorSpec`.
- [x] Update root package exports.
- [ ] Add validation for incompatible combinations (deferred to hardening).

### Atmosphere models supported

- NRLMSISE00, DTM2000, JB2008 (with CSSI / MSAFE space weather)
- HarrisPriester
- SimpleExponentialAtmosphere

### Reference implementation source

- `/home/astror/Projects/MOSAIC/src/mosaicpy/propagation/auxiliary.py`

## Phase 2 - Ephemeris and source-spec lane (still same repo)

Goal: unify numerical/analytical/TLE with local/remote ephemeris inputs.

### Tasks

- Add `SourceSpec` hierarchy:
  - local OEM/OCM
  - remote CPF/SP3
  - TLE retrieval policy
- Add `bounded propagator` creation pathways.
- Add optional data clients and caching abstractions.
- Add aggregate bounded propagator composition.

### Reference implementation source

- `/home/astror/Projects/ASTROR/src/astror/propagation/spec.py`
- `/home/astror/Projects/ASTROR/src/astror/propagation/factory.py`
- `/home/astror/Projects/ASTROR/src/astror/parsing/ephemeris/readers.py`

## Phase 3 - GEQOE plugin extraction and extension lane

Goal: integrate custom analytical propagation without polluting core.

### Tasks

- Create plugin namespace `astrodyn_core.plugins.geqoe`.
- Refactor legacy GEQOE logic into clear model + state transition modules.
- Provide provider implementation with explicit capabilities.
- Document STM/Jacobian policy for non-native Orekit harvesters.

### Reference implementation source

- `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/propagator.py`
- `/home/astror/Projects/MOSAIC/src/mosaicpy/geqoe_utils/jacobians.py`

## Phase 4 - Multi-satellite and field-based derivatives

Goal: enable modern advanced workflows.

### Tasks

- Multi-sat orchestration API (Orekit `PropagatorsParallelizer` integration).
- Field-based propagation lane prototypes.
- Capability flags and runtime checks for field/STM support.

## Phase 5 - Hardening and release readiness

Goal: package reliability and adoption.

### Tasks

- Comprehensive docs and examples.
- Versioning policy and changelog.
- Benchmarks and performance reports.
- Release automation and distribution strategy.

---

## 6) Detailed Backlog for Next Sessions (Actionable)

These are immediate high-value tasks in execution order.

1. **Set up local dev environment**
   - Use `python setup_env.py` to create/update env from `environment.yml`.
   - Activate with `conda activate astrodyn-core-env`.
   - Editable package install with dev deps is handled by the script.
   - Verify `pytest` and lint tooling available.

2. **Add CI pipeline**
   - GitHub Actions for `ruff`, `pytest`, and package build.
   - Matrix for Python versions if desired.

3. **Strengthen provider tests**
   - Unit test registry and validation edge cases.
   - Mock Orekit imports for graceful failure paths.
   - Add integration tests that run with Orekit in CI.

4. **Improve error ergonomics**
   - Standardize custom exceptions (`errors.py`).
   - Include context-rich messages (kind, missing fields, import status).

5. **Add example scripts**
   - Minimal numerical builder example.
   - TLE direct propagator example.
   - DSST builder example.

6. **~~Start Phase 1.1 specs~~ (DONE)**
   - Typed force/attitude/spacecraft specs implemented.
   - Assembly module implemented with full MOSAIC force model parity.

7. **Public API contract file**
   - Add `docs/public-api.md` listing root-exported symbols and compatibility policy.

---

## 7) Technical Conventions

### Code style

- Python 3.11+ features allowed.
- `src` layout with explicit exports.
- Keep comments only where logic is non-obvious.
- Keep provider methods small and composable.

### API design

- Prefer explicit dataclasses/enums over loosely typed dict config.
- Keep `orekit_options: Mapping[str, Any]` for escape hatches.
- Validate early in `__post_init__`.

### Dependency strategy

- Keep dependencies minimal but explicit.
- Orekit is a mandatory runtime dependency (`orekit>=13.1`).
- Run Orekit-dependent tests in CI by default.

---

## 8) Risks and Mitigations

### Risk: Orekit wrapper edge cases for custom analytical classes

- Observed caveat: matrix harvester support may not be available automatically for Python-subclassed analytical propagators.
- Mitigation:
  - Expose capability flags clearly.
  - Provide fallback Jacobian interfaces for plugin models.
  - Avoid claiming native Orekit harvester support unless verified.

### Risk: Over-abstraction hides Orekit semantics

- Mitigation:
  - Keep return types Orekit-native.
  - Preserve builder/progator direct access in public API.
  - Limit wrapper logic to orchestration and validation.

### Risk: Config sprawl

- Mitigation:
  - Introduce specs incrementally.
  - Keep strict ownership per module.
  - Avoid one giant config dataclass.

---

## 9) Session Bootstrap Checklist for New Agent

When starting a new session, do this first:

1. Open this file and `docs/phase1-architecture.md`.
2. Read current package exports:
   - `src/astrodyn_core/__init__.py`
   - `src/astrodyn_core/propagation/__init__.py`
3. Inspect provider implementation:
   - `src/astrodyn_core/propagation/providers/orekit_native.py`
4. Run local sanity checks:
   - `python -m compileall src tests`
   - `python -m pytest` (if installed)
5. Pick next backlog item from Section 6 and implement.
6. Update this document with progress and decisions.

---

## 10) How to Use ASTRODYN-CORE from Another Project

### Option A: Editable local path install (recommended during active development)

From consumer project environment:

```bash
pip install -e /home/astror/Projects/ASTRODYN-CORE
```

Orekit is installed automatically because it is a core dependency.

### Option B: Git submodule + editable install

If included as submodule, still install package path in the environment:

```bash
pip install -e path/to/submodule/ASTRODYN-CORE
```

Then root imports are available:

```python
from astrodyn_core import PropagatorFactory, PropagatorSpec, PropagatorKind
```

---

## 11) Decision Log

- **2026-02-13**: single repository selected over multi-repo split.
- **2026-02-13**: architecture must remain Orekit-native and builder-first.
- **2026-02-13**: top-level import ergonomics (`from astrodyn_core import ...`) is a requirement.

- **2026-02-13**: Phase 1.1 completed: force specs, spacecraft spec, attitude spec, and assembly module.
- **2026-02-13**: SpacecraftSpec is a separate top-level spec (not nested in PropagatorSpec).
- **2026-02-13**: Force specs live on PropagatorSpec (declarative), with backward-compatible fallback to raw context.force_models.
- **2026-02-13**: Maneuver specs deferred to Phase 2+.
- **2026-02-13**: Estimation parameter drivers (cd/cr) deferred to Phase 3+.

Keep appending key decisions here in future sessions.
