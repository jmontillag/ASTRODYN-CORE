---
name: astrodyn-core-consumer
description: Use `astrodyn_core` as the primary interface for propagation, state-file, mission, uncertainty, TLE, and ephemeris tasks in repositories that consume ASTRODYN-CORE as a dependency. Trigger for requests that involve `AstrodynClient`, `PropagatorSpec`, `BuildContext`, trajectory export, scenario simulation, covariance propagation, TLE resolution, or migration away from raw Orekit glue code. Prefer facade clients and root exports, and use Orekit MCP lookups when exact Orekit signatures are required.
---

# ASTRODYN-CORE Consumer

## Goal

Use `astrodyn_core` directly (no extra wrapper layer) so dependent repositories
get consistent propagation workflows while preserving Orekit-native outputs.

## Use Facade Tier First

Start with `AstrodynClient` and domain clients:

- `app.propagation`
- `app.state`
- `app.mission`
- `app.uncertainty`
- `app.tle`
- `app.ephemeris`

Prefer root exports from `astrodyn_core` over deep internal imports.

## Required Environment Assumptions

- Run Python/tests in `astror-env`.
- Use editable dependency install from:
  - `/home/astror/Projects/ASTRODYN-CORE`
- Use:
  - `conda run -n astror-env python ...`
  - `conda run -n astror-env pytest ...`

## Standard Workflow

1. Identify task lane:
   - propagation build/propagate
   - state-file load/save/export
   - mission scenario execution
   - uncertainty covariance propagation
   - TLE resolution
   - ephemeris ingest
2. Build typed inputs (`PropagatorSpec`, `BuildContext`, `OutputEpochSpec`,
   `UncertaintySpec`, `TLEQuery`) before calling execution methods.
3. Select builder lane vs direct propagator lane:
   - `build_builder(...)` for numerical/keplerian/dsst
   - `build_propagator(...)` for tle and analytical providers
4. Produce artifacts using facade clients instead of ad-hoc serialization.
5. Run targeted validation in `astror-env`.

## Builder vs Propagator Rule

- Use `build_builder` and `builder.buildPropagator(...)` when the provider
  exposes an Orekit builder.
- Use `build_propagator` when the provider is direct-construction
  (for example `PropagatorKind.TLE`).
- Use `BuildContext.from_state_record(...)` or
  `app.propagation.context_from_state(...)` when the initial condition is a
  serialized state record.

## Guardrails

- Keep app code on public imports (`astrodyn_core` root, or documented
  subpackages).
- Avoid provider-internal imports (`astrodyn_core.propagation.providers.*`)
  unless explicitly requested.
- Avoid raw Orekit setup when `astrodyn_core` already provides the capability.

## Orekit API Uncertainty Rule

If exact Orekit Java signatures/overloads are needed, use `orekit_docs` MCP
tools before writing code:

1. `orekit_docs_info`
2. `orekit_search_symbols`
3. `orekit_get_class_doc`
4. `orekit_get_member_doc`

If tools are unavailable, state that and proceed with clearly marked
assumptions.

## References

Read concrete code patterns from:

- `references/patterns.md`
