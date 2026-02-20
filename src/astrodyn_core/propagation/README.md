# `propagation` module

Typed propagation configuration and Orekit-native provider system.

## Purpose

The `propagation` module turns declarative specs into concrete Orekit builders/propagators while keeping Orekit classes first-class.

Core pieces:
- **Specs**: `PropagatorSpec`, `IntegratorSpec`, `TLESpec`, force/attitude/spacecraft specs.
- **Factory/Registry**: `PropagatorFactory` + `ProviderRegistry`.
- **Providers**: default Orekit-native implementations for numerical, Keplerian, DSST, and TLE.
- **Assembly**: converts force and attitude specs into Orekit objects.
- **Config loading**: YAML/dict loaders for dynamics, spacecraft, and universe models.

## Internal flow

1. User creates or loads a `PropagatorSpec`.
2. User builds a `BuildContext` (initial orbit, tolerance, optional universe/force/attitude overrides).
3. `PropagatorFactory` selects provider by `PropagatorKind` from `ProviderRegistry`.
4. Provider builds a builder or propagator using Orekit-native APIs.
5. For numerical/DSST cases, force models and attitude can come from typed specs via `assembly`.

## Key public API

Common entry points:
- `PropagatorFactory`
- `register_default_orekit_providers(registry)`
- `PropagatorSpec`, `PropagatorKind`, `IntegratorSpec`, `TLESpec`
- `assemble_force_models(...)`, `assemble_attitude_provider(...)`
- `load_dynamics_config(...)`, `load_spacecraft_config(...)`, `load_universe_config(...)`

## Intended use cases

1. **Baseline orbital propagation**
   - Keplerian for quick checks.
2. **Higher-fidelity mission analysis**
   - Numerical/DSST with force model stacks and spacecraft properties.
3. **TLE-based operations**
   - SGP4-style propagation from line pairs.
4. **Extensibility**
   - custom providers registered into the same factory/registry path.

See `examples/quickstart.py` modes: `keplerian`, `numerical`, `dsst`, `tle`.

## Boundaries

- Propagation construction lives here.
- Scenario file workflows, mission execution, and persistence live in `states` and `mission`.
