# ASTRODYN-CORE

Phase 1 foundation for a builder-first propagation package that keeps Orekit APIs first-class while enabling future extension plugins.

## Design principles

- Orekit-native semantics stay visible: providers return real Orekit builders/propagators.
- Builder-first API: `PropagatorSpec` drives provider selection and construction.
- Registry-based extensibility: new providers can be plugged in without editing core factory logic.
- Single-repo architecture: propagation and source-spec concerns live together for now.

## Phase 1 scope

- Orekit-native builder providers for:
  - numerical
  - keplerian
  - dsst
  - tle
- Declarative configuration objects (`PropagatorSpec`, `IntegratorSpec`, `TLESpec`).
- Factory registry and high-level construction entry points.

## Quick start

Environment setup (recommended):

```bash
python setup_env.py
conda activate astrodyn-core-env
```

```python
from astrodyn_core.propagation import (
    BuildContext,
    IntegratorSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    register_default_orekit_providers,
)

factory = PropagatorFactory()
register_default_orekit_providers(factory.registry)

spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=1200.0,
    integrator=IntegratorSpec(
        kind="dormand_prince_853",
        min_step=0.001,
        max_step=300.0,
        position_tolerance=10.0,
    ),
)

ctx = BuildContext(
    initial_orbit=initial_orbit,
    position_tolerance=10.0,
)

builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
```

See `src/astrodyn_core/propagation` for the core interfaces and default Orekit-native providers.

## Examples

Run from the project root:

```bash
python examples/demo_propagation.py
python examples/demo_orbit_plot.py
python examples/demo_keplerian_builder.py
python examples/demo_dsst_builder.py
python examples/demo_tle_propagation.py
python examples/demo_state_file_numerical.py
```

State-file driven example input lives at:

- `examples/state_files/leo_initial_state.yaml`

## Development planning

- Detailed long-term plan: `docs/implementation-plan.md`
- Current architecture snapshot: `docs/phase1-architecture.md`
