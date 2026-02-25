# First Propagation

This page shows the shortest facade-first workflow to build an Orekit propagator
from typed specs.

## Goal

Learn the core objects you will use most often:

- `AstrodynClient`
- `PropagatorSpec`
- `IntegratorSpec`
- `BuildContext`

## Minimal example (facade-first)

```python
from astrodyn_core import AstrodynClient, BuildContext, IntegratorSpec, PropagatorKind, PropagatorSpec

app = AstrodynClient()

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

# Supply an Orekit orbit object obtained from your own setup code.
ctx = BuildContext(initial_orbit=initial_orbit, position_tolerance=10.0)

builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
```

## Where does `initial_orbit` come from?

It is an Orekit orbit instance (for example `KeplerianOrbit`). For a runnable
end-to-end script, start with:

- `examples/quickstart.py`

## Next steps

- Read [Run Examples](examples.md)
- Continue to the propagation tutorial in [Tutorials](../tutorials/propagation-quickstart.md)
