# How-To: Numerical Propagation from a Typed Initial State (Self-Contained)

This recipe is designed for **package users** who installed `astrodyn-core`
with `pip` and want a realistic Orekit-backed propagation example without
cloning the repo.

It uses:

- `OrbitStateRecord` for the initial condition (serializable, state-file-friendly)
- `PropagatorSpec` + `IntegratorSpec` + `GravitySpec` for the propagator config
- `AstrodynClient.propagation.build_propagator_from_state(...)` for the facade-first workflow

## When to use this recipe

Use this when you want to:

- run a numerical propagator from a typed state record
- avoid manual Orekit orbit construction in your first script
- build intuition for the facade + spec API pattern

## Prerequisites

- `astrodyn-core` installed
- Orekit data configured

This snippet assumes `orekit-data.zip` is in the current working directory and
uses `setup_orekit_curdir()`. If that is not true in your setup, see:

- [Orekit Data Setup](../getting-started/orekit-data.md)

## Run (copy-paste)

```python
import math

import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.frames import FramesFactory

from astrodyn_core import (
    AstrodynClient,
    GravitySpec,
    IntegratorSpec,
    OrbitStateRecord,
    PropagatorKind,
    PropagatorSpec,
)

app = AstrodynClient()

# Serializable initial condition (can later be saved to a state file)
initial_state = OrbitStateRecord(
    epoch="2026-02-19T00:00:00Z",
    frame="GCRF",
    representation="keplerian",
    elements={
        "a_m": 7578137.0,
        "e": 0.001,
        "i_deg": 51.6,
        "argp_deg": 45.0,
        "raan_deg": 120.0,
        "anomaly_deg": 30.0,
        "anomaly_type": "MEAN",
    },
    mu_m3_s2="WGS84",
    mass_kg=450.0,
)

# Numerical propagator configuration (typed spec)
spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=450.0,
    integrator=IntegratorSpec(
        kind="dp853",
        min_step=1e-6,
        max_step=300.0,
        position_tolerance=1e-3,
    ),
    force_specs=[
        GravitySpec(degree=8, order=8),
    ],
)

propagator = app.propagation.build_propagator_from_state(initial_state, spec)

start_date = app.state.to_orekit_date(initial_state.epoch)
target_date = start_date.shiftedBy(3600.0)  # +1 hour
state = propagator.propagate(target_date)

orbit = state.getOrbit()
pos = state.getPVCoordinates(FramesFactory.getGCRF()).getPosition()

print(f"a = {orbit.getA()/1e3:.2f} km")
print(f"e = {orbit.getE():.6f}")
print(f"i = {math.degrees(orbit.getI()):.3f} deg")
print(f"r = ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f}) km")
```

## What this demonstrates

- `OrbitStateRecord` as a user-facing initial condition format
- typed numerical propagation config via `PropagatorSpec`
- facade method `build_propagator_from_state(...)`
- propagation to a target epoch and basic state inspection

## Why this pattern is useful for research workflows

This is a good baseline for student/research code because:

- the initial condition is explicit and serializable
- the numerical configuration is explicit and versionable
- the same `OrbitStateRecord` can later be reused in:
  - state files
  - mission scenarios
  - uncertainty workflows

## Extended repo examples (same ideas)

If you cloned the repo, see:

- `examples/quickstart.py --mode numerical`
- `examples/cookbook/multi_fidelity_comparison.py`
- `examples/cookbook/force_model_sweep.py`

## Next steps

- [Export a Trajectory to YAML/HDF5](export-trajectory-from-propagator.md)
- [Propagation Quickstart](../tutorials/propagation-quickstart.md)
- [Scenario + Mission Workflows](../tutorials/scenario-missions.md)
