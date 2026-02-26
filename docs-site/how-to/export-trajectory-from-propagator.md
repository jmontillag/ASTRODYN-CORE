# How-To: Export a Trajectory from a Propagator (YAML / HDF5)

This recipe shows how to:

1. build a propagator,
2. define an output epoch grid,
3. export a trajectory to a serialized file, and
4. load it back as a `StateSeries`.

This is a key workflow in ASTRODYN-CORE because it connects propagation to:

- mission analysis
- plotting
- state-file pipelines
- uncertainty/ephemeris downstream workflows

## Prerequisites

- `astrodyn-core` installed
- Orekit data configured

See:

- [Orekit Data Setup](../getting-started/orekit-data.md)
- [Numerical Propagation from a Typed Initial State](numerical-propagation-from-state.md)

## Run (copy-paste, YAML export)

```python
from pathlib import Path

import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from astrodyn_core import (
    AstrodynClient,
    GravitySpec,
    IntegratorSpec,
    OrbitStateRecord,
    OutputEpochSpec,
    PropagatorKind,
    PropagatorSpec,
)

app = AstrodynClient()

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

spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=450.0,
    integrator=IntegratorSpec(
        kind="dp853",
        min_step=1e-6,
        max_step=300.0,
        position_tolerance=1e-3,
    ),
    force_specs=[GravitySpec(degree=8, order=8)],
)

propagator = app.propagation.build_propagator_from_state(initial_state, spec)

epoch_spec = OutputEpochSpec(
    start_epoch=initial_state.epoch,
    end_epoch="2026-02-19T03:00:00Z",
    step_seconds=120.0,
)

output_path = Path("trajectory_demo.yaml")

saved_path = app.state.export_trajectory_from_propagator(
    propagator,
    epoch_spec,
    output_path,
    series_name="demo-trajectory",
    representation="keplerian",
    frame="GCRF",
)

series = app.state.load_state_series(saved_path)

print(f"Saved trajectory: {saved_path}")
print(f"Series name: {series.name}")
print(f"Samples: {len(series.states)}")
print(f"First epoch: {series.states[0].epoch}")
print(f"Last epoch:  {series.states[-1].epoch}")
```

## What this demonstrates

- `OutputEpochSpec` for defining the sampling grid
- `app.state.export_trajectory_from_propagator(...)`
- serialized trajectory output (`.yaml`)
- reloading exported output with `app.state.load_state_series(...)`

## Optional: HDF5 export (if `h5py` is installed)

Change the output path suffix to `.h5`:

```python
output_path = Path("trajectory_demo.h5")
```

The same export call will write HDF5, and `load_state_series(...)` will load it
back automatically based on the file suffix.

## Why this workflow matters for students and research users

Exported trajectories make experiments reproducible:

- you can archive exact sampled states used for plots/analysis
- you can compare methods (numerical vs DSST vs GEqOE) using a common output format
- you can version/share state series independently of the propagator build code

## Common pitfalls

- **Too-large `step_seconds` hides dynamics**
  Start with a finer cadence, then downsample later if needed.

- **Representation choice affects downstream convenience**
  - `representation="keplerian"` is often easier for orbital-element plots
  - `representation="cartesian"` may be better for external tool interop

- **Output files are not raw CSV**
  ASTRODYN-CORE writes structured state-series formats (YAML/JSON/HDF5), which
  preserve metadata and are intended for round-trip use with the library.

## Extended repo examples (same workflow family)

If you cloned the repo, see:

- `examples/quickstart.py --mode plot`
- `examples/scenario_missions.py --mode io`
- `examples/cookbook/multi_fidelity_comparison.py`

## Next steps

- [Scenario + Mission Workflows](../tutorials/scenario-missions.md)
- [Uncertainty Workflows](../tutorials/uncertainty.md)
- [How-To / Cookbook](index.md)
