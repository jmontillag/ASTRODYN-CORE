# ASTRODYN-CORE Consumer Patterns

Use these patterns in dependent repositories that install
`/home/astror/Projects/ASTRODYN-CORE` as an editable dependency.

## 1) Numerical propagation (builder lane)

```python
from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    IntegratorSpec,
    PropagatorKind,
    PropagatorSpec,
)

app = AstrodynClient()

spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=600.0,
    integrator=IntegratorSpec(
        kind="dormand_prince_853",
        min_step=1.0e-3,
        max_step=300.0,
        position_tolerance=10.0,
    ),
)

ctx = BuildContext(initial_orbit=initial_orbit, position_tolerance=10.0)
builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

state = propagator.propagate(target_date)
```

## 2) Load a packaged dynamics model

```python
from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    SpacecraftSpec,
    get_propagation_model,
    load_dynamics_config,
)

app = AstrodynClient()
spec = load_dynamics_config(get_propagation_model("high_fidelity"))
spec = spec.with_spacecraft(SpacecraftSpec(mass=500.0, drag_area=4.0, srp_area=4.0))

ctx = BuildContext(initial_orbit=initial_orbit)
builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
```

## 3) TLE propagation (direct propagator lane)

```python
from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    PropagatorKind,
    PropagatorSpec,
    TLESpec,
)

app = AstrodynClient()

tle = TLESpec(line1=tle_line1, line2=tle_line2)
spec = PropagatorSpec(kind=PropagatorKind.TLE, tle=tle)
propagator = app.propagation.build_propagator(spec, BuildContext())
state = propagator.propagate(target_date)
```

## 4) Resolve TLE by NORAD + epoch, then propagate

```python
from datetime import datetime, timezone

from astrodyn_core import AstrodynClient, BuildContext, PropagatorKind, PropagatorSpec

app = AstrodynClient(tle_base_dir="data/tle", tle_allow_download=False)

query = app.tle.build_query(
    norad_id=25544,
    target_epoch=datetime.now(timezone.utc),
)
tle_spec = app.tle.resolve_tle_spec(query)

propagator = app.propagation.build_propagator(
    PropagatorSpec(kind=PropagatorKind.TLE, tle=tle_spec),
    BuildContext(),
)
```

## 5) Build context from serialized state

```python
from astrodyn_core import AstrodynClient, PropagatorKind, PropagatorSpec

app = AstrodynClient()
initial_record = app.state.load_initial_state("data/initial_state.yaml")

ctx = app.propagation.context_from_state(initial_record)
spec = PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0)
builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
```

## 6) Export trajectory to state-series file

```python
from astrodyn_core import AstrodynClient, OutputEpochSpec

app = AstrodynClient()

start_epoch = app.state.from_orekit_date(start_date)
end_epoch = app.state.from_orekit_date(end_date)
epoch_spec = OutputEpochSpec(
    start_epoch=start_epoch,
    end_epoch=end_epoch,
    step_seconds=60.0,
)

out_path = app.state.export_trajectory_from_propagator(
    propagator,
    epoch_spec,
    "outputs/trajectory.yaml",
    series_name="trajectory",
    representation="cartesian",
    frame="GCRF",
)

series = app.state.load_state_series(out_path)
```

## 7) Simulate scenario maneuvers and export

```python
from astrodyn_core import AstrodynClient, OutputEpochSpec

app = AstrodynClient()
scenario = app.state.load_state_file("data/scenario.yaml")

epoch_spec = OutputEpochSpec(
    start_epoch=scenario.initial_state.epoch,
    end_epoch="2026-01-01T03:00:00Z",
    step_seconds=60.0,
)

series, compiled = app.mission.simulate_scenario_series(
    propagator,
    scenario,
    epoch_spec,
    series_name="scenario-trajectory",
)

saved_path, compiled = app.mission.export_trajectory_from_scenario(
    propagator,
    scenario,
    epoch_spec,
    "outputs/scenario_series.yaml",
)
```

## 8) Propagate covariance with STM method

```python
import numpy as np

from astrodyn_core import AstrodynClient, OutputEpochSpec, UncertaintySpec

app = AstrodynClient()

initial_covariance = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
epoch_spec = OutputEpochSpec(
    start_epoch=start_epoch_utc,
    end_epoch=end_epoch_utc,
    step_seconds=120.0,
)

state_series, covariance_series = app.uncertainty.propagate_with_covariance(
    propagator,
    initial_covariance,
    epoch_spec,
    spec=UncertaintySpec(method="stm"),
    series_name="trajectory",
    covariance_name="covariance",
)

app.uncertainty.save_covariance_series("outputs/covariance.yaml", covariance_series)
```

## 9) Create propagator from OEM ephemeris

```python
from astrodyn_core import AstrodynClient

app = AstrodynClient()
ephem_propagator = app.ephemeris.create_propagator_from_oem("data/input.oem")
state = ephem_propagator.propagate(target_date)
```

## 10) Verification commands

```bash
conda run -n astror-env python -c "import astrodyn_core; print(astrodyn_core.__name__)"
conda run -n astror-env pytest -q tests/test_propagation*.py
```
