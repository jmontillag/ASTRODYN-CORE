# ASTRODYN-CORE

Builder-first astrodynamics tooling that keeps Orekit APIs first-class while adding typed configuration, state-file workflows, and mission-profile helpers.

## Current status (2026-02-19)

Implemented:

- Orekit-native propagation providers: `numerical`, `keplerian`, `dsst`, `tle`
- Typed propagation configuration and registry/factory construction flow
- Force model, attitude, and spacecraft declarative assembly
- Unified state-file client API (`StateFileClient`) for YAML/JSON/HDF5 state workflows
- Unified TLE client API (`TLEClient`) for cache/download/parse/resolve workflows
- Unified mission client API (`MissionClient`) for planning/execution/plot workflows
- Unified uncertainty client API (`UncertaintyClient`) for covariance propagation/I/O workflows
- App-level façade (`AstrodynClient`) composing `state` + `mission` + `uncertainty` + `tle` workflows
- Conversion between state files and Orekit objects, including state-series to Orekit ephemeris
- Scenario maneuver tooling:
  - timeline events (`epoch`, `elapsed`, `apogee/perigee`, node triggers)
  - event-referenced maneuver triggers
  - intent maneuvers with fast Keplerian solving (`raise_perigee`, `raise_semimajor_axis`, `maintain_semimajor_axis_above`, `change_inclination`)
  - increment and absolute target support (for raise intents)
- Orbital-element mission plotting to PNG

In progress / next:

- detector-driven maneuver execution in the numerical propagation loop (closed-loop mission control)
- richer timeline semantics for recurrence, windows, and event dependencies
- expanded mission schema and validation hardening

## Design principles

- Orekit-native semantics stay visible: providers return real Orekit builders/propagators.
- Builder-first API: `PropagatorSpec` drives provider selection and construction.
- Registry-based extensibility: new providers can be plugged in without editing core factory logic.
- Single-repo architecture for now (propagation + state/mission workflows).

## Quick start

Environment setup (recommended):

```bash
python setup_env.py
conda activate astrodyn-core-env
```

```python
from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    StateFileClient,
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

ctx = BuildContext(initial_orbit=initial_orbit, position_tolerance=10.0)
builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

states = StateFileClient()
```

## Examples

Run from the project root:

```bash
python examples/quickstart.py --mode all
python examples/scenario_missions.py --mode all
python examples/uncertainty.py
```

State-file examples:

- `examples/state_files/leo_initial_state.yaml`
- `examples/state_files/leo_state_series.yaml`
- `examples/state_files/leo_mission_timeline.yaml`
- `examples/state_files/leo_intent_mission.yaml`
- `examples/state_files/leo_detector_mission.yaml`
- `examples/state_files/leo_sma_maintenance_timeline.yaml`

Generated artifacts from examples are written to:

- `examples/generated/`

## Planning docs

- Long-term implementation plan: `docs/implementation-plan.md`
- Current architecture snapshot: `docs/phase1-architecture.md`
- State I/O and scenario schema plan: `docs/state-io-design.md`
