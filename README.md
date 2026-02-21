# ASTRODYN-CORE

Builder-first astrodynamics tooling that keeps Orekit APIs first-class while adding typed configuration, state-file workflows, and mission-profile helpers.

## Current status (2026-02-21)

Implemented:

- Orekit-native propagation providers: `numerical`, `keplerian`, `dsst`, `tle`
- Typed propagation configuration and registry/factory construction flow
- Force model, attitude, and spacecraft declarative assembly
- Unified state-file client API (`StateFileClient`) for YAML/JSON/HDF5 state workflows
- Unified propagation client API (`PropagationClient`) for factory/context/builder workflows
- Unified TLE client API (`TLEClient`) for cache/download/parse/resolve workflows
- Unified mission client API (`MissionClient`) for planning/execution/plot workflows
- Unified uncertainty client API (`UncertaintyClient`) for covariance propagation/I/O workflows
- Unified ephemeris client API (`EphemerisClient`) for OEM/OCM/SP3/CPF ephemeris-based propagation
- App-level facade (`AstrodynClient`) composing `propagation` + `state` + `mission` + `uncertainty` + `tle` + `ephemeris` workflows
- Conversion between state files and Orekit objects, including state-series to Orekit ephemeris
- Scenario maneuver tooling:
  - timeline events (`epoch`, `elapsed`, `apogee/perigee`, node triggers)
  - event-referenced maneuver triggers
  - intent maneuvers with fast Keplerian solving (`raise_perigee`, `raise_semimajor_axis`, `maintain_semimajor_axis_above`, `change_inclination`)
  - increment and absolute target support (for raise intents)
- Orbital-element mission plotting to PNG

Completed governance:

- Phase D: deprecated code removed, ephemeris module added (complete)
- Phase C: API governance and boundary hardening (complete)
- Root API organized into three tiers (facade, models, advanced)
- Import hygiene enforced in tests

Next:

- Implement Unscented Transform covariance propagation
- Phase 3 source-spec lane and interoperability
- CI pipeline

## Design principles

- Orekit-native semantics stay visible: providers return real Orekit builders/propagators.
- Builder-first API: `PropagatorSpec` drives provider selection and construction.
- Registry-based extensibility: new providers can be plugged in without editing core factory logic.
- Single-repo architecture for now (propagation + state/mission workflows).

## API tiers (recommended)

Use one of two API tiers based on your goal:

1. **Stable facade tier (recommended for most users)**
    - Start with `AstrodynClient`
    - Use domain facades via `app.propagation`, `app.state`, `app.mission`, `app.uncertainty`, `app.tle`, `app.ephemeris`
    - Preferred for notebooks, scripts, and long-lived user code

2. **Advanced low-level tier (power users)**
    - Use `PropagatorFactory`, `ProviderRegistry`, `BuildContext`, typed specs, and assembly helpers directly
    - Best when you need fine-grained Orekit-native control

Compatibility note:
- New features should appear in the façade tier first when practical.
- Low-level APIs remain supported for expert workflows.

## Quick start

Environment setup (recommended):

```bash
python setup_env.py
conda activate astrodyn-core-env
```

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
    mass_kg=1200.0,
    integrator=IntegratorSpec(
        kind="dormand_prince_853",
        min_step=0.001,
        max_step=300.0,
        position_tolerance=10.0,
    ),
)

ctx = BuildContext(initial_orbit=initial_orbit, position_tolerance=10.0)
builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

states = app.state
```

## Examples

Run from the project root:

```bash
python examples/quickstart.py --mode all
python examples/scenario_missions.py --mode all
python examples/uncertainty.py
```

### Cookbook

Self-contained topical examples in `examples/cookbook/`:

- `multi_fidelity_comparison.py` — Compare Keplerian, DSST, and numerical propagation fidelity
- `orbit_comparison.py` — Cartesian to Keplerian round-trip verification
- `force_model_sweep.py` — Gravity field degree/order convergence analysis
- `ephemeris_from_oem.py` — OEM file parse and ephemeris round-trip
- `sma_maintenance_analysis.py` — Full SMA maintenance mission workflow (scenario -> detector execution -> analysis)

### State-file examples

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
- API governance and deprecation policy: `docs/phasec-api-governance.md`
- Maintainability roadmap: `docs/maintainability-cleanup-roadmap.md`
