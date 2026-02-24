# ASTRODYN-CORE

Builder-first astrodynamics tooling that keeps Orekit APIs first-class while adding typed configuration, state-file workflows, and mission-profile helpers.

## Current status (2026-02-24)

Implemented:

- Orekit-native propagation providers: `numerical`, `keplerian`, `dsst`, `tle`
- GEqOE J2 Taylor-series propagator (Python implementation with numpy engine, Orekit adapter, and provider integration)
- Extensible registry for custom/analytical propagators (any string kind)
- Typed propagation configuration and registry/factory construction flow
- Force model, attitude, and spacecraft declarative assembly
- Unified client APIs: `PropagationClient`, `StateFileClient`, `TLEClient`, `MissionClient`, `UncertaintyClient`, `EphemerisClient`
- App-level facade (`AstrodynClient`) composing all domain workflows
- State I/O: YAML/JSON/HDF5, compact series format, Orekit object conversion
- OEM/OCM/SP3/CPF ephemeris-based propagation
- Scenario maneuver tooling: timeline events, detector-driven execution, intent maneuvers
- STM-based covariance propagation
- Root API organized into three tiers (facade, models, advanced)
- Import hygiene enforced in tests
- Comprehensive examples: quickstart, scenario/mission workflows, uncertainty, multi-fidelity comparisons, force model sweep, OEM parsing, SMA maintenance analysis

Architecture is frozen and ready for extension with custom propagators.

In progress:

- GEqOE J2 Taylor propagator C++ implementation and staged parity testing

Next:

- CI pipeline
- Unscented Transform covariance propagation

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
python examples/geqoe_propagator.py --mode all
```

### Cookbook

Self-contained topical examples in `examples/cookbook/`:

- `multi_fidelity_comparison.py` — Compare Keplerian, DSST, and numerical propagation fidelity
- `orbit_comparison.py` — Cartesian to Keplerian round-trip verification
- `force_model_sweep.py` — Gravity field degree/order convergence analysis
- `ephemeris_from_oem.py` — OEM file parse and ephemeris round-trip
- `sma_maintenance_analysis.py` — Full SMA maintenance mission workflow (scenario → detector execution → analysis)

Parity/comparison tools:

- `geqoe_cpp_order1_parity.py` — Validate C++ staged GEqOE against Python reference
- `geqoe_legacy_vs_staged.py` — Compare legacy vs optimized GEqOE implementations
- `math_cpp_comparison.py` — Verify C++ math utils against Python equivalents

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

- Implementation plan and architecture: `docs/implementation-plan.md`
- API governance and boundary policy: `docs/api-governance.md`
- Extension guide for custom propagators: `docs/extending-propagators.md`
- Architecture hardening and finalization roadmap: `docs/architecture-hardening-plan.md`
