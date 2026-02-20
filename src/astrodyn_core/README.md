# ASTRODYN-CORE Package Architecture

This folder contains the core Python package used by the examples and tests.

## High-level architecture

`astrodyn_core` is organized as layered modules:

1. **`data/`**
   - Bundled YAML presets and data assets (propagation models, spacecraft models, space-weather files).
   - Used to load ready-to-run configs without writing YAML from scratch.

2. **`propagation/`**
   - Typed propagation specs and Orekit-native provider/factory system.
   - Builds numerical/Keplerian/DSST/TLE builders and propagators.

3. **`states/`**
   - Typed scenario/state models + YAML/JSON/HDF5 I/O + Orekit conversion.
   - `StateFileClient` is the orchestration entry point for most workflows.

4. **`mission/`**
   - Scenario maneuver planning and detector-driven mission execution.
   - Includes mission profile plotting utilities.

5. **`uncertainty/`**
   - Covariance propagation around trajectories (STM today; unscented scaffold present).
   - Saves/loads covariance series in YAML/HDF5.

## Typical workflow paths

### Path A: propagation + trajectory export

- Build spec (`PropagatorSpec`) and context (`BuildContext`)
- Use `PropagatorFactory` + default providers
- Export trajectory via `StateFileClient.export_trajectory_from_propagator`

See: `examples/quickstart.py` (`keplerian`, `numerical`, `dsst`, `tle`, `plot` modes).

### Path B: scenario missions

- Load scenario YAML with `StateFileClient.load_state_file`
- Execute intent/event maneuvers via scenario helpers
- Use detector-driven execution for closed-loop timing validation

See: `examples/scenario_missions.py` (`io`, `inspect`, `intent`, `detector` modes).

### Path C: uncertainty propagation

- Configure `UncertaintySpec`
- Create STM covariance propagator from an Orekit propagator
- Propagate and export state + covariance series

See: `examples/uncertainty.py`.

## Public API aggregation

`astrodyn_core/__init__.py` re-exports major classes/functions from all modules, so users can start with:

```python
from astrodyn_core import PropagatorSpec, StateFileClient, UncertaintySpec
```

For module-level details, read the local `README.md` files in each subfolder.
