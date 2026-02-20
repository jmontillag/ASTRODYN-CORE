# `mission` module

Scenario maneuver planning, execution, and mission-level analysis helpers.

## Purpose

The `mission` module consumes scenario definitions from `states` and applies mission logic on top of propagators.

It supports two mission-execution styles:

1. **Compiled/approximate flow** (`maneuvers.py`)
   - resolve timeline + maneuver triggers with Keplerian approximation,
   - compile inertial delta-v events,
   - replay impulses while sampling trajectory.

2. **Detector-driven closed-loop flow** (`executor.py` + `detectors.py`)
   - attach Orekit event detectors to a numerical propagator,
   - fire maneuvers at physically detected events,
   - produce execution report with applied/skipped events and total Δv.

## Key API

- `compile_scenario_maneuvers(...)`
- `simulate_scenario_series(...)`
- `export_scenario_series(...)`
- `ScenarioExecutor`
- `MissionExecutionReport`, `ManeuverFiredEvent`
- `plot_orbital_elements_series(...)`

## Intended use cases

1. **Rapid mission design iterations**
   - compile and inspect planned impulses quickly.
2. **Operational-style timing validation**
   - run detector-driven execution for higher-fidelity trigger timing.
3. **Mission reporting and visualization**
   - summarize applied/skipped events and visualize orbital element evolution.

See `examples/scenario_missions.py`:
- `intent` mode (compiled/replay flow),
- `detector` mode (closed-loop detector-driven flow),
- `inspect` mode (scenario metadata and timeline inspection).

## Boundaries

- Mission logic assumes propagators are already built (from `propagation`).
- Scenario schema and persistence remain in `states`.
