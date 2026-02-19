"""Detector-driven mission execution demo.

Demonstrates :class:`~astrodyn_core.mission.executor.ScenarioExecutor`:
closed-loop maneuver execution where Orekit EventDetectors trigger maneuvers
at physically accurate orbital events during numerical propagation.

Compare to ``demo_mission_maneuver_profile.py`` which uses the faster
Keplerian-approximation + propagation-replay approach.

Run with:
    conda run -n astrodyn-core-env python examples/demo_detector_mission.py
"""

from __future__ import annotations

from pathlib import Path

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir

setup_orekit_curdir()

from astrodyn_core import (
    BuildContext,
    IntegratorSpec,
    OutputEpochSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    StateFileClient,
    register_default_orekit_providers,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIO_FILE = Path("examples/state_files/leo_detector_mission.yaml")
OUTPUT_STATES = Path("examples/output/detector_mission_trajectory.yaml")
OUTPUT_PLOT = Path("examples/output/detector_mission_elements.png")
OUTPUT_STATES.parent.mkdir(parents=True, exist_ok=True)

# Output epochs: 6-hour simulation at 5-minute resolution
EPOCH_SPEC = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-19T06:00:00Z",
    step_seconds=300.0,
)

# ---------------------------------------------------------------------------
# Load scenario
# ---------------------------------------------------------------------------

client = StateFileClient()
scenario = client.load_state_file(SCENARIO_FILE)
initial_state = scenario.initial_state
assert initial_state is not None, "Scenario must have an initial_state."

print(f"Scenario: {scenario.metadata.get('mission', '?')}")
print(f"  Initial epoch : {initial_state.epoch}")
print(f"  SMA           : {initial_state.elements['a_m'] / 1e3:.3f} km")
print(f"  Eccentricity  : {initial_state.elements['e']:.5f}")
print(f"  Maneuvers     : {len(scenario.maneuvers)}")

# ---------------------------------------------------------------------------
# Build numerical propagator
# ---------------------------------------------------------------------------

ctx = BuildContext.from_state_record(initial_state, universe=scenario.universe)
registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=float(initial_state.mass_kg or 450.0),
    integrator=IntegratorSpec(
        kind="dp853",
        min_step=1.0e-6,
        max_step=300.0,
        position_tolerance=1.0e-3,
    ),
)
builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

# ---------------------------------------------------------------------------
# Run in detector mode
# ---------------------------------------------------------------------------

print("\n--- Running detector-driven propagation ---")
state_series, report = client.run_scenario_detector_mode(
    propagator,
    scenario,
    EPOCH_SPEC,
    representation="keplerian",
    output_path=OUTPUT_STATES,
)

# ---------------------------------------------------------------------------
# Print execution report
# ---------------------------------------------------------------------------

print(f"\nMission Execution Report")
print(f"  Propagation   : {report.propagation_start} → {report.propagation_end}")
print(f"  Events fired  : {len(report.events)}")
print(f"  Applied       : {len(report.applied_events())}")
print(f"  Skipped       : {len(report.skipped_events())}")
print(f"  Total Δv      : {report.total_dv_mps:.4f} m/s")

if report.events:
    print("\n  Event log:")
    for event in report.events:
        status = "APPLIED" if event.applied else f"SKIPPED ({event.guard_skip_reason})"
        dv_str = ""
        if event.dv_inertial_mps is not None:
            import math
            dv_mag = math.sqrt(sum(c * c for c in event.dv_inertial_mps))
            dv_str = f"  |Δv|={dv_mag:.4f} m/s"
        print(f"    [{event.epoch}] {event.maneuver_name} | {event.trigger_type} | {status}{dv_str}")

# ---------------------------------------------------------------------------
# Plot orbital elements
# ---------------------------------------------------------------------------

if len(state_series.states) > 0:
    try:
        plot_path = client.plot_orbital_elements(state_series, OUTPUT_PLOT)
        print(f"\nOrbital elements plot: {plot_path}")
    except Exception as exc:
        print(f"\n[Warning] Could not generate plot: {exc}")

print(f"\nTrajectory saved to: {OUTPUT_STATES}")
print("Done.")
