#!/usr/bin/env python
"""Analyze a semimajor-axis maintenance mission profile.

Loads a maintenance timeline scenario, executes it in detector mode, and
analyzes the resulting maneuver cadence, total delta-v budget, and orbital
element evolution.  This demonstrates the full mission workflow:
scenario file -> detector execution -> analysis.

Run with:
    conda run -n astrodyn-core-env python examples/cookbook/sma_maintenance_analysis.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import init_orekit, make_generated_dir

init_orekit()

from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    IntegratorSpec,
    OutputEpochSpec,
    PropagatorKind,
    PropagatorSpec,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = AstrodynClient()
base = Path(__file__).resolve().parent.parent
out_dir = make_generated_dir()

scenario_path = base / "state_files" / "leo_sma_maintenance_timeline.yaml"
scenario = app.state.load_state_file(scenario_path)

if scenario.initial_state is None:
    raise RuntimeError("Scenario requires initial_state.")

print("=" * 72)
print("  SMA Maintenance Mission Analysis")
print("=" * 72)
print(f"  Scenario: {scenario_path.name}")
print(f"  Maneuver definitions: {len(scenario.maneuvers)}")
print(f"  Timeline events: {len(scenario.timeline)}")

# Build propagator
mass = float(scenario.initial_state.mass_kg or 450.0)
spec = PropagatorSpec(
    kind=PropagatorKind.NUMERICAL,
    mass_kg=mass,
    integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-3),
)
ctx = app.propagation.context_from_state(scenario.initial_state, universe=scenario.universe)
builder = app.propagation.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

# Execute in detector mode over 12 hours
epoch_spec = OutputEpochSpec(
    start_epoch="2026-02-19T00:00:00Z",
    end_epoch="2026-02-19T12:00:00Z",
    step_seconds=120.0,
)

state_series, report = app.mission.run_scenario_detector_mode(
    propagator, scenario, epoch_spec,
    representation="keplerian",
)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

print(f"\n  Execution Results:")
print(f"    Propagation steps: {len(state_series.states)}")
print(f"    Events fired:     {len(report.events)}")
print(f"    Applied:          {len(report.applied_events())}")
print(f"    Skipped:          {len(report.skipped_events())}")
print(f"    Total delta-v:    {report.total_dv_mps:.4f} m/s")

if report.applied_events():
    print(f"\n  Applied maneuvers:")
    for event in report.applied_events():
        print(f"    [{event.epoch}] {event.maneuver_name} ({event.trigger_type})")

if report.skipped_events():
    print(f"\n  Skip reasons:")
    reasons = Counter(e.guard_skip_reason or "unspecified" for e in report.skipped_events())
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {count}x {reason}")

# SMA statistics from trajectory
if state_series.states:
    sma_values = []
    for s in state_series.states:
        if s.elements and "a_m" in s.elements:
            sma_values.append(s.elements["a_m"])
        elif s.elements and "a" in s.elements:
            sma_values.append(s.elements["a"])

    if sma_values:
        sma_min = min(sma_values) / 1e3
        sma_max = max(sma_values) / 1e3
        sma_range = sma_max - sma_min
        print(f"\n  SMA Statistics:")
        print(f"    Min: {sma_min:.3f} km")
        print(f"    Max: {sma_max:.3f} km")
        print(f"    Range: {sma_range * 1e3:.1f} m")

# Save outputs
out_plot = out_dir / "sma_maintenance_elements.png"
app.state.save_state_series(out_dir / "sma_maintenance_trajectory.yaml", state_series)
app.mission.plot_orbital_elements_series(state_series, out_plot, title="SMA Maintenance: Orbital Elements")
print(f"\n  Saved plot: {out_plot}")
print("Done.")
