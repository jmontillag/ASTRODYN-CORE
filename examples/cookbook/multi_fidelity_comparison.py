#!/usr/bin/env python
"""Compare Keplerian, DSST, and numerical propagation for the same orbit.

Demonstrates how propagation fidelity affects orbit prediction by running
three propagators from the same initial state and comparing the resulting
positions after 1 day.

Outputs
-------
  examples/generated/fidelity_comparison.yaml
  examples/generated/fidelity_comparison.png

Run with:
    conda run -n astrodyn-core-env python examples/cookbook/multi_fidelity_comparison.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow importing _common from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import init_orekit, make_generated_dir, make_leo_orbit

init_orekit()

from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    GravitySpec,
    IntegratorSpec,
    OutputEpochSpec,
    PropagatorKind,
    PropagatorSpec,
    SpacecraftSpec,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = AstrodynClient()
orbit, epoch, frame = make_leo_orbit()
out_dir = make_generated_dir()

period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
n_orbits = 16  # ~1 day for LEO
end_date = epoch.shiftedBy(n_orbits * period_s)

epoch_spec = OutputEpochSpec(
    start_epoch=app.state.from_orekit_date(epoch),
    end_epoch=app.state.from_orekit_date(end_date),
    step_seconds=120.0,
)

# ---------------------------------------------------------------------------
# Three propagators at different fidelity levels
# ---------------------------------------------------------------------------

configs = {
    "keplerian": PropagatorSpec(kind=PropagatorKind.KEPLERIAN, mass_kg=450.0),
    "dsst": PropagatorSpec(
        kind=PropagatorKind.DSST,
        spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0, srp_area=5.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-3, max_step=300.0, position_tolerance=1e-3),
        dsst_propagation_type="OSCULATING",
        dsst_state_type="OSCULATING",
        force_specs=[GravitySpec(degree=8, order=8)]
    ),
    "numerical": PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0, drag_area=5.0, srp_area=5.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-3),
        force_specs=[GravitySpec(degree=8, order=8)]
    ),
}

print("=" * 72)
print("  Multi-Fidelity Propagation Comparison")
print("=" * 72)
print(f"  Duration: {n_orbits} orbits ({n_orbits * period_s / 3600:.1f} hours)")
print()

results = {}
for name, spec in configs.items():
    ctx = BuildContext(initial_orbit=orbit)

    # First propagator: get final-state position for comparison
    builder = app.propagation.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    final_state = propagator.propagate(end_date)
    pos = final_state.getPVCoordinates(frame).getPosition()
    results[name] = (pos.getX(), pos.getY(), pos.getZ())
    print(f"  {name:12s}: x={pos.getX()/1e3:+11.3f} km  y={pos.getY()/1e3:+11.3f} km  z={pos.getZ()/1e3:+11.3f} km")

    # Fresh propagator for trajectory export (propagate() above consumes
    # the ephemeris generator, so a second builder is needed)
    builder2 = app.propagation.build_builder(spec, ctx)
    propagator2 = builder2.buildPropagator(builder2.getSelectedNormalizedParameters())
    out_path = out_dir / f"fidelity_{name}_series.yaml"
    app.state.export_trajectory_from_propagator(
        propagator2, epoch_spec, out_path,
        series_name=f"fidelity-{name}",
        representation="keplerian", frame="GCRF",
    )

# Compute differences relative to numerical
print()
print("  Differences from numerical (km):")
ref = results["numerical"]
for name in ("keplerian", "dsst"):
    pos = results[name]
    dx = (pos[0] - ref[0]) / 1e3
    dy = (pos[1] - ref[1]) / 1e3
    dz = (pos[2] - ref[2]) / 1e3
    total = math.sqrt(dx**2 + dy**2 + dz**2)
    print(f"  {name:12s}: |dr| = {total:.3f} km")

# Plot numerical trajectory
series = app.state.load_state_series(out_dir / "fidelity_numerical_series.yaml")
plot_path = out_dir / "fidelity_comparison.png"
app.mission.plot_orbital_elements_series(series, plot_path, title="Numerical Propagation: Orbital Elements")
print(f"\n  Plot: {plot_path}")
print("Done.")
