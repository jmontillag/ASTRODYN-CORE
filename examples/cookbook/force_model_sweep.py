#!/usr/bin/env python
"""Sweep gravity field degree/order to show convergence.

Propagates the same orbit with increasing gravity field resolution
(2x2, 4x4, 8x8, 16x16, 32x32) and measures position drift relative
to the highest-fidelity run.  Demonstrates force model specification
and how gravity field truncation affects orbit prediction accuracy.

Run with:
    conda run -n astrodyn-core-env python examples/cookbook/force_model_sweep.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import init_orekit, make_leo_orbit

init_orekit()

from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    GravitySpec,
    IntegratorSpec,
    PropagatorKind,
    PropagatorSpec,
    SpacecraftSpec,
)

app = AstrodynClient()
orbit, epoch, frame = make_leo_orbit()

# 12-hour propagation
target = epoch.shiftedBy(12.0 * 3600.0)

degrees = [2, 4, 8, 16, 32]
positions = {}

print("=" * 72)
print("  Gravity Field Degree/Order Convergence")
print("=" * 72)
print(f"  Orbit: a={orbit.getA()/1e3:.1f} km, i={math.degrees(orbit.getI()):.1f} deg")
print(f"  Duration: 12 hours")
print()

for deg in degrees:
    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        spacecraft=SpacecraftSpec(mass=450.0),
        integrator=IntegratorSpec(kind="dp853", min_step=1e-6, max_step=300.0, position_tolerance=1e-6),
        force_specs=[GravitySpec(degree=deg, order=deg)],
    )
    builder = app.propagation.build_builder(spec, BuildContext(initial_orbit=orbit))
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    state = propagator.propagate(target)
    pos = state.getPVCoordinates(frame).getPosition()
    positions[deg] = (pos.getX(), pos.getY(), pos.getZ())

# Use highest degree as reference
ref = positions[degrees[-1]]
print(f"  {'Degree':>6}  {'Order':>6}  {'|dr| vs {0}x{0} (m)'.format(degrees[-1]):>22}")
print("  " + "-" * 40)

for deg in degrees:
    pos = positions[deg]
    dx = pos[0] - ref[0]
    dy = pos[1] - ref[1]
    dz = pos[2] - ref[2]
    dr = math.sqrt(dx**2 + dy**2 + dz**2)
    marker = " (reference)" if deg == degrees[-1] else ""
    print(f"  {deg:>6}  {deg:>6}  {dr:>22.3f}{marker}")

print()
print("  Higher gravity degree/order reduces position error at the cost")
print("  of computation time.  8x8 is typically sufficient for LEO.")
print("Done.")
