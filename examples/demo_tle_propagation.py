#!/usr/bin/env python
"""Minimal TLE propagator example.

Run from repo root:
    python examples/demo_tle_propagation.py
"""

from __future__ import annotations

import math

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402

from astrodyn_core import (  # noqa: E402
    BuildContext,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    TLESpec,
    register_default_orekit_providers,
)


tle = TLESpec(
    line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002",
    line2="2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
)

registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

spec = PropagatorSpec(kind=PropagatorKind.TLE, tle=tle)
ctx = BuildContext()

propagator = factory.build_propagator(spec, ctx)

utc = TimeScalesFactory.getUTC()
start = AbsoluteDate(2024, 1, 1, 12, 0, 0.0, utc)
state = propagator.propagate(start.shiftedBy(45.0 * 60.0))

frame = FramesFactory.getGCRF()
pos = state.getPVCoordinates(frame).getPosition()
orbit = state.getOrbit()

print("Propagated:", propagator.getClass().getSimpleName())
print(f"Semi-major axis (km): {orbit.getA()/1e3:.2f}")
print(f"Eccentricity: {orbit.getE():.6f}")
print(f"Inclination (deg): {math.degrees(orbit.getI()):.3f}")
print(f"Position after 45 min (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")
