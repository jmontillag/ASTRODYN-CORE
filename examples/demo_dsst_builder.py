#!/usr/bin/env python
"""Minimal DSST builder example.

Run from repo root:
    python examples/demo_dsst_builder.py
"""

from __future__ import annotations

import math

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import KeplerianOrbit, PositionAngleType  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants  # noqa: E402

from astrodyn_core import (  # noqa: E402
    BuildContext,
    IntegratorSpec,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    register_default_orekit_providers,
)


def make_initial_orbit():
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)
    frame = FramesFactory.getGCRF()
    mu = Constants.WGS84_EARTH_MU

    orbit = KeplerianOrbit(
        7_200_000.0,
        0.01,
        math.radians(63.4),
        math.radians(5.0),
        math.radians(45.0),
        math.radians(20.0),
        PositionAngleType.MEAN,
        frame,
        epoch,
        mu,
    )
    return orbit, epoch, frame


orbit, epoch, frame = make_initial_orbit()

registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

spec = PropagatorSpec(
    kind=PropagatorKind.DSST,
    mass_kg=550.0,
    integrator=IntegratorSpec(kind="dp853", min_step=1.0e-3, max_step=300.0, position_tolerance=10.0),
    dsst_propagation_type="MEAN",
    dsst_state_type="OSCULATING",
)
ctx = BuildContext(initial_orbit=orbit)

builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

target_date = epoch.shiftedBy(2.0 * 3600.0)
state = propagator.propagate(target_date)
pos = state.getPVCoordinates(frame).getPosition()

print("Built:", builder.getClass().getSimpleName())
print("Propagated:", propagator.getClass().getSimpleName())
print(f"Position after 2 hours (km): ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})")
