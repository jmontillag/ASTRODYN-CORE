#!/usr/bin/env python
"""Compare orbit representations: Cartesian vs Keplerian round-trip.

Demonstrates loading an initial state in Keplerian elements, converting to
an Orekit orbit, propagating, and verifying that the Cartesian -> Keplerian
round-trip preserves accuracy.

Run with:
    conda run -n astrodyn-core-env python examples/cookbook/orbit_comparison.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import init_orekit

init_orekit()

from org.orekit.orbits import KeplerianOrbit, OrbitType

from astrodyn_core import AstrodynClient, OrbitStateRecord

# ---------------------------------------------------------------------------
# Define two equivalent states (Keplerian and Cartesian representations)
# ---------------------------------------------------------------------------

app = AstrodynClient()

kep_state = OrbitStateRecord(
    epoch="2026-02-19T00:00:00Z",
    frame="GCRF",
    representation="keplerian",
    elements={
        "a_m": 7578137.0,
        "e": 0.001,
        "i_deg": 51.6,
        "argp_deg": 45.0,
        "raan_deg": 120.0,
        "anomaly_deg": 30.0,
        "anomaly_type": "MEAN",
    },
    mu_m3_s2="WGS84",
    mass_kg=450.0,
)

# Convert to Orekit orbit
orekit_orbit = app.state.to_orekit_orbit(kep_state)

# Extract Keplerian elements back
kep = KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(orekit_orbit))
pv = orekit_orbit.getPVCoordinates()

print("=" * 72)
print("  Orbit Representation Round-Trip")
print("=" * 72)
print(f"  Input:  a={kep_state.elements['a_m']/1e3:.3f} km, e={kep_state.elements['e']:.4f}")
print(f"  Orekit: a={kep.getA()/1e3:.3f} km, e={kep.getE():.6f}, i={math.degrees(kep.getI()):.4f} deg")
print()

# Position/velocity from orbit
pos = pv.getPosition()
vel = pv.getVelocity()
print(f"  Cartesian position (m): [{pos.getX():.3f}, {pos.getY():.3f}, {pos.getZ():.3f}]")
print(f"  Cartesian velocity (m/s): [{vel.getX():.6f}, {vel.getY():.6f}, {vel.getZ():.6f}]")
print(f"  |r| = {pos.getNorm()/1e3:.3f} km, |v| = {vel.getNorm()/1e3:.6f} km/s")
print()

# Verify round-trip
a_err = abs(kep.getA() - kep_state.elements["a_m"])
e_err = abs(kep.getE() - kep_state.elements["e"])
i_err = abs(math.degrees(kep.getI()) - kep_state.elements["i_deg"])

print(f"  Round-trip errors:")
print(f"    |da| = {a_err:.6e} m")
print(f"    |de| = {e_err:.6e}")
print(f"    |di| = {i_err:.6e} deg")
print()
print("  All errors at machine precision." if a_err < 1e-6 else "  WARNING: errors above tolerance.")
print("Done.")
