#!/usr/bin/env python
"""Propagate a LEO satellite and plot the orbit.

This script shows the typical ASTRODYN-CORE workflow:

  1. Pick a dynamics model from a bundled YAML preset
  2. Pick a spacecraft model from a bundled YAML preset
  3. Build the propagator through the factory
  4. Propagate and collect position samples
  5. Plot the ground-track and 3-D orbit

Run from the project root:
    python examples/demo_orbit_plot.py
"""

from __future__ import annotations

import math

# ── Orekit bootstrap ────────────────────────────────────────────────────────

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

# ── Orekit imports ──────────────────────────────────────────────────────────

from org.orekit.frames import FramesFactory  # noqa: E402
from org.orekit.orbits import KeplerianOrbit, PositionAngleType  # noqa: E402
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # noqa: E402
from org.orekit.utils import Constants  # noqa: E402
from org.orekit.bodies import OneAxisEllipsoid  # noqa: E402
from org.orekit.utils import IERSConventions  # noqa: E402

# ── ASTRODYN-CORE imports ──────────────────────────────────────────────────

from astrodyn_core import (  # noqa: E402
    BuildContext,
    PropagatorFactory,
    ProviderRegistry,
    get_propagation_model,
    get_spacecraft_model,
    load_dynamics_config,
    load_spacecraft_config,
    register_default_orekit_providers,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Define the initial orbit
# ═══════════════════════════════════════════════════════════════════════════

utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2024, 6, 15, 12, 0, 0.0, utc)
gcrf = FramesFactory.getGCRF()
mu = Constants.WGS84_EARTH_MU

initial_orbit = KeplerianOrbit(
    6_878_137.0,  # semi-major axis  (m)  ~500 km altitude
    0.0012,  # eccentricity
    math.radians(51.6),  # inclination       (ISS-like)
    math.radians(45.0),  # argument of perigee
    math.radians(120.0),  # RAAN
    math.radians(0.0),  # mean anomaly
    PositionAngleType.MEAN,
    gcrf,
    epoch,
    mu,
)

period_s = 2.0 * math.pi * math.sqrt(initial_orbit.getA() ** 3 / mu)
print(f"Orbital period : {period_s:.1f} s  ({period_s / 60:.1f} min)")
print(
    f"Altitude       : {(initial_orbit.getA() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 1e3:.1f} km"
)
print(f"Inclination    : {math.degrees(initial_orbit.getI()):.1f} deg")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load dynamics + spacecraft YAML presets
# ═══════════════════════════════════════════════════════════════════════════

preset_name = "high_fidelity"
spacecraft_name = "leo_smallsat"
spec = load_dynamics_config(get_propagation_model(preset_name))
spacecraft = load_spacecraft_config(get_spacecraft_model(spacecraft_name))
spec = spec.with_spacecraft(spacecraft)

print(f"\nDynamics model : {preset_name}")
print(f"Spacecraft cfg : {spacecraft_name}")
print(
    f"Spacecraft     : {spacecraft.mass} kg, Cd={spacecraft.drag_coeff}, Cr={spacecraft.srp_coeff}"
)
print(f"Force specs    : {len(spec.force_specs)}")
for fs in spec.force_specs:
    print(f"  - {type(fs).__name__}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Build the propagator
# ═══════════════════════════════════════════════════════════════════════════

registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

ctx = BuildContext(initial_orbit=initial_orbit)
builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

forces = propagator.getAllForceModels()
print(f"\nOrekit forces  : {[f.getClass().getSimpleName() for f in forces]}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Propagate for 3 orbits and sample positions
# ═══════════════════════════════════════════════════════════════════════════

n_orbits = 3
duration_s = n_orbits * period_s
dt = 30.0  # sample every 30 seconds

n_samples = int(duration_s / dt) + 1
print(f"\nPropagating {n_orbits} orbits ({duration_s / 3600:.2f} h), {n_samples} samples ...")

# Earth body for lat/lon conversion
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    itrf,
)

times_min = []
x_km, y_km, z_km = [], [], []
lat_deg, lon_deg, alt_km = [], [], []

for i in range(n_samples):
    t = i * dt
    state = propagator.propagate(epoch.shiftedBy(t))
    pv = state.getPVCoordinates(gcrf)
    pos = pv.getPosition()

    times_min.append(t / 60.0)
    x_km.append(pos.getX() / 1e3)
    y_km.append(pos.getY() / 1e3)
    z_km.append(pos.getZ() / 1e3)

    # Ground-track
    geo = earth.transform(pos, gcrf, state.getDate())
    lat_deg.append(math.degrees(geo.getLatitude()))
    lon_deg.append(math.degrees(geo.getLongitude()))
    alt_km.append(geo.getAltitude() / 1e3)

print(f"Done. Altitude range: {min(alt_km):.1f} – {max(alt_km):.1f} km")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Plot
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f"ASTRODYN-CORE  —  {preset_name} dynamics, {n_orbits} orbits",
    fontsize=14,
    fontweight="bold",
)

# ── 3-D orbit ───────────────────────────────────────────────────────────
ax3d = fig.add_subplot(2, 2, 1, projection="3d")

# Earth sphere
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
Re = Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1e3
xs = Re * np.outer(np.cos(u), np.sin(v))
ys = Re * np.outer(np.sin(u), np.sin(v))
zs = Re * np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_surface(xs, ys, zs, alpha=0.15, color="steelblue", edgecolor="none")

ax3d.plot(x_km, y_km, z_km, linewidth=0.8, color="orangered")
ax3d.scatter(x_km[0], y_km[0], z_km[0], s=40, c="green", zorder=5, label="Start")
ax3d.scatter(x_km[-1], y_km[-1], z_km[-1], s=40, c="red", marker="x", zorder=5, label="End")
ax3d.set_xlabel("X (km)")
ax3d.set_ylabel("Y (km)")
ax3d.set_zlabel("Z (km)")
ax3d.set_title("GCRF orbit")
ax3d.legend(fontsize=8)

# ── Ground track ────────────────────────────────────────────────────────
ax_gt = fig.add_subplot(2, 2, 2)
ax_gt.scatter(lon_deg, lat_deg, s=1, c=times_min, cmap="plasma")
ax_gt.set_xlim(-180, 180)
ax_gt.set_ylim(-90, 90)
ax_gt.set_xlabel("Longitude (deg)")
ax_gt.set_ylabel("Latitude (deg)")
ax_gt.set_title("Ground track")
ax_gt.grid(True, alpha=0.3)
ax_gt.set_aspect("equal")

# ── Altitude vs time ───────────────────────────────────────────────────
ax_alt = fig.add_subplot(2, 2, 3)
ax_alt.plot(times_min, alt_km, linewidth=0.8, color="teal")
ax_alt.set_xlabel("Time (min)")
ax_alt.set_ylabel("Altitude (km)")
ax_alt.set_title("Altitude profile")
ax_alt.grid(True, alpha=0.3)

# ── Velocity magnitude vs time ─────────────────────────────────────────
ax_vel = fig.add_subplot(2, 2, 4)

vel_kms = []
propagator2 = builder.buildPropagator(builder.getSelectedNormalizedParameters())
for i in range(n_samples):
    t = i * dt
    state = propagator2.propagate(epoch.shiftedBy(t))
    vel = state.getPVCoordinates(gcrf).getVelocity()
    v_mag = math.sqrt(vel.getX() ** 2 + vel.getY() ** 2 + vel.getZ() ** 2) / 1e3
    vel_kms.append(v_mag)

ax_vel.plot(times_min, vel_kms, linewidth=0.8, color="coral")
ax_vel.set_xlabel("Time (min)")
ax_vel.set_ylabel("Velocity (km/s)")
ax_vel.set_title("Velocity magnitude")
ax_vel.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/orbit_demo.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved to examples/orbit_demo.png")
plt.show()
