#!/usr/bin/env python
"""GEqOE J2 Taylor-series propagator -- user-facing example.

Demonstrates all three usage levels of the analytical GEqOE propagator:

1. **Provider pipeline** -- build via ``AstrodynClient`` / ``PropagatorFactory``
   exactly like any other propagator kind (keplerian, numerical, DSST, etc.).
2. **Direct Orekit adapter** -- instantiate ``GEqOEPropagator`` with an Orekit
   ``Orbit`` and call ``propagate()``, ``get_native_state()``, etc.
3. **Pure-numpy engine** -- call ``taylor_cart_propagator`` directly with a
   Cartesian state vector and body constants (no Orekit dependency).

Run from project root:
    python examples/geqoe_propagator.py --mode all
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from _common import init_orekit, make_generated_dir, make_leo_orbit


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# -----------------------------------------------------------------------
# 1) Provider pipeline  (AstrodynClient -> factory -> GEqOE propagator)
# -----------------------------------------------------------------------

def run_provider_pipeline() -> None:
    _header("GEqOE via Provider Pipeline")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    orbit, epoch, frame = make_leo_orbit()
    app = AstrodynClient()

    # Build a GEqOE propagator through the standard factory pipeline.
    # The provider is registered automatically -- just set kind="geqoe".
    spec = PropagatorSpec(
        kind="geqoe",
        mass_kg=450.0,
        orekit_options={"taylor_order": 4},
    )
    # No body_constants in the BuildContext -- they are resolved from
    # Orekit's WGS84 constants automatically by require_body_constants().
    ctx = BuildContext(initial_orbit=orbit)

    propagator = app.propagation.build_propagator(spec, ctx)
    bc = propagator.body_constants
    print(f"Propagator type: {type(propagator).__name__}")
    print(f"Taylor order:    {propagator.order}")
    print(f"Body constants (resolved from Orekit):")
    print(f"  mu = {bc['mu']:.10e} m^3/s^2")
    print(f"  j2 = {bc['j2']:.10e}")
    print(f"  re = {bc['re']:.2f} m")

    # Propagate 1 orbit period
    period_s = 2.0 * math.pi * math.sqrt(orbit.getA() ** 3 / orbit.getMu())
    target = epoch.shiftedBy(period_s)
    state = propagator.propagate(target)

    pos = state.getPVCoordinates(frame).getPosition()
    vel = state.getPVCoordinates(frame).getVelocity()
    print(f"\nAfter 1 period ({period_s:.1f} s):")
    print(f"  Position (km):  [{pos.getX()/1e3:.3f}, {pos.getY()/1e3:.3f}, {pos.getZ()/1e3:.3f}]")
    print(f"  Velocity (m/s): [{vel.getX():.4f}, {vel.getY():.4f}, {vel.getZ():.4f}]")


# -----------------------------------------------------------------------
# 2) Direct Orekit adapter  (GEqOEPropagator instantiated explicitly)
# -----------------------------------------------------------------------

def run_direct_adapter() -> None:
    _header("GEqOE Direct Adapter")
    init_orekit()

    from astrodyn_core.propagation.providers.geqoe import GEqOEPropagator

    orbit, epoch, frame = make_leo_orbit()

    # body_constants is omitted -- the propagator resolves Earth WGS84
    # constants (mu, j2, re) from Orekit automatically.
    prop = GEqOEPropagator(
        initial_orbit=orbit,
        order=4,
        mass_kg=450.0,
    )
    bc = prop.body_constants
    print(f"Body constants (resolved from Orekit):")
    print(f"  mu = {bc['mu']:.10e} m^3/s^2")
    print(f"  j2 = {bc['j2']:.10e}")
    print(f"  re = {bc['re']:.2f} m")

    # --- Single-epoch propagation with Orekit SpacecraftState output ---
    dt = 600.0  # 10 minutes
    target = epoch.shiftedBy(dt)
    state = prop.propagate(target)

    pos = state.getPVCoordinates(frame).getPosition()
    vel = state.getPVCoordinates(frame).getVelocity()
    print(f"Propagate +{dt:.0f}s:")
    print(f"  Position (km):  [{pos.getX()/1e3:.3f}, {pos.getY()/1e3:.3f}, {pos.getZ()/1e3:.3f}]")
    print(f"  Velocity (m/s): [{vel.getX():.4f}, {vel.getY():.4f}, {vel.getZ():.4f}]")

    # --- get_native_state: Cartesian + STM as raw numpy arrays ---
    y, stm = prop.get_native_state(target)
    print(f"\nNative state (numpy):")
    print(f"  y   = {y}")
    print(f"  STM diagonal = {np.diag(stm)}")

    # --- propagate_array: batch propagation (no Orekit overhead) ---
    dt_grid = np.linspace(0, 3600, 13)  # 0..1h in 5-min steps
    y_out, stm_out = prop.propagate_array(dt_grid)

    print(f"\nBatch propagation ({len(dt_grid)} epochs, 0 to 3600 s):")
    print(f"  y_out shape: {y_out.shape}   (N, 6)")
    print(f"  stm shape:   {stm_out.shape}  (6, 6, N)")
    print(f"  Range in X (km): {y_out[:, 0].min()/1e3:.1f} .. {y_out[:, 0].max()/1e3:.1f}")
    print(f"  Range in Y (km): {y_out[:, 1].min()/1e3:.1f} .. {y_out[:, 1].max()/1e3:.1f}")

    # --- resetInitialState: maneuver-like workflow ---
    state_600 = prop.propagate(epoch.shiftedBy(600.0))
    prop.resetInitialState(state_600)
    state_again = prop.propagate(epoch.shiftedBy(600.0))  # dt=0 from new epoch
    pos2 = state_again.getPVCoordinates(frame).getPosition()
    print(f"\nAfter resetInitialState at +600s, propagate dt=0:")
    print(f"  Position (km): [{pos2.getX()/1e3:.3f}, {pos2.getY()/1e3:.3f}, {pos2.getZ()/1e3:.3f}]")


# -----------------------------------------------------------------------
# 3) Pure-numpy engine  (no Orekit required)
# -----------------------------------------------------------------------

def run_numpy_engine() -> None:
    _header("GEqOE Pure-Numpy Engine")

    from astrodyn_core.propagation.geqoe.conversion import BodyConstants
    from astrodyn_core.propagation.geqoe.core import taylor_cart_propagator

    # Initial Cartesian state [rx, ry, rz, vx, vy, vz] in SI (m, m/s)
    y0 = np.array([
        6_878_137.0, 0.0, 0.0,       # position (m)
        0.0, 7200.0, 2400.0,         # velocity (m/s)
    ])

    body = BodyConstants(
        mu=3.986004418e14,   # m^3/s^2
        j2=1.08262668e-3,    # dimensionless
        re=6_378_137.0,      # equatorial radius (m)
    )

    # Time grid: 0 to 1 hour in 60-s steps
    tspan = np.arange(0, 3601, 60, dtype=float)

    # --- Order comparison ---
    print("Taylor order comparison at t = 300 s:\n")
    print(f"  {'Order':<7} {'X (km)':>14} {'Y (km)':>14} {'Z (km)':>14}"
          f" {'Vx (m/s)':>14} {'Vy (m/s)':>14} {'Vz (m/s)':>14}")
    print(f"  {'-'*7} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    t_sample = np.array([300.0])
    for order in range(1, 5):
        y_out, stm = taylor_cart_propagator(
            tspan=t_sample,
            y0=y0,
            p=body,
            order=order,
        )
        s = y_out[0]
        print(f"  {order:<7}"
              f" {s[0]/1e3:>14.4f}"
              f" {s[1]/1e3:>14.4f}"
              f" {s[2]/1e3:>14.4f}"
              f" {s[3]:>14.6f}"
              f" {s[4]:>14.6f}"
              f" {s[5]:>14.6f}")

    # --- Full trajectory at order 4 ---
    y_out, stm = taylor_cart_propagator(tspan=tspan, y0=y0, p=body, order=4)

    r_mag = np.linalg.norm(y_out[:, :3], axis=1) / 1e3  # km
    v_mag = np.linalg.norm(y_out[:, 3:], axis=1) / 1e3  # km/s

    print(f"\nOrder 4 trajectory ({len(tspan)} epochs, 0 to {tspan[-1]:.0f} s):")
    print(f"  y_out shape: {y_out.shape}")
    print(f"  STM shape:   {stm.shape}")
    print(f"  |r| range (km):  {r_mag.min():.2f} .. {r_mag.max():.2f}")
    print(f"  |v| range (km/s): {v_mag.min():.4f} .. {v_mag.max():.4f}")

    # --- STM at t=300s ---
    idx_300 = 5  # 300s / 60s = index 5
    stm_300 = stm[:, :, idx_300]
    print(f"\n  STM at t=300s (diagonal): {np.diag(stm_300)}")
    print(f"  STM determinant:          {np.linalg.det(stm_300):.12f}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEqOE J2 Taylor-series propagator examples",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "provider", "adapter", "numpy"),
        default="all",
        help="Choose one demo or run all (default).",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=6, linewidth=120)

    steps = {
        "provider": run_provider_pipeline,
        "adapter": run_direct_adapter,
        "numpy": run_numpy_engine,
    }

    if args.mode == "all":
        for fn in steps.values():
            fn()
    else:
        steps[args.mode]()

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
