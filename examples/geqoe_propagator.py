#!/usr/bin/env python
"""GEqOE J2 Taylor-series propagator -- user-facing example.

Demonstrates all four usage levels of the analytical GEqOE propagator:

1. **Provider pipeline** -- build via ``AstrodynClient`` / ``PropagatorFactory``
   exactly like any other propagator kind (keplerian, numerical, DSST, etc.).
2. **Direct Orekit adapter** -- instantiate ``GEqOEPropagator`` with an Orekit
   ``Orbit`` and call ``propagate()``, ``get_native_state()``, etc.
3. **Pure-numpy engine** -- call ``taylor_cart_propagator`` directly with a
   Cartesian state vector and body constants (no Orekit dependency).
4. **Multi-epoch benchmark** -- compare the monolithic and staged
   (``prepare_taylor_coefficients`` + ``evaluate_taylor``) strategies when
   the same initial state is evaluated at many independent epochs.

Run from project root:
    python examples/geqoe_propagator.py --mode all
    python examples/geqoe_propagator.py --mode benchmark
"""

from __future__ import annotations

import argparse
import math
import time

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
# 4) Multi-epoch benchmark  (staged vs monolithic)
# -----------------------------------------------------------------------

def run_benchmark() -> None:
    """Benchmark the high-level GEqOEPropagator across repeated independent epochs.

    ``GEqOEPropagator`` now caches the dt-independent Taylor coefficients at
    construction time (via ``prepare_cart_coefficients``).  Each call to
    ``propagate()`` or ``propagate_array()`` therefore only performs the cheap
    polynomial evaluation + Cartesian conversion, not the full coefficient
    computation.

    Two scenarios are timed to show where the benefit is visible:

    **Scenario A — repeated scalar propagate() calls**
        Simulates a filter or tasking loop that repeatedly asks the propagator
        for the state at a new epoch from the same initial condition.  The new
        implementation pays the coefficient cost once at construction; the old
        pattern (calling ``taylor_cart_propagator`` each time) would pay it on
        every call.

    **Scenario B — propagate_array() vs repeated propagate() calls**
        Shows that ``propagate_array()`` with a pre-built grid is fastest
        (single Python call, vectorised numpy), but that repeated
        ``propagate()`` calls now use the cache and are far cheaper than they
        would be without it.

    The reference baseline is the time taken to run the same number of
    propagations by calling the lower-level ``taylor_cart_propagator`` in a
    loop (which recomputes coefficients each time).
    """
    _header("GEqOEPropagator Benchmark: Cached Coefficients")
    init_orekit()

    from astrodyn_core.propagation.geqoe.conversion import BodyConstants, rv2geqoe
    from astrodyn_core.propagation.geqoe.core import taylor_cart_propagator
    from astrodyn_core.propagation.providers.geqoe import GEqOEPropagator

    orbit, epoch, frame = make_leo_orbit()
    rng = np.random.default_rng(1)
    N_EPOCHS = 100
    N_REPEAT = 5

    def _timeit(fn) -> float:
        best = float("inf")
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    print(f"\n  {N_EPOCHS} independent epochs, order=4, {N_REPEAT} timing repetitions (min reported)\n")
    print(f"  {'Strategy':<42} {'Time (ms)':>10}  {'speedup':>9}")
    print(f"  {'-'*42} {'-'*10}  {'-'*9}")

    for order in (1, 2, 3, 4):
        # Build the propagator once — coefficients precomputed at construction.
        prop = GEqOEPropagator(initial_orbit=orbit, order=order)
        y0 = prop._y0
        bc = prop._bc

        dt_epochs = rng.uniform(60.0, 3600.0, size=N_EPOCHS)
        target_dates = [epoch.shiftedBy(float(dt)) for dt in dt_epochs]
        dt_arr = np.array(dt_epochs)

        # --- Baseline: lower-level function called in a loop (re-computes each time) ---
        def _baseline_loop():
            for dt in dt_arr:
                taylor_cart_propagator(
                    tspan=np.array([dt]), y0=y0, p=bc, order=order
                )

        # --- New: GEqOEPropagator.propagate() in a loop (uses cache) ---
        def _propagator_loop():
            for date in target_dates:
                prop.propagate(date)

        # --- New: GEqOEPropagator.propagate_array() in one batch call ---
        def _propagate_array():
            prop.propagate_array(dt_arr)

        t_baseline = _timeit(_baseline_loop)
        t_loop = _timeit(_propagator_loop)
        t_batch = _timeit(_propagate_array)

        su_loop = t_baseline / t_loop
        su_batch = t_baseline / t_batch

        print(f"  order={order}  baseline loop (taylor_cart per call)  {t_baseline*1000:>10.2f}  {'1.00x':>9}  <-- reference")
        print(f"  order={order}  GEqOEPropagator.propagate() loop      {t_loop*1000:>10.2f}  {su_loop:>8.2f}x")
        print(f"  order={order}  GEqOEPropagator.propagate_array()     {t_batch*1000:>10.2f}  {su_batch:>8.2f}x")
        print()

    # Verify numerical parity: GEqOEPropagator vs baseline at order 4
    prop4 = GEqOEPropagator(initial_orbit=orbit, order=4)
    dt_check = rng.uniform(60.0, 3600.0, size=50)
    y_base = np.vstack([
        taylor_cart_propagator(tspan=np.array([dt]), y0=prop4._y0, p=prop4._bc, order=4)[0]
        for dt in dt_check
    ])
    y_prop, _ = prop4.propagate_array(dt_check)
    max_diff = np.max(np.abs(y_prop - y_base))
    print(f"  Parity check (order 4, 50 epochs): max |propagator - baseline| = {max_diff:.2e}")
    if max_diff < 1e-8:
        print("  [OK] Results agree to better than 1e-8 m.")
    else:
        print("  [WARN] Larger than expected difference!")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEqOE J2 Taylor-series propagator examples",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "provider", "adapter", "numpy", "benchmark"),
        default="all",
        help="Choose one demo or run all (default).",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=6, linewidth=120)

    steps = {
        "provider": run_provider_pipeline,
        "adapter": run_direct_adapter,
        "numpy": run_numpy_engine,
        "benchmark": run_benchmark,
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
