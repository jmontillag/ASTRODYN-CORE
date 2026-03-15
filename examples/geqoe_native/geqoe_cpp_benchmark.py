#!/usr/bin/env python
"""GEqOE C++ backend benchmark — performance, precision, and parity.

Four sections:

1. **Performance** — Python vs C++ backend timing for prepare + evaluate
   phases across orders 1-4.
2. **Precision** — GEqOE orders 1-4 vs Orekit numerical J2 propagator
   (built via the standard ``AstrodynClient`` pipeline), showing error
   convergence with Taylor order.
3. **Speed** — GEqOE (Python and C++) vs Orekit numerical J2, single-epoch
   and batch propagation wall-clock time comparison.
4. **Parity** — Spot-check that Python and C++ backends produce identical
   results (max difference < 1e-6 m).

Run from project root:
    conda run -n astrodyn-core-env python examples/geqoe_native/geqoe_cpp_benchmark.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples._common import init_orekit, make_leo_orbit


def _header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# -----------------------------------------------------------------------
# Section 1: Performance — Python vs C++ backend
# -----------------------------------------------------------------------


def run_performance() -> None:
    _header("Section 1: Performance — Python vs C++ backend")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, _, _ = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)
    dt_grid = np.linspace(0, 3600, 1000)
    N_REPEAT = 5

    def _best_of(fn) -> float:
        best = float("inf")
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    print(f"\n  1000 time-points, 0-3600s, {N_REPEAT} repetitions (best reported)\n")
    print(f"  {'Order':<6} {'Phase':<10} {'Python (ms)':>12} {'C++ (ms)':>12} {'Speedup':>9}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*9}")

    for order in range(1, 5):
        # Build via the standard provider pipeline, varying the backend.
        def _build(backend):
            return app.propagation.build_propagator(
                PropagatorSpec(
                    kind="geqoe",
                    orekit_options={"taylor_order": order, "backend": backend},
                ),
                ctx,
            )

        # --- Prepare phase (includes coefficient computation) ---
        t_prep_py = _best_of(lambda: _build("python"))
        t_prep_cpp = _best_of(lambda: _build("cpp"))

        # Build once for evaluate timing
        prop_py = _build("python")
        prop_cpp = _build("cpp")

        # --- Evaluate phase ---
        t_eval_py = _best_of(lambda: prop_py.propagate_array(dt_grid))
        t_eval_cpp = _best_of(lambda: prop_cpp.propagate_array(dt_grid))

        su_prep = t_prep_py / t_prep_cpp if t_prep_cpp > 0 else float("inf")
        su_eval = t_eval_py / t_eval_cpp if t_eval_cpp > 0 else float("inf")

        print(
            f"  {order:<6} {'prepare':<10} {t_prep_py*1000:>12.2f} "
            f"{t_prep_cpp*1000:>12.2f} {su_prep:>8.1f}x"
        )
        print(
            f"  {'':<6} {'evaluate':<10} {t_eval_py*1000:>12.2f} "
            f"{t_eval_cpp*1000:>12.2f} {su_eval:>8.1f}x"
        )
        print()


# -----------------------------------------------------------------------
# Section 2: Precision — GEqOE orders 1–4 vs Orekit numerical J2
# -----------------------------------------------------------------------


def run_precision() -> None:
    _header("Section 2: Precision — GEqOE vs Orekit numerical J2")
    init_orekit()

    from astrodyn_core import (
        AstrodynClient,
        BuildContext,
        IntegratorSpec,
        PropagatorSpec,
        load_dynamics_config,
        get_propagation_model,
    )

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)

    # --- Build reference Orekit numerical J2 propagator via the standard
    #     pipeline using the built-in "j2_model" preset (degree 2, order 0),
    #     with a tighter integrator for reference-quality accuracy.
    from dataclasses import replace

    j2_spec = load_dynamics_config(get_propagation_model("j2_model"))
    j2_spec = replace(
        j2_spec,
        integrator=IntegratorSpec(
            kind="dp853",
            min_step=1e-6,
            max_step=300.0,
            position_tolerance=1e-3,
        ),
    )
    num_builder = app.propagation.build_builder(j2_spec, ctx)
    num_prop = num_builder.buildPropagator(
        num_builder.getSelectedNormalizedParameters()
    )

    # --- Time grid: 0 to 600s in 10s steps ---
    # The Taylor series is a polynomial expansion around t=0; higher orders
    # converge better at moderate dt but diverge faster at very large dt.
    # We use 600s (~10% of the orbit period) to stay within the convergence
    # radius and clearly show error decreasing with order.
    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    dt_grid = np.arange(0, 601, 10.0)
    n_pts = len(dt_grid)

    # Reference trajectory from numerical propagator
    ref = np.zeros((n_pts, 6))
    for i, dt in enumerate(dt_grid):
        state = num_prop.propagate(epoch.shiftedBy(float(dt)))
        pv = state.getPVCoordinates(frame)
        p = pv.getPosition()
        v = pv.getVelocity()
        ref[i] = [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()]

    print(f"\n  Reference: Orekit numerical J2 (j2_model preset, degree 2 order 0)")
    print(f"  Grid: {n_pts} points, 0 to {dt_grid[-1]:.0f}s (period = {period:.1f}s)\n")
    print(
        f"  {'Order':<7} {'Max pos err (m)':>16} {'RMS pos err (m)':>16} "
        f"{'Max vel err (m/s)':>18}"
    )
    print(f"  {'-'*7} {'-'*16} {'-'*16} {'-'*18}")

    for order in range(1, 5):
        geqoe_prop = app.propagation.build_propagator(
            PropagatorSpec(
                kind="geqoe",
                orekit_options={"taylor_order": order, "backend": "cpp"},
            ),
            ctx,
        )
        y_geqoe, _ = geqoe_prop.propagate_array(dt_grid)

        pos_err = np.linalg.norm(y_geqoe[:, :3] - ref[:, :3], axis=1)
        vel_err = np.linalg.norm(y_geqoe[:, 3:] - ref[:, 3:], axis=1)

        print(
            f"  {order:<7} {pos_err.max():>16.4f} {np.sqrt(np.mean(pos_err**2)):>16.4f} "
            f"{vel_err.max():>18.6f}"
        )

    print("\n  (Errors should decrease with increasing order)")


# -----------------------------------------------------------------------
# Section 3: GEqOE vs Orekit numerical — propagation speed
# -----------------------------------------------------------------------


def run_speed_comparison() -> None:
    _header("Section 3: GEqOE vs Orekit Numerical — Propagation Speed")
    init_orekit()

    from dataclasses import replace as dc_replace

    from org.orekit.propagation import SpacecraftState

    from astrodyn_core import (
        AstrodynClient,
        BuildContext,
        IntegratorSpec,
        PropagatorSpec,
        get_propagation_model,
        load_dynamics_config,
    )

    app = AstrodynClient()
    orbit, epoch, _ = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)
    initial_state = SpacecraftState(orbit)

    # Build the three propagators
    j2_spec = load_dynamics_config(get_propagation_model("j2_model"))
    j2_spec = dc_replace(
        j2_spec,
        integrator=IntegratorSpec(
            kind="dp853", min_step=1e-3, max_step=200.0, position_tolerance=1e-3,
        ),
    )
    num_builder = app.propagation.build_builder(j2_spec, ctx)
    num_prop = num_builder.buildPropagator(
        num_builder.getSelectedNormalizedParameters()
    )

    prop_py = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe",
            orekit_options={"taylor_order": 4, "backend": "python"},
        ),
        ctx,
    )
    prop_cpp = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe",
            orekit_options={"taylor_order": 4, "backend": "cpp"},
        ),
        ctx,
    )

    N_REPEAT = 10
    target = epoch.shiftedBy(600.0)

    def _best_of(fn, n=N_REPEAT) -> float:
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    # --- Measure GEqOE preparation cost (one-time) ---
    geqoe_py_spec = PropagatorSpec(
        kind="geqoe",
        orekit_options={"taylor_order": 4, "backend": "python"},
    )
    geqoe_cpp_spec = PropagatorSpec(
        kind="geqoe",
        orekit_options={"taylor_order": 4, "backend": "cpp"},
    )
    t_prep_py = _best_of(
        lambda: app.propagation.build_propagator(geqoe_py_spec, ctx)
    )
    t_prep_cpp = _best_of(
        lambda: app.propagation.build_propagator(geqoe_cpp_spec, ctx)
    )

    # --- Single epoch (dt = 600s) ---
    def _num_single():
        num_prop.setInitialState(initial_state)
        num_prop.propagate(target)

    t_num = _best_of(_num_single)
    t_eval_py = _best_of(lambda: prop_py.propagate_array(np.array([600.0])))
    t_eval_cpp = _best_of(lambda: prop_cpp.propagate_array(np.array([600.0])))

    print(f"\n  Single epoch (dt = 600s), order 4, best of {N_REPEAT}\n")
    print(f"  {'Propagator':<34} {'Time (ms)':>10} {'Speedup':>9}")
    print(f"  {'-'*34} {'-'*10} {'-'*9}")
    print(f"  {'Orekit numerical J2':<34} {t_num*1000:>10.3f} {'1.0x':>9}")
    print(f"  {'GEqOE Python (prepare + eval)':<34} {(t_prep_py+t_eval_py)*1000:>10.3f} {t_num/(t_prep_py+t_eval_py):>8.1f}x")
    print(f"  {'GEqOE Python (eval only)':<34} {t_eval_py*1000:>10.3f} {t_num/t_eval_py:>8.0f}x")
    print(f"  {'GEqOE C++    (prepare + eval)':<34} {(t_prep_cpp+t_eval_cpp)*1000:>10.3f} {t_num/(t_prep_cpp+t_eval_cpp):>8.0f}x")
    print(f"  {'GEqOE C++    (eval only)':<34} {t_eval_cpp*1000:>10.3f} {t_num/t_eval_cpp:>8.0f}x")

    # --- Batch: 1000 epochs ---
    dt_grid = np.linspace(0, 3600, 1000)
    sorted_dates = [epoch.shiftedBy(float(dt)) for dt in dt_grid]

    def _num_batch():
        num_prop.setInitialState(initial_state)
        for d in sorted_dates:
            num_prop.propagate(d)

    N_BATCH = 3
    t_num_b = _best_of(_num_batch, n=N_BATCH)
    t_eval_py_b = _best_of(lambda: prop_py.propagate_array(dt_grid), n=N_BATCH)
    t_eval_cpp_b = _best_of(lambda: prop_cpp.propagate_array(dt_grid), n=N_BATCH)

    print(f"\n  Batch: 1000 epochs (0-3600s), order 4, best of {N_BATCH}\n")
    print(f"  {'Propagator':<34} {'Time (ms)':>10} {'Speedup':>9}")
    print(f"  {'-'*34} {'-'*10} {'-'*9}")
    print(f"  {'Orekit numerical J2':<34} {t_num_b*1000:>10.2f} {'1.0x':>9}")
    print(f"  {'GEqOE Python (prepare + eval)':<34} {(t_prep_py+t_eval_py_b)*1000:>10.2f} {t_num_b/(t_prep_py+t_eval_py_b):>8.0f}x")
    print(f"  {'GEqOE Python (eval only)':<34} {t_eval_py_b*1000:>10.2f} {t_num_b/t_eval_py_b:>8.0f}x")
    print(f"  {'GEqOE C++    (prepare + eval)':<34} {(t_prep_cpp+t_eval_cpp_b)*1000:>10.2f} {t_num_b/(t_prep_cpp+t_eval_cpp_b):>8.0f}x")
    print(f"  {'GEqOE C++    (eval only)':<34} {t_eval_cpp_b*1000:>10.2f} {t_num_b/t_eval_cpp_b:>8.0f}x")


# -----------------------------------------------------------------------
# Section 4: Parity — Python vs C++ backend
# -----------------------------------------------------------------------


def run_parity() -> None:
    _header("Section 4: Parity — Python vs C++ backend")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, _, _ = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)
    rng = np.random.default_rng(42)
    dt_check = rng.uniform(10.0, 3600.0, size=50)

    prop_py = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe",
            orekit_options={"taylor_order": 4, "backend": "python"},
        ),
        ctx,
    )
    prop_cpp = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe",
            orekit_options={"taylor_order": 4, "backend": "cpp"},
        ),
        ctx,
    )

    y_py, _ = prop_py.propagate_array(dt_check)
    y_cpp, _ = prop_cpp.propagate_array(dt_check)

    diff = np.abs(y_py - y_cpp)
    max_pos_diff = diff[:, :3].max()
    max_vel_diff = diff[:, 3:].max()

    print(f"\n  Order 4, {len(dt_check)} random epochs (10-3600s)")
    print(f"  Max position difference: {max_pos_diff:.2e} m")
    print(f"  Max velocity difference: {max_vel_diff:.2e} m/s")

    if max_pos_diff < 1e-6:
        print("  [OK] Backends agree to < 1e-6 m (floating-point parity)")
    else:
        print("  [WARN] Difference exceeds 1e-6 m — investigate!")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main() -> None:
    np.set_printoptions(precision=6, linewidth=120)
    run_performance()
    run_precision()
    run_speed_comparison()
    run_parity()
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
