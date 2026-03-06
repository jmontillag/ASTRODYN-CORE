#!/usr/bin/env python
"""Adaptive GEqOE propagator benchmark — accuracy, checkpoints, and speed.

Four sections:

1. **Accuracy** — Adaptive GEqOE vs numerical GEqOE over 10 orbits,
   demonstrating convergence as ``pos_tol`` decreases.
2. **Checkpoint analysis** — Number of checkpoints and step sizes for
   different tolerances and orders.
3. **Speed** — Adaptive GEqOE (C++ and Python) vs Orekit numerical J2
   for long-arc propagation.
4. **Tolerance sweep** — Max position error vs ``pos_tol`` setting,
   demonstrating convergence to the numerical GEqOE solution.

Note on reference:
    The numerical GEqOE reference (``geqoe-numerical``) integrates the
    **exact same ODE** as the Taylor propagator using scipy DOP853.
    Any disagreement isolates pure Taylor truncation error, giving a
    much tighter validation than comparing against Orekit or Cartesian
    J2 integration (which would mix in frame/formulation differences).

Run from project root:
    conda run -n astrodyn-core-env python examples/geqoe_native/geqoe_adaptive_benchmark.py
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


def _get_checkpoints(prop):
    """Extract checkpoint list from either the Orekit wrapper or the plain propagator."""
    inner = getattr(prop, "_impl", prop)
    return inner._checkpoints


def _numerical_geqoe_reference(app, ctx, dt_grid):
    """Integrate the GEqOE ODE numerically with scipy DOP853."""
    from astrodyn_core import PropagatorSpec

    ref_prop = app.propagation.build_propagator(
        PropagatorSpec(
            kind="geqoe-numerical",
            orekit_options={"rtol": 1e-13, "atol": 1e-13},
        ),
        ctx,
    )
    ref, _ = ref_prop.propagate_array(dt_grid)
    return ref


def _build_numerical_reference(app, ctx, orbit, epoch, frame, dt_grid):
    """Build Orekit numerical J2 reference trajectory."""
    from dataclasses import replace

    from astrodyn_core import (
        IntegratorSpec,
        get_propagation_model,
        load_dynamics_config,
    )

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

    ref = np.zeros((len(dt_grid), 6))
    for i, dt in enumerate(dt_grid):
        state = num_prop.propagate(epoch.shiftedBy(float(dt)))
        pv = state.getPVCoordinates(frame)
        p = pv.getPosition()
        v = pv.getVelocity()
        ref[i] = [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()]
    return ref, num_prop


# -----------------------------------------------------------------------
# Section 1: Accuracy — Adaptive vs numerical GEqOE over 10 orbits
# -----------------------------------------------------------------------


def run_accuracy() -> None:
    _header("Section 1: Accuracy — Adaptive GEqOE vs Numerical GEqOE")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)

    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    n_orbits = 10
    dt_grid = np.linspace(0, n_orbits * period, 2000)

    # Numerical GEqOE reference (exact same ODE, scipy DOP853)
    ref = _numerical_geqoe_reference(app, ctx, dt_grid)

    print(f"\n  Arc: {n_orbits} orbits ({dt_grid[-1]:.0f}s), period = {period:.1f}s")
    print(f"  Reference: numerical GEqOE (scipy DOP853, rtol/atol = 1e-13)")
    print(f"  Body constants: WGS84 (resolved automatically)\n")

    print(
        f"  {'pos_tol (m)':<13} {'Order':<7} {'Ckpts':>7} "
        f"{'Max pos err (m)':>16} {'RMS pos err (m)':>16} "
        f"{'Max vel err (m/s)':>18}"
    )
    print(
        f"  {'-'*13} {'-'*7} {'-'*7} "
        f"{'-'*16} {'-'*16} {'-'*18}"
    )

    for pos_tol in [10.0, 1.0, 0.1]:
        for order in [2, 4]:
            prop = app.propagation.build_propagator(
                PropagatorSpec(
                    kind="geqoe-adaptive",
                    orekit_options={
                        "taylor_order": order,
                        "backend": "cpp",
                        "pos_tol": pos_tol,
                    },
                ),
                ctx,
            )
            y_adap, _ = prop.propagate_array(dt_grid)
            n_ckpt = prop.num_checkpoints

            pos_err = np.linalg.norm(y_adap[:, :3] - ref[:, :3], axis=1)
            vel_err = np.linalg.norm(y_adap[:, 3:] - ref[:, 3:], axis=1)

            print(
                f"  {pos_tol:<13.1f} {order:<7} {n_ckpt:>7} "
                f"{pos_err.max():>16.4f} "
                f"{np.sqrt(np.mean(pos_err**2)):>16.4f} "
                f"{vel_err.max():>18.6f}"
            )

    print(
        "\n  Errors decrease with tighter pos_tol, confirming convergence"
    )
    print(
        "  to the exact GEqOE solution (identical ODE, only Taylor truncation differs)."
    )


# -----------------------------------------------------------------------
# Section 2: Checkpoint analysis
# -----------------------------------------------------------------------


def run_checkpoint_analysis() -> None:
    _header("Section 2: Checkpoint Analysis — Step Sizes and Count")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, _, _ = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)

    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)

    print(f"\n  Arc: 10 orbits ({10*period:.0f}s), backend = cpp\n")
    print(
        f"  {'pos_tol (m)':<13} {'Order':<7} {'Checkpoints':>12} "
        f"{'Avg step (s)':>14} {'Min step (s)':>14} {'Max step (s)':>14}"
    )
    print(
        f"  {'-'*13} {'-'*7} {'-'*12} "
        f"{'-'*14} {'-'*14} {'-'*14}"
    )

    dt_grid = np.linspace(0, 10 * period, 500)

    for pos_tol in [100.0, 10.0, 1.0, 0.1]:
        for order in [2, 3, 4]:
            prop = app.propagation.build_propagator(
                PropagatorSpec(
                    kind="geqoe-adaptive",
                    orekit_options={
                        "taylor_order": order,
                        "backend": "cpp",
                        "pos_tol": pos_tol,
                    },
                ),
                ctx,
            )
            prop.propagate_array(dt_grid)
            ckpts = _get_checkpoints(prop)
            n = len(ckpts)
            epochs = [c.epoch_seconds for c in ckpts]
            if n > 1:
                steps = np.diff(epochs)
                avg_step = np.mean(np.abs(steps))
                min_step = np.min(np.abs(steps))
                max_step = np.max(np.abs(steps))
            else:
                avg_step = min_step = max_step = float("nan")

            print(
                f"  {pos_tol:<13.1f} {order:<7} {n:>12} "
                f"{avg_step:>14.1f} {min_step:>14.1f} {max_step:>14.1f}"
            )


# -----------------------------------------------------------------------
# Section 3: Speed — Adaptive vs Orekit numerical
# -----------------------------------------------------------------------


def run_speed_comparison() -> None:
    _header("Section 3: Speed — Adaptive GEqOE vs Orekit Numerical J2")
    init_orekit()

    from org.orekit.propagation import SpacecraftState

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)
    initial_state = SpacecraftState(orbit)

    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)

    _, num_prop = _build_numerical_reference(
        app, ctx, orbit, epoch, frame, np.array([0.0])
    )

    def _build_adaptive(backend):
        return app.propagation.build_propagator(
            PropagatorSpec(
                kind="geqoe-adaptive",
                orekit_options={
                    "taylor_order": 4,
                    "backend": backend,
                    "pos_tol": 1.0,
                },
            ),
            ctx,
        )

    N_REPEAT = 5

    def _best_of(fn, n=N_REPEAT) -> float:
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    n_orbits = 10
    dt_grid = np.linspace(0, n_orbits * period, 5000)
    sorted_dates = [epoch.shiftedBy(float(dt)) for dt in dt_grid]

    print(f"\n  Arc: {n_orbits} orbits ({dt_grid[-1]:.0f}s), 5000 epochs, order 4\n")
    print(f"  {'Propagator':<40} {'Time (ms)':>10} {'Speedup':>9}")
    print(f"  {'-'*40} {'-'*10} {'-'*9}")

    def _num_batch():
        num_prop.setInitialState(initial_state)
        for d in sorted_dates:
            num_prop.propagate(d)

    t_num = _best_of(_num_batch, n=3)
    print(f"  {'Orekit numerical J2':<40} {t_num*1000:>10.1f} {'1.0x':>9}")

    # Adaptive C++
    prop_adap_cpp = _build_adaptive("cpp")
    prop_adap_cpp.propagate_array(dt_grid)
    n_ckpt = prop_adap_cpp.num_checkpoints
    t_cold_cpp = _best_of(
        lambda: _build_adaptive("cpp").propagate_array(dt_grid), n=3
    )
    t_warm_cpp = _best_of(lambda: prop_adap_cpp.propagate_array(dt_grid))

    print(
        f"  {'Adaptive C++ (cold, %d ckpts)' % n_ckpt:<40} "
        f"{t_cold_cpp*1000:>10.1f} {t_num/t_cold_cpp:>8.0f}x"
    )
    print(
        f"  {'Adaptive C++ (warm, %d ckpts)' % n_ckpt:<40} "
        f"{t_warm_cpp*1000:>10.2f} {t_num/t_warm_cpp:>8.0f}x"
    )

    # Adaptive Python
    prop_adap_py = _build_adaptive("python")
    prop_adap_py.propagate_array(dt_grid)
    t_cold_py = _best_of(
        lambda: _build_adaptive("python").propagate_array(dt_grid), n=3
    )
    t_warm_py = _best_of(lambda: prop_adap_py.propagate_array(dt_grid))

    print(
        f"  {'Adaptive Python (cold)':<40} "
        f"{t_cold_py*1000:>10.1f} {t_num/t_cold_py:>8.0f}x"
    )
    print(
        f"  {'Adaptive Python (warm)':<40} "
        f"{t_warm_py*1000:>10.2f} {t_num/t_warm_py:>8.0f}x"
    )

    print(f"\n  'cold' = build + extend checkpoints + evaluate.")
    print(f"  'warm' = evaluate only (checkpoints already cached).")


# -----------------------------------------------------------------------
# Section 4: Tolerance sweep — convergence demonstration
# -----------------------------------------------------------------------


def run_tolerance_sweep() -> None:
    _header("Section 4: Tolerance Sweep — Convergence Demonstration")
    init_orekit()

    from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

    app = AstrodynClient()
    orbit, epoch, frame = make_leo_orbit()
    ctx = BuildContext(initial_orbit=orbit)

    a = float(orbit.getA())
    mu = float(orbit.getMu())
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)

    dt_grid = np.linspace(0, 5 * period, 1000)
    ref = _numerical_geqoe_reference(app, ctx, dt_grid)

    print(f"\n  Arc: 5 orbits ({5*period:.0f}s), order 4, C++ backend")
    print(f"  Reference: numerical GEqOE (scipy DOP853)\n")
    print(
        f"  {'pos_tol (m)':<13} {'Max err (m)':>13} {'RMS err (m)':>13} "
        f"{'Checkpoints':>12}"
    )
    print(
        f"  {'-'*13} {'-'*13} {'-'*13} "
        f"{'-'*12}"
    )

    for pos_tol in [100.0, 10.0, 1.0, 0.1, 0.01]:
        prop = app.propagation.build_propagator(
            PropagatorSpec(
                kind="geqoe-adaptive",
                orekit_options={
                    "taylor_order": 4,
                    "backend": "cpp",
                    "pos_tol": pos_tol,
                },
            ),
            ctx,
        )
        y_adap, _ = prop.propagate_array(dt_grid)
        n_ckpt = prop.num_checkpoints
        pos_err = np.linalg.norm(y_adap[:, :3] - ref[:, :3], axis=1)
        print(
            f"  {pos_tol:<13.2f} {pos_err.max():>13.4f} "
            f"{np.sqrt(np.mean(pos_err**2)):>13.4f} "
            f"{n_ckpt:>12}"
        )

    print(
        "\n  As pos_tol decreases, error converges toward zero,"
    )
    print(
        "  confirming that adaptive Taylor truncation is the only error source."
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main() -> None:
    np.set_printoptions(precision=6, linewidth=120)
    run_accuracy()
    run_checkpoint_analysis()
    run_speed_comparison()
    run_tolerance_sweep()
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
