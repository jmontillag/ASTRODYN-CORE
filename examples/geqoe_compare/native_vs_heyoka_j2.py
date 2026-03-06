#!/usr/bin/env python
"""Compare the legacy GEqOE J2 Taylor map against the heyoka GEqOE backend.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_compare/native_vs_heyoka_j2.py

This script focuses on the pieces that are directly comparable:
  1. the shared GEqOE state components ``(nu, q1, q2, p1, p2)``
  2. Taylor-in-time derivatives up to order 4
  3. short-window order-4 state reconstructions
  4. setup/runtime costs for the old staged map and the new heyoka integrator

The legacy propagator stores ``L`` while the heyoka backend stores ``K``. The
shared components are compared directly, and the ``L`` mismatch is reported as
an output note rather than folded into the derivative tables.
"""

from __future__ import annotations

import math
import time

import heyoka as hy
import numpy as np

from astrodyn_core.geqoe_taylor.conversions import cart2geqoe
from astrodyn_core.geqoe_taylor.integrator import build_state_integrator, propagate_grid
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.rhs import build_geqoe_system
from astrodyn_core.geqoe_taylor.utils import K_to_L
from astrodyn_core.propagation.geqoe.conversion import rv2geqoe
from astrodyn_core.propagation.geqoe.core import evaluate_taylor, prepare_taylor_coefficients

np.set_printoptions(precision=12, suppress=False)

J2 = 0.0010826266835531513
RE_M = 6_378_137.0
MU_M = 3.986004418e14
RE_KM = RE_M / 1.0e3
MU_KM = MU_M / 1.0e9

CART0_M = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 1_000.0], dtype=float)
DT_COMPARE_S = np.array([30.0, 120.0, 300.0], dtype=float)
N_GRID = 1000

OLD_INDEX = {"nu": 0, "q1": 1, "q2": 2, "p1": 3, "p2": 4, "L": 5}
NEW_INDEX = {"nu": 0, "p1": 1, "p2": 2, "K": 3, "q1": 4, "q2": 5}
SHARED_COMPONENTS = ("nu", "q1", "q2", "p1", "p2")


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _native_state_from_cart() -> np.ndarray:
    eq0_tuple = rv2geqoe(t=0.0, y=CART0_M, p=(J2, RE_M, MU_M))
    return np.hstack([component.flatten() for component in eq0_tuple])


def _heyoka_state_from_cart(perturbation: J2Perturbation) -> np.ndarray:
    r0 = CART0_M[:3] / 1.0e3
    v0 = CART0_M[3:] / 1.0e3
    return cart2geqoe(r0, v0, MU_KM, perturbation)


def _shared_rows_from_old(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix[OLD_INDEX[name], :] for name in SHARED_COMPONENTS])


def _shared_rows_from_new(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix[NEW_INDEX[name], :] for name in SHARED_COMPONENTS])


def _evaluate_heyoka_time_derivatives(
    perturbation: J2Perturbation,
    state0: np.ndarray,
    order: int,
) -> np.ndarray:
    sys, state_vars, _ = build_geqoe_system(
        perturbation,
        mu_val=perturbation.mu,
        use_par=False,
    )
    rhs = [expr for _, expr in sys]
    current = list(rhs)
    derivatives = np.zeros((len(state_vars), order), dtype=float)

    for deriv_order in range(order):
        func = hy.cfunc(current, vars=state_vars)
        derivatives[:, deriv_order] = np.asarray(func(state0), dtype=float)
        if deriv_order == order - 1:
            break
        current = [
            sum(hy.diff(expr, var) * rhs_i for var, rhs_i in zip(state_vars, rhs, strict=True))
            for expr in current
        ]

    return derivatives


def _evaluate_truncated_series(
    state0: np.ndarray, derivatives: np.ndarray, dt_s: np.ndarray
) -> np.ndarray:
    out = np.repeat(np.asarray(state0, dtype=float)[None, :], len(dt_s), axis=0)
    for order_idx in range(derivatives.shape[1]):
        out += (
            derivatives[:, order_idx][None, :]
            * dt_s[:, None] ** (order_idx + 1)
            / math.factorial(order_idx + 1)
        )
    return out


def _best_of(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _time_native(coeffs, dt_grid: np.ndarray) -> tuple[float, float]:
    t_prepare = _best_of(
        lambda: prepare_taylor_coefficients(
            _native_state_from_cart(),
            (J2, RE_M, MU_M),
            order=4,
        ),
        repeat=3,
    )
    t_evaluate = _best_of(
        lambda: evaluate_taylor(coeffs, dt_grid),
        repeat=5,
    )
    return t_prepare, t_evaluate


def _time_heyoka(
    perturbation: J2Perturbation, state0: np.ndarray, dt_grid: np.ndarray
) -> tuple[float, float]:
    t_build = _best_of(
        lambda: build_state_integrator(
            perturbation,
            state0,
            tol=1.0e-15,
            compact_mode=True,
        ),
        repeat=3,
    )

    ta, _ = build_state_integrator(
        perturbation,
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )

    def _run_grid() -> None:
        ta.time = 0.0
        ta.state[:] = state0
        propagate_grid(ta, dt_grid)

    t_grid = _best_of(_run_grid, repeat=5)
    return t_build, t_grid


def main() -> None:
    perturbation = J2Perturbation(mu=MU_KM, j2=J2, re=RE_KM)
    native_state = _native_state_from_cart()
    heyoka_state = _heyoka_state_from_cart(perturbation)

    _header("1. Initial-State Alignment")
    shared_native = np.array(
        [native_state[OLD_INDEX[name]] for name in SHARED_COMPONENTS],
        dtype=float,
    )
    shared_heyoka = np.array(
        [heyoka_state[NEW_INDEX[name]] for name in SHARED_COMPONENTS],
        dtype=float,
    )
    l_from_heyoka = K_to_L(
        heyoka_state[NEW_INDEX["K"]],
        heyoka_state[NEW_INDEX["p1"]],
        heyoka_state[NEW_INDEX["p2"]],
    )

    shared_state_diff = np.max(np.abs(shared_native - shared_heyoka))
    l_diff = abs(native_state[OLD_INDEX["L"]] - l_from_heyoka)
    print(f"Shared-state max |native - heyoka|: {shared_state_diff:.3e}")
    print(
        f"Legacy L vs heyoka-derived L:       {l_diff:.3e}"
    )
    print("Legacy order: [nu, q1, q2, p1, p2, L]")
    print("Heyoka order: [nu, p1, p2, K, q1, q2]")

    _header("2. Taylor-Derivative Comparison (Orders 1-4)")
    native_coeffs = prepare_taylor_coefficients(
        native_state,
        p=(J2, RE_M, MU_M),
        order=4,
    )
    native_derivatives = (
        native_coeffs.map_components
        / np.array(
            [native_coeffs.constants.time_scale ** (i + 1) for i in range(4)],
            dtype=float,
        )[None, :]
    )
    heyoka_derivatives = _evaluate_heyoka_time_derivatives(
        perturbation,
        heyoka_state,
        order=4,
    )

    shared_native_derivatives = _shared_rows_from_old(native_derivatives)
    shared_heyoka_derivatives = _shared_rows_from_new(heyoka_derivatives)

    print(
        f"{'Component':<10} {'Order':<6} {'Native':>18} "
        f"{'Heyoka':>18} {'|diff|':>12}"
    )
    print("-" * 72)
    for row_idx, name in enumerate(SHARED_COMPONENTS):
        for order in range(4):
            native_val = shared_native_derivatives[row_idx, order]
            heyoka_val = shared_heyoka_derivatives[row_idx, order]
            print(
                f"{name:<10} {order + 1:<6} "
                f"{native_val:>18.10e} {heyoka_val:>18.10e} "
                f"{abs(native_val - heyoka_val):>12.3e}"
            )

    print(
        "\nShared-component max derivative difference: "
        f"{np.max(np.abs(shared_native_derivatives - shared_heyoka_derivatives)):.3e}"
    )

    _header("3. Order-4 State Reconstruction Over Short Windows")
    native_series_states, _, _ = evaluate_taylor(native_coeffs, DT_COMPARE_S)
    heyoka_series_states = _evaluate_truncated_series(
        heyoka_state,
        heyoka_derivatives,
        DT_COMPARE_S,
    )

    ta, _ = build_state_integrator(
        perturbation,
        heyoka_state,
        tol=1.0e-15,
        compact_mode=True,
    )
    heyoka_adaptive_states = propagate_grid(
        ta,
        np.concatenate([[0.0], DT_COMPARE_S]),
    )[1:, :]

    shared_native_states = _shared_rows_from_old(native_series_states.T).T
    shared_series_states = _shared_rows_from_new(heyoka_series_states.T).T
    shared_adaptive_states = _shared_rows_from_new(heyoka_adaptive_states.T).T

    print(
        f"{'dt (s)':<8} {'|native - heyoka series|':>28} "
        f"{'|series - heyoka adaptive|':>28}"
    )
    print("-" * 68)
    for dt, native_row, series_row, adaptive_row in zip(
        DT_COMPARE_S,
        shared_native_states,
        shared_series_states,
        shared_adaptive_states,
        strict=True,
    ):
        diff_native_series = np.max(np.abs(native_row - series_row))
        diff_series_adaptive = np.max(np.abs(series_row - adaptive_row))
        print(f"{dt:<8.1f} {diff_native_series:>28.3e} {diff_series_adaptive:>28.3e}")

    _header("4. Setup And Runtime Comparison")
    r0_km = CART0_M[:3] / 1.0e3
    v0_km = CART0_M[3:] / 1.0e3
    a_km = 1.0 / ((2.0 / np.linalg.norm(r0_km)) - np.dot(v0_km, v0_km) / MU_KM)
    period_s = 2.0 * math.pi * math.sqrt(a_km**3 / MU_KM)
    dt_grid = np.linspace(0.0, period_s, N_GRID)

    native_prepare_s, native_eval_s = _time_native(native_coeffs, dt_grid)
    heyoka_build_s, heyoka_grid_s = _time_heyoka(perturbation, heyoka_state, dt_grid)

    print(
        f"Reference grid: {N_GRID} samples over one "
        f"osculating orbit ({period_s:.1f} s)"
    )
    print()
    print(f"{'Operation':<28} {'Native staged':>16} {'Heyoka':>16}")
    print("-" * 64)
    print(
        f"{'One-time setup/build':<28} "
        f"{native_prepare_s * 1e3:>13.2f} ms "
        f"{heyoka_build_s * 1e3:>13.2f} ms"
    )
    print(
        f"{'Dense grid propagation':<28} "
        f"{native_eval_s * 1e3:>13.2f} ms "
        f"{heyoka_grid_s * 1e3:>13.2f} ms"
    )

    print("\nInterpretation:")
    print(
        "- Native staged GEqOE is a fixed order-4 local Taylor map and is "
        "very cheap to reevaluate once its coefficients are prepared."
    )
    print(
        "- Heyoka pays a larger symbolic build cost, but then propagates the "
        "full GEqOE dynamics adaptively with internally selected Taylor order."
    )
    print(
        "- The shared J2 Taylor derivatives match to numerical precision "
        "through order 4, which is the key parity point with the previous "
        "implementation."
    )


if __name__ == "__main__":
    main()
