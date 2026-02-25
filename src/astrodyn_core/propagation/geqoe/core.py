"""Core GEqOE Taylor-series propagation API (staged J2 backend).

This module provides both:

- one-shot propagation helpers (`j2_taylor_propagator`, `taylor_cart_propagator`)
- two-stage APIs that precompute dt-independent Taylor coefficients and then
  evaluate them cheaply on arbitrary time grids
"""

from typing import Tuple, Union

import numpy as np

from astrodyn_core.propagation.geqoe.conversion import BodyConstants, geqoe2rv, rv2geqoe
from astrodyn_core.propagation.geqoe.jacobians import get_pEqpY, get_pYpEq
from astrodyn_core.propagation.geqoe.state import (
    GEqOEPropagationConstants,
    GEqOEPropagationContext,
    GEqOEState,
    GEqOETaylorCoefficients,
)
from astrodyn_core.propagation.geqoe.taylor_order_1 import compute_coefficients_1, compute_order_1, evaluate_order_1
from astrodyn_core.propagation.geqoe.taylor_order_2 import compute_coefficients_2, compute_order_2, evaluate_order_2
from astrodyn_core.propagation.geqoe.taylor_order_3 import compute_coefficients_3, compute_order_3, evaluate_order_3
from astrodyn_core.propagation.geqoe.taylor_order_4 import compute_coefficients_4, compute_order_4, evaluate_order_4


def _validate_order(order: int) -> int:
    """Validate and normalize Taylor expansion order.

    Args:
        order: Requested Taylor order.

    Returns:
        Integer order in ``[1, 4]``.

    Raises:
        ValueError: If the order is outside ``[1, 4]``.
    """
    order_int = int(order)
    if order_int < 1 or order_int > 4:
        raise ValueError("Taylor order must be an integer in the range [1, 4].")
    return order_int


def build_context(
    dt: Union[float, np.ndarray],
    y0: np.ndarray,
    p: Union[BodyConstants, tuple, list],
    order: int,
) -> GEqOEPropagationContext:
    """Build a normalized GEqOE propagation context for staged execution.

    Args:
        dt: Propagation time offset(s) in seconds.
        y0: Initial GEqOE state vector ``[nu, q1, q2, p1, p2, Lr]``.
        p: Body constants as ``BodyConstants`` or ``(j2, re, mu)`` tuple/list.
        order: Taylor expansion order (1-4).

    Returns:
        Populated propagation context with normalized time/state scales.
    """
    order = _validate_order(order)
    if isinstance(p, BodyConstants):
        j2, re, mu = p.j2, p.re, p.mu
    else:
        j2, re, mu = p

    length_scale = re
    time_scale = (re**3 / mu) ** 0.5
    dt_norm = np.atleast_1d(dt).astype(float) / time_scale

    constants = GEqOEPropagationConstants(
        j2=float(j2),
        re=float(re),
        mu=float(mu),
        length_scale=float(length_scale),
        time_scale=float(time_scale),
        mu_norm=1.0,
        a_half_j2=float(j2) / 2.0,
    )
    return GEqOEPropagationContext(
        dt_seconds=np.atleast_1d(dt).astype(float),
        dt_norm=dt_norm,
        initial_state=GEqOEState.from_array(y0),
        order=order,
        constants=constants,
    )


def _run_staged_j2(
    context: GEqOEPropagationContext,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the staged GEqOE backend and assemble the GEqOE-level STM.

    Args:
        context: Propagation context prepared by :func:`build_context`.

    Returns:
        Tuple ``(y_prop, y_y0, map_components)`` containing propagated GEqOE
        states, GEqOE-to-GEqOE STMs, and Taylor coefficient map components.

    Raises:
        RuntimeError: If the staged backend did not populate expected outputs.
    """
    staged_dispatch = {
        1: compute_order_1,
        2: compute_order_2,
        3: compute_order_3,
        4: compute_order_4,
    }
    staged_dispatch[context.order](context)

    if context.y_prop is None or context.map_components is None:
        raise RuntimeError("Staged backend did not populate propagation outputs.")

    # ----------------------------------------------------------------
    # Final STM assembly  (legacy lines 2215-2253)
    # Build y_y0[6, 6, M] from accumulated STM partials in scratch.
    # Column 0 (_nu partials) is scaled by T (time_scale).
    # ----------------------------------------------------------------
    T = context.constants.time_scale
    s = context.scratch
    M = len(context.dt_norm)

    y_y0 = np.zeros((6, 6, M))

    # Row 0 – nu
    y_y0[0, 0, :] = s["nu_nu"]

    # Row 1 – q1
    y_y0[1, 0, :] = s["q1_nu"] * T
    y_y0[1, 1, :] = s["q1_q1"]
    y_y0[1, 2, :] = s["q1_q2"]
    y_y0[1, 3, :] = s["q1_p1"]
    y_y0[1, 4, :] = s["q1_p2"]
    y_y0[1, 5, :] = s["q1_Lr"]

    # Row 2 – q2
    y_y0[2, 0, :] = s["q2_nu"] * T
    y_y0[2, 1, :] = s["q2_q1"]
    y_y0[2, 2, :] = s["q2_q2"]
    y_y0[2, 3, :] = s["q2_p1"]
    y_y0[2, 4, :] = s["q2_p2"]
    y_y0[2, 5, :] = s["q2_Lr"]

    # Row 3 – p1
    y_y0[3, 0, :] = s["p1_nu"] * T
    y_y0[3, 1, :] = s["p1_q1"]
    y_y0[3, 2, :] = s["p1_q2"]
    y_y0[3, 3, :] = s["p1_p1"]
    y_y0[3, 4, :] = s["p1_p2"]
    y_y0[3, 5, :] = s["p1_Lr"]

    # Row 4 – p2
    y_y0[4, 0, :] = s["p2_nu"] * T
    y_y0[4, 1, :] = s["p2_q1"]
    y_y0[4, 2, :] = s["p2_q2"]
    y_y0[4, 3, :] = s["p2_p1"]
    y_y0[4, 4, :] = s["p2_p2"]
    y_y0[4, 5, :] = s["p2_Lr"]

    # Row 5 – Lr
    y_y0[5, 0, :] = s["Lr_nu"] * T
    y_y0[5, 1, :] = s["Lr_q1"]
    y_y0[5, 2, :] = s["Lr_q2"]
    y_y0[5, 3, :] = s["Lr_p1"]
    y_y0[5, 4, :] = s["Lr_p2"]
    y_y0[5, 5, :] = s["Lr_Lr"]

    # ----------------------------------------------------------------
    # Output normalization  (legacy line 2255)
    # Convert nu back from normalised to physical units.
    # ----------------------------------------------------------------
    context.y_prop[:, 0] /= T

    context.y_y0 = y_y0
    return context.y_prop, context.y_y0, context.map_components


def prepare_taylor_coefficients(
    y0: np.ndarray,
    p: Union[BodyConstants, tuple, list],
    order: int = 4,
) -> GEqOETaylorCoefficients:
    """Precompute all dt-independent Taylor coefficients for a given initial state.

    This is the first half of the two-stage propagation API.  Call this once per
    initial state, then call :func:`evaluate_taylor` as many times as needed with
    different time grids.  This avoids repeating the expensive coefficient
    computation (~95 % of the work) for each evaluation.

    Args:
        y0: Initial GEqOE state vector ``[nu, q1, q2, p1, p2, Lr]``.
        p: Body constants as ``BodyConstants`` or ``(j2, re, mu)``.
        order: Taylor expansion order (1-4).

    Returns:
        Frozen coefficient container consumed by :func:`evaluate_taylor`.

    Raises:
        RuntimeError: If the staged coefficient backend did not populate
            required outputs.
    """
    order = _validate_order(order)

    # Build a temporary context with a single dummy dt (used only to size
    # arrays inside compute_coefficients_1 -- M = 1 is enough).
    dummy_dt = np.array([1.0])
    ctx = build_context(dt=dummy_dt, y0=y0, p=p, order=order)

    coeff_dispatch = {
        1: compute_coefficients_1,
        2: compute_coefficients_2,
        3: compute_coefficients_3,
        4: compute_coefficients_4,
    }
    coeff_dispatch[order](ctx)

    if ctx.map_components is None:
        raise RuntimeError("compute_coefficients did not populate map_components.")

    # Extract only the dt-independent (scalar) entries from scratch.
    # The dt-dependent STM accumulator arrays (shape (M,)) are NOT copied --
    # evaluate_taylor will recompute them for each new dt grid.
    dt_dependent_keys = {
        "nu_nu",
        "Lr_nu", "Lr_Lr", "Lr_q1", "Lr_q2", "Lr_p1", "Lr_p2",
        "q1_nu", "q1_Lr", "q1_q1", "q1_q2", "q1_p1", "q1_p2",
        "q2_nu", "q2_Lr", "q2_q1", "q2_q2", "q2_p1", "q2_p2",
        "p1_nu", "p1_Lr", "p1_q1", "p1_q2", "p1_p1", "p1_p2",
        "p2_nu", "p2_Lr", "p2_q1", "p2_q2", "p2_p1", "p2_p2",
    }
    static_scratch = {k: v for k, v in ctx.scratch.items() if k not in dt_dependent_keys}

    if isinstance(p, BodyConstants):
        j2, re, mu = p.j2, p.re, p.mu
    else:
        j2, re, mu = p

    return GEqOETaylorCoefficients(
        initial_geqoe=np.asarray(y0, dtype=float).flatten(),
        peq_py_0=np.zeros((6, 6)),  # not used in GEqOE-level evaluate; filled in cart wrapper
        constants=ctx.constants,
        order=order,
        scratch=static_scratch,
        map_components=ctx.map_components.copy(),
        initial_state=ctx.initial_state,
        body_params=p,
    )


_EVALUATE_DISPATCH = {
    1: evaluate_order_1,
    2: evaluate_order_2,
    3: evaluate_order_3,
    4: evaluate_order_4,
}


def evaluate_taylor(
    coeffs: GEqOETaylorCoefficients,
    dt: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the precomputed Taylor polynomial at new time offsets.

    This is the second half of the two-stage propagation API.  Given a
    :class:`GEqOETaylorCoefficients` object produced by
    :func:`prepare_taylor_coefficients`, this function performs only the cheap
    polynomial evaluation step (O(M)) for an arbitrary time grid ``dt``.

    Args:
        coeffs: Precomputed coefficients from :func:`prepare_taylor_coefficients`.
        dt: Time offset(s) from the epoch in seconds.

    Returns:
        Tuple ``(y_prop, y_y0, map_components)`` where ``y_prop`` are GEqOE
        states, ``y_y0`` are GEqOE STMs, and ``map_components`` is the Taylor
        coefficient matrix.
    """
    dt_arr = np.atleast_1d(np.asarray(dt, dtype=float))
    M = len(dt_arr)
    dt_norm = dt_arr / coeffs.constants.time_scale

    # Build a fresh context reusing the precomputed scratch (scalar values only).
    # y_prop and map_components are re-initialised for the new M.
    ctx = GEqOEPropagationContext(
        dt_seconds=dt_arr,
        dt_norm=dt_norm,
        initial_state=coeffs.initial_state,
        order=coeffs.order,
        constants=coeffs.constants,
        scratch=dict(coeffs.scratch),  # mutable copy; evaluate writes STM accumulators
        y_prop=np.zeros((M, 6)),
        map_components=coeffs.map_components.copy(),
    )

    _EVALUATE_DISPATCH[coeffs.order](ctx)

    # --- STM assembly (mirrors _run_staged_j2) ---
    T = coeffs.constants.time_scale
    s = ctx.scratch

    y_y0 = np.zeros((6, 6, M))
    y_y0[0, 0, :] = s["nu_nu"]
    y_y0[1, 0, :] = s["q1_nu"] * T;  y_y0[1, 1, :] = s["q1_q1"];  y_y0[1, 2, :] = s["q1_q2"]
    y_y0[1, 3, :] = s["q1_p1"];      y_y0[1, 4, :] = s["q1_p2"];  y_y0[1, 5, :] = s["q1_Lr"]
    y_y0[2, 0, :] = s["q2_nu"] * T;  y_y0[2, 1, :] = s["q2_q1"];  y_y0[2, 2, :] = s["q2_q2"]
    y_y0[2, 3, :] = s["q2_p1"];      y_y0[2, 4, :] = s["q2_p2"];  y_y0[2, 5, :] = s["q2_Lr"]
    y_y0[3, 0, :] = s["p1_nu"] * T;  y_y0[3, 1, :] = s["p1_q1"];  y_y0[3, 2, :] = s["p1_q2"]
    y_y0[3, 3, :] = s["p1_p1"];      y_y0[3, 4, :] = s["p1_p2"];  y_y0[3, 5, :] = s["p1_Lr"]
    y_y0[4, 0, :] = s["p2_nu"] * T;  y_y0[4, 1, :] = s["p2_q1"];  y_y0[4, 2, :] = s["p2_q2"]
    y_y0[4, 3, :] = s["p2_p1"];      y_y0[4, 4, :] = s["p2_p2"];  y_y0[4, 5, :] = s["p2_Lr"]
    y_y0[5, 0, :] = s["Lr_nu"] * T;  y_y0[5, 1, :] = s["Lr_q1"];  y_y0[5, 2, :] = s["Lr_q2"]
    y_y0[5, 3, :] = s["Lr_p1"];      y_y0[5, 4, :] = s["Lr_p2"];  y_y0[5, 5, :] = s["Lr_Lr"]

    ctx.y_prop[:, 0] /= T
    ctx.y_y0 = y_y0

    return ctx.y_prop, y_y0, ctx.map_components


def prepare_cart_coefficients(
    y0_cart: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
    order: int = 4,
) -> Tuple[GEqOETaylorCoefficients, np.ndarray]:
    """Precompute all dt-independent quantities for a Cartesian initial state.

    This is the Cartesian-space equivalent of :func:`prepare_taylor_coefficients`.
    It performs the ``rv2geqoe`` conversion and the epoch Jacobian
    ``d(GEqOE)/d(Cartesian)`` *once*, then delegates to
    :func:`prepare_taylor_coefficients` for the Taylor coefficient computation.

    Use this together with :func:`evaluate_cart_taylor` when the same initial
    Cartesian state will be propagated to many different time grids.

    Args:
        y0_cart: Initial Cartesian state ``[rx, ry, rz, vx, vy, vz]`` in SI.
        p: Body constants as ``BodyConstants`` or ``(j2, re, mu)``.
        order: Taylor expansion order (1-4).

    Returns:
        Tuple ``(coeffs, peq_py_0)`` containing GEqOE Taylor coefficients and
        the epoch Jacobian ``d(GEqOE)/d(Cartesian)``.

    Raises:
        ValueError: If ``y0_cart`` is not a 6-element state vector.
    """
    y0_flat = np.asarray(y0_cart, dtype=float).flatten()
    if y0_flat.shape != (6,):
        raise ValueError("y0_cart must be a 6-element Cartesian state vector.")

    eq0_tuple = rv2geqoe(t=0.0, y=y0_flat, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    peq_py_0 = get_pEqpY(t=0.0, y=y0_flat, p=p)
    if peq_py_0.ndim == 3:
        peq_py_0 = peq_py_0[0, :, :]

    coeffs = prepare_taylor_coefficients(y0=eq0, p=p, order=order)
    return coeffs, peq_py_0


def evaluate_cart_taylor(
    coeffs: GEqOETaylorCoefficients,
    peq_py_0: np.ndarray,
    tspan: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the precomputed Taylor polynomial and convert to Cartesian.

    Cheap second half of the two-stage Cartesian propagation API.  Calls
    :func:`evaluate_taylor` for the GEqOE polynomial evaluation, then
    applies the ``geqoe2rv`` conversion and composes the full Cartesian
    State Transition Matrix using the cached epoch Jacobian ``peq_py_0``.

    Args:
        coeffs: Precomputed coefficients from :func:`prepare_cart_coefficients`.
        peq_py_0: Cached epoch Jacobian ``d(GEqOE)/d(Cartesian)`` (6x6).
        tspan: Time offset(s) from the epoch in seconds.

    Returns:
        Tuple ``(y_out, dy_dy0)`` with Cartesian states and Cartesian STMs.
    """
    tspan = np.atleast_1d(np.asarray(tspan, dtype=float))
    p = coeffs.body_params

    eq_taylor, eq_eq0, _ = evaluate_taylor(coeffs, tspan)

    py_peq = get_pYpEq(t=tspan, y=eq_taylor, p=p)
    rv_prop, rpv_prop = geqoe2rv(t=tspan, y=eq_taylor, p=p)
    y_out = np.hstack((rv_prop, rpv_prop))

    n_steps = len(tspan)
    dy_dy0 = np.zeros((6, 6, n_steps))
    for i in range(n_steps):
        dy_dy0[:, :, i] = py_peq[i, :, :] @ eq_eq0[:, :, i] @ peq_py_0

    return y_out, dy_dy0


def j2_taylor_propagator(
    dt: Union[float, np.ndarray],
    y0: np.ndarray,
    p: Union[BodyConstants, tuple, list],
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot GEqOE-space J2 Taylor propagation.

    Args:
        dt: Time offset(s) from epoch in seconds.
        y0: Initial GEqOE state vector ``(6,)``.
        p: Body constants as ``BodyConstants`` or ``(j2, re, mu)``.
        order: Taylor expansion order (1-4).

    Returns:
        Tuple ``(y_prop, y_y0, map_components)`` in GEqOE coordinates.
    """
    context = build_context(dt=dt, y0=y0, p=p, order=order)
    return _run_staged_j2(context)


def taylor_cart_propagator(
    tspan: np.ndarray,
    y0: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot Cartesian J2 Taylor propagation with Cartesian STM output.

    Args:
        tspan: Time offset(s) from epoch in seconds.
        y0: Initial Cartesian state vector ``[rx, ry, rz, vx, vy, vz]``.
        p: Body constants as ``BodyConstants`` or ``(j2, re, mu)``.
        order: Taylor expansion order (1-4).

    Returns:
        Tuple ``(y_out, dy_dy0)`` with Cartesian states ``(N, 6)`` and STMs
        ``(6, 6, N)``.

    Raises:
        ValueError: If ``y0`` is not a 6-element vector.
    """
    y0_flat = np.asarray(y0, dtype=float).flatten()
    if y0_flat.shape != (6,):
        raise ValueError("y0 must be a 6-element state vector [rx, ry, rz, vx, vy, vz].")

    tspan = np.atleast_1d(np.asarray(tspan, dtype=float))

    eq0_tuple = rv2geqoe(t=0.0, y=y0_flat, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])
    eq_taylor, eq_eq0, _ = j2_taylor_propagator(dt=tspan, y0=eq0, p=p, order=order)
    peq_py_0 = get_pEqpY(t=0.0, y=y0_flat, p=p)
    if peq_py_0.ndim == 3:
        peq_py_0 = peq_py_0[0, :, :]
    py_peq = get_pYpEq(t=tspan, y=eq_taylor, p=p)

    rv_prop, rpv_prop = geqoe2rv(t=tspan, y=eq_taylor, p=p)
    y_out = np.hstack((rv_prop, rpv_prop))

    n_steps = len(tspan)
    dy_dy0 = np.zeros((6, 6, n_steps))
    for i in range(n_steps):
        dy_dy0[:, :, i] = py_peq[i, :, :] @ eq_eq0[:, :, i] @ peq_py_0

    return y_out, dy_dy0
