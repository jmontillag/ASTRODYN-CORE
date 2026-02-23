from typing import Literal, Tuple, Union

import numpy as np

from astrodyn_core.propagation.geqoe.conversion import BodyConstants, geqoe2rv, rv2geqoe
from astrodyn_core.propagation.geqoe._legacy_loader import import_legacy_module
from astrodyn_core.propagation.geqoe.jacobians import get_pEqpY, get_pYpEq
from astrodyn_core.propagation.geqoe.state import (
    GEqOEPropagationConstants,
    GEqOEPropagationContext,
    GEqOEState,
)
from astrodyn_core.propagation.geqoe.taylor_order_1 import compute_order_1
from astrodyn_core.propagation.geqoe.taylor_order_2 import compute_order_2
from astrodyn_core.propagation.geqoe.taylor_order_3 import compute_order_3
from astrodyn_core.propagation.geqoe.taylor_order_4 import compute_order_4


BackendKind = Literal["legacy", "staged"]


def _legacy_propagator_module():
    return import_legacy_module("temp_mosaic_modules.geqoe_utils.propagator")


def _validate_order(order: int) -> int:
    order_int = int(order)
    if order_int < 1 or order_int > 4:
        raise ValueError("Taylor order must be an integer in the range [1, 4].")
    return order_int


def _validate_backend(backend: str) -> BackendKind:
    if backend not in {"legacy", "staged"}:
        raise ValueError("backend must be either 'legacy' or 'staged'.")
    return backend  # type: ignore[return-value]


def build_context(
    dt: Union[float, np.ndarray],
    y0: np.ndarray,
    p: Union[BodyConstants, tuple, list],
    order: int,
) -> GEqOEPropagationContext:
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


def j2_taylor_propagator(
    dt: Union[float, np.ndarray],
    y0: np.ndarray,
    p: Union[BodyConstants, tuple, list],
    order: int = 4,
    backend: BackendKind = "legacy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    backend = _validate_backend(backend)
    context = build_context(dt=dt, y0=y0, p=p, order=order)
    if backend == "staged":
        return _run_staged_j2(context)

    legacy = _legacy_propagator_module()
    return legacy.j2_taylor_propagator(dt=dt, y0=y0, p=p, order=context.order)


def taylor_cart_propagator(
    tspan: np.ndarray,
    y0: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
    order: int = 4,
    backend: BackendKind = "legacy",
) -> Tuple[np.ndarray, np.ndarray]:
    backend = _validate_backend(backend)
    y0_flat = np.asarray(y0, dtype=float).flatten()
    if y0_flat.shape != (6,):
        raise ValueError("y0 must be a 6-element state vector [rx, ry, rz, vx, vy, vz].")

    tspan = np.atleast_1d(np.asarray(tspan, dtype=float))
    if backend == "legacy":
        legacy = _legacy_propagator_module()
        return legacy.taylor_cart_propagator(tspan=tspan, Y0=y0_flat, p=p, order=_validate_order(order))

    eq0_tuple = rv2geqoe(t=0.0, y=y0_flat, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])
    eq_taylor, eq_eq0, _ = j2_taylor_propagator(dt=tspan, y0=eq0, p=p, order=order, backend=backend)
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
