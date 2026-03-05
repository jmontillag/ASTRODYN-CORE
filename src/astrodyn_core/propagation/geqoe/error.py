"""GEqOE Taylor truncation error estimation.

Provides utilities to estimate the position error of a Taylor polynomial
at a given time offset and to compute the maximum step size that keeps
the error below a user-specified tolerance.

The error model is the **embedded-pair** difference between the order-k
and order-(k-1) solutions:

    err(dt) = ||J_pos @ c_k|| * (dt/T)^k / k!

where ``c_k`` is the k-th raw derivative column in ``map_components``,
``J_pos`` is the 3×6 position block of the GEqOE-to-Cartesian Jacobian,
and the ``1/k!`` accounts for the factorial applied during Taylor
evaluation.
"""

from __future__ import annotations

import math

import numpy as np


def estimate_position_error(
    map_components: np.ndarray,
    pYpEq_epoch: np.ndarray,
    dt_seconds: float,
    time_scale: float,
    order: int,
) -> float:
    """Embedded-pair error estimate: ||y_k - y_{k-1}|| via Jacobian.

    Args:
        map_components: ``(6, order)`` Taylor coefficient matrix.
        pYpEq_epoch: ``(6, 6)`` GEqOE-to-Cartesian Jacobian at the expansion epoch.
        dt_seconds: Time offset in seconds.
        time_scale: Normalization time scale ``sqrt(re^3/mu)``.
        order: Taylor expansion order (1-4).

    Returns:
        Estimated position error in meters (scalar).
    """
    dt_norm = dt_seconds / time_scale
    J_pos = pYpEq_epoch[:3, :]
    delta_eq = map_components[:, order - 1] * dt_norm**order / math.factorial(order)
    return float(np.linalg.norm(J_pos @ delta_eq))


def compute_max_dt(
    map_components: np.ndarray,
    pYpEq_epoch: np.ndarray,
    time_scale: float,
    order: int,
    pos_tol: float,
    safety_factor: float = 0.8,
) -> float:
    """Compute max dt (seconds) that keeps position error below *pos_tol*.

    Standard step-size inversion of the embedded-pair error model::

        ||J_pos @ c_k|| / k! * (dt/T)^k = pos_tol
        → dt = T * (pos_tol * k! / ||J_pos @ c_k||)^(1/k) * safety

    Args:
        map_components: ``(6, order)`` Taylor coefficient matrix.
        pYpEq_epoch: ``(6, 6)`` GEqOE-to-Cartesian Jacobian at the expansion epoch.
        time_scale: Normalization time scale ``sqrt(re^3/mu)``.
        order: Taylor expansion order (1-4).
        pos_tol: Position tolerance in meters.
        safety_factor: Multiplicative safety margin (default 0.8).

    Returns:
        Maximum time offset in seconds.
    """
    J_pos = pYpEq_epoch[:3, :]
    pc_k = np.linalg.norm(J_pos @ map_components[:, order - 1])
    if pc_k == 0.0:
        return np.inf
    return float(
        time_scale
        * (pos_tol * math.factorial(order) / pc_k) ** (1.0 / order)
        * safety_factor
    )
