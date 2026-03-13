"""Vectorized GEqOE <-> Cartesian conversions for zonal perturbations.

Stage A performance optimization: replaces N scalar geqoe2cart calls with
a single batch numpy call.  The zonal potential depends only on (r, zhat),
so the full 3D gradient / U_numeric callback is unnecessary.
"""

from __future__ import annotations

import numpy as np


def _legendre_P_arr(n: int, x: np.ndarray) -> np.ndarray:
    """Legendre polynomial P_n(x) via Bonnet recurrence (array-safe)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_prev = np.ones_like(x)
    P_curr = x.copy()
    for k in range(2, n + 1):
        P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
        P_prev, P_curr = P_curr, P_next
    return P_curr


def geqoe2cart_zonal_batch(
    states: np.ndarray,
    mu: float,
    zonal_pert,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch GEqOE -> Cartesian for zonal perturbations.

    Parameters
    ----------
    states : (N, 6) array of [nu, p1, p2, K, q1, q2]
    mu : gravitational parameter [km^3/s^2]
    zonal_pert : ZonalPerturbation instance (uses ``._Cn`` dict)

    Returns
    -------
    r_vecs : (N, 3) position [km]
    v_vecs : (N, 3) velocity [km/s]
    """
    nu = states[:, 0]
    p1 = states[:, 1]
    p2 = states[:, 2]
    K = states[:, 3]
    q1 = states[:, 4]
    q2 = states[:, 5]

    # Equinoctial frame vectors (Eq. 37) — each component is (N,)
    q1s = q1 * q1
    q2s = q2 * q2
    q1q2 = q1 * q2
    gamma_inv = 1.0 / (1.0 + q1s + q2s)

    eX = np.column_stack([
        gamma_inv * (1.0 - q1s + q2s),
        gamma_inv * (2.0 * q1q2),
        gamma_inv * (-2.0 * q1),
    ])
    eY = np.column_stack([
        gamma_inv * (2.0 * q1q2),
        gamma_inv * (1.0 + q1s - q2s),
        gamma_inv * (2.0 * q2),
    ])

    # Shape quantities (Eq. 21, 40)
    g2 = p1 * p1 + p2 * p2
    beta = np.sqrt(1.0 - g2)
    alpha = 1.0 / (1.0 + beta)
    a = (mu / (nu * nu)) ** (1.0 / 3.0)

    # Position from K (Eq. 42)
    sinK = np.sin(K)
    cosK = np.cos(K)

    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)

    # Cartesian position: r_vec[i] = X[i]*eX[i] + Y[i]*eY[i]
    r_vecs = X[:, None] * eX + Y[:, None] * eY

    # Distance and radial velocity (Eq. 31-32)
    r = a * (1.0 - p1 * sinK - p2 * cosK)
    sqrt_mu_a = np.sqrt(mu * a)
    rdot = sqrt_mu_a / r * (p2 * sinK - p1 * cosK)

    # True longitude trig
    cosL = X / r
    sinL = Y / r

    # Generalized angular momentum (Eq. 23)
    c = (mu * mu / nu) ** (1.0 / 3.0) * beta

    # Zonal potential U = sum_n Cn * r^-(n+1) * P_n(zhat)
    zhat = r_vecs[:, 2] / r
    U = np.zeros_like(r)
    for n_deg in sorted(zonal_pert._Cn):
        Cn = zonal_pert._Cn[n_deg]
        Pn = _legendre_P_arr(n_deg, zhat)
        U += Cn * r ** (-(n_deg + 1)) * Pn

    # Physical angular momentum (Eq. 44)
    h = np.sqrt(c * c - 2.0 * r * r * U)

    # Velocity components (Eq. 43)
    Xdot = rdot * cosL - (h / r) * sinL
    Ydot = rdot * sinL + (h / r) * cosL

    v_vecs = Xdot[:, None] * eX + Ydot[:, None] * eY

    return r_vecs, v_vecs
