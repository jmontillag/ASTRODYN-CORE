"""Cartesian <-> Generalized Equinoctial Orbital Elements (GEqOE) conversions.

Vectorized implementations following Giulio Bau's formulation with J2
perturbation effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from astrodyn_core.propagation.geqoe.utils import solve_kep_gen


@dataclass
class BodyConstants:
    """Physical constants of a celestial body for GEqOE calculations.

    Attributes
    ----------
    j2 : float
        J2 gravitational coefficient (dimensionless, positive).
    re : float
        Equatorial radius in metres.
    mu : float
        Gravitational parameter (GM) in m^3/s^2.
    """

    j2: float
    re: float
    mu: float


def rv2geqoe(
    t: float,
    y: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian states to GEqOE.

    Parameters
    ----------
    t : float
        Time in seconds (kept for interface consistency; unused).
    y : array, shape (N, 6) or (6,)
        Cartesian states ``[rx, ry, rz, vx, vy, vz]`` in SI units.
    p : BodyConstants or (J2, Re, mu)
        Central body constants.

    Returns
    -------
    (nu, q1, q2, p1, p2, Lr) : tuple of arrays, each of length N
    """
    if isinstance(p, BodyConstants):
        J2, Re, mu = p.j2, p.re, p.mu
    else:
        J2, Re, mu = p

    y = np.atleast_2d(y)
    if y.shape[1] != 6:
        raise ValueError("Input state vector 'y' must have 6 columns [rx,ry,rz,vx,vy,vz].")

    # Normalisation
    L = Re
    T = (Re**3 / mu) ** 0.5
    mu_norm = 1.0
    A = J2 / 2.0

    rv = y[:, :3] / L
    rpv = y[:, 3:6] / L * T

    # Geometric quantities
    r2 = np.sum(rv**2, axis=1)
    r = np.sqrt(r2)
    rp = np.sum(rv * rpv, axis=1) / r
    v2 = np.sum(rpv**2, axis=1)

    # J2 potential
    zg = rv[:, 2] / r
    U = -A / r**3 * (1 - 3 * zg**2)

    # Orbital frame
    er = rv / r[:, np.newaxis]
    ez = np.array([0.0, 0.0, 1.0])

    hv = np.cross(rv, rpv)
    h2 = np.sum(hv**2, axis=1)
    h = np.sqrt(h2)
    Ueff = h2 / (2 * r2) + U

    eh = hv / h[:, np.newaxis]
    ef = np.cross(eh, er)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])

    # Total energy & nu
    E_val = v2 / 2 - mu_norm / r + U
    nu_norm = (1 / mu_norm) * (-2 * E_val) ** 1.5

    # Orientation parameters
    ehez = np.dot(eh, ez)
    q1 = np.dot(eh, ex) / (1 + ehez)
    q2 = -np.dot(eh, ey) / (1 + ehez)

    # Equinoctial frame
    q1s = q1**2
    q2s = q2**2
    q1q2 = q1 * q2
    eTerm = 1 / (1 + q1s + q2s)
    eX = eTerm[:, np.newaxis] * np.vstack([1 - q1s + q2s, 2 * q1q2, -2 * q1]).T
    eY = eTerm[:, np.newaxis] * np.vstack([2 * q1q2, 1 + q1s - q2s, 2 * q2]).T

    # Generalized orbital elements
    c = np.sqrt(2 * r2 * Ueff)
    gv = rp[:, np.newaxis] * er + c[:, np.newaxis] / r[:, np.newaxis] * ef
    g_num = np.cross(gv, np.cross(rv, gv)) - mu_norm * er
    g = g_num / mu_norm

    p1 = np.sum(g * eY, axis=1)
    p2 = np.sum(g * eX, axis=1)
    p1s = p1**2
    p2s = p2**2

    X = np.sum(rv * eX, axis=1)
    Y = np.sum(rv * eY, axis=1)

    # True longitude
    beta = np.sqrt(1 - p1s - p2s)
    alpha = 1 / (1 + beta)
    a = (mu_norm / nu_norm**2) ** (1 / 3)

    cosK_term1 = (1 - alpha * p2s) * X
    cosK_term2 = alpha * p1 * p2 * Y
    cosK = p2 + (1 / (a * beta)) * (cosK_term1 - cosK_term2)

    sinK_term1 = (1 - alpha * p1s) * Y
    sinK_term2 = alpha * p1 * p2 * X
    sinK = p1 + (1 / (a * beta)) * (sinK_term1 - sinK_term2)

    Lr = np.arctan2(sinK, cosK) + (1 / (a * beta)) * (X * p1 - Y * p2)

    nu_out = nu_norm / T
    return nu_out, q1, q2, p1, p2, Lr


def geqoe2rv(
    t: float,
    y: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert GEqOE to Cartesian states.

    Parameters
    ----------
    t : float
        Time in seconds (kept for interface consistency; unused).
    y : array, shape (N, 6) or (6,)
        GEqOE states ``[nu, q1, q2, p1, p2, Lr]``.
    p : BodyConstants or (J2, Re, mu)
        Central body constants.

    Returns
    -------
    (rv, rpv) : tuple of arrays, shapes (N, 3) each
        Position (m) and velocity (m/s).
    """
    if isinstance(p, BodyConstants):
        J2, Re, mu = p.j2, p.re, p.mu
    else:
        J2, Re, mu = p

    y_in = np.atleast_2d(y)
    if y_in.shape[1] != 6:
        raise ValueError("Input state vector 'y' must have 6 columns [nu,q1,q2,p1,p2,Lr].")

    L = Re
    T = (Re**3 / mu) ** 0.5
    mu_norm = 1.0
    A = J2 / 2.0

    nu_norm = y_in[:, 0].copy()
    q1, q2, p1, p2 = y_in[:, 1], y_in[:, 2], y_in[:, 3], y_in[:, 4]
    Lr_in = y_in[:, 5]
    nu_norm *= T
    Lr = np.mod(Lr_in, 2 * np.pi)

    K = solve_kep_gen(Lr, p1, p2)
    sinK = np.sin(K)
    cosK = np.cos(K)

    q1s, q2s, q1q2 = q1**2, q2**2, q1 * q2
    eTerm = 1 / (1 + q1s + q2s)
    eX = eTerm[:, np.newaxis] * np.vstack([1 - q1s + q2s, 2 * q1q2, -2 * q1]).T
    eY = eTerm[:, np.newaxis] * np.vstack([2 * q1q2, 1 + q1s - q2s, 2 * q2]).T

    p1s, p2s = p1**2, p2**2
    gs = p1s + p2s
    beta = np.sqrt(1 - gs)
    alpha = 1 / (1 + beta)
    a = (mu_norm / nu_norm**2) ** (1 / 3)

    X = a * (alpha * p1 * p2 * sinK + (1 - alpha * p1s) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1 - alpha * p2s) * sinK - p1)

    rv_norm = X[:, np.newaxis] * eX + Y[:, np.newaxis] * eY

    z = rv_norm[:, 2]
    r = a * (1 - p1 * sinK - p2 * cosK)
    rp = np.sqrt(mu_norm * a) / r * (p2 * sinK - p1 * cosK)

    cosL = X / r
    sinL = Y / r
    c = (mu_norm**2 / nu_norm) ** (1 / 3) * beta

    zg = z / r
    U = -A / r**3 * (1 - 3 * zg**2)
    h = np.sqrt(c**2 - 2 * r**2 * U)

    Xp = rp * cosL - h / r * sinL
    Yp = rp * sinL + h / r * cosL

    rpv_norm = Xp[:, np.newaxis] * eX + Yp[:, np.newaxis] * eY

    rv_out = rv_norm * L
    rpv_out = rpv_norm * L / T

    return rv_out, rpv_out


__all__ = ["BodyConstants", "rv2geqoe", "geqoe2rv"]
