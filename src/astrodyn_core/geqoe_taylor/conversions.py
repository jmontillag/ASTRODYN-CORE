"""Cartesian <-> GEqOE conversions using K (eccentric longitude).

State vector: [nu, p1, p2, K, q1, q2] in physical units (km, s).
Reference: Baù et al. (2021), Sections 3-4, Appendix B.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel


def cart2geqoe(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    mu: float,
    perturbation: PerturbationModel,
    t: float = 0.0,
) -> np.ndarray:
    """Convert Cartesian state to GEqOE with K as fast variable.

    Args:
        r_vec: position (3,) in km.
        v_vec: velocity (3,) in km/s.
        mu: gravitational parameter in km^3/s^2.
        perturbation: PerturbationModel providing U_numeric.
        t: epoch time in seconds (for time-dependent potentials).

    Returns:
        GEqOE state [nu, p1, p2, K, q1, q2].
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    # Geometric quantities
    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    rdot = np.dot(r_vec, v_vec) / r

    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Unit vectors
    er = r_vec / r
    eh = h_vec / h

    # Disturbing potential
    U = perturbation.U_numeric(r_vec, t)

    # Total energy (Eq. 16)
    E_val = v2 / 2.0 - mu / r + U

    # Generalized mean motion (Eq. 16 with a = (mu/nu^2)^(1/3))
    nu = (1.0 / mu) * (-2.0 * E_val) ** 1.5

    # Orientation parameters q1, q2 (Eq. 35-36)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    eh_dot_ez = np.dot(eh, ez)
    denom = 1.0 + eh_dot_ez
    q1 = np.dot(eh, ex) / denom
    q2 = -np.dot(eh, ey) / denom

    # Equinoctial frame (Eq. 37)
    q1s = q1**2
    q2s = q2**2
    q1q2 = q1 * q2
    gamma_inv = 1.0 / (1.0 + q1s + q2s)

    eX = gamma_inv * np.array([1.0 - q1s + q2s, 2.0 * q1q2, -2.0 * q1])
    eY = gamma_inv * np.array([2.0 * q1q2, 1.0 + q1s - q2s, 2.0 * q2])

    # Effective angular momentum quantities (Eq. 6, 7, 23)
    Ueff = h**2 / (2.0 * r**2) + U
    c = r * np.sqrt(2.0 * Ueff)

    # Generalized radial velocity and force vector (Eq. 10)
    ef = np.cross(eh, er)
    upsilon_vec = rdot * er + (c / r) * ef

    # Generalized Laplace vector g (Eq. 10)
    g_vec = np.cross(upsilon_vec, np.cross(r_vec, upsilon_vec)) / mu - er

    # Eccentricity-like projections
    p1 = np.dot(g_vec, eY)
    p2 = np.dot(g_vec, eX)

    # Position projections onto equinoctial frame
    X = np.dot(r_vec, eX)
    Y = np.dot(r_vec, eY)

    # Compute sinK, cosK (Eq. 39 inverted from Eq. 42)
    g2 = p1**2 + p2**2
    beta = np.sqrt(1.0 - g2)
    alpha = 1.0 / (1.0 + beta)
    a = (mu / nu**2) ** (1.0 / 3.0)
    ab = a * beta

    cosK = p2 + (1.0 / ab) * ((1.0 - alpha * p2**2) * X - alpha * p1 * p2 * Y)
    sinK = p1 + (1.0 / ab) * ((1.0 - alpha * p1**2) * Y - alpha * p1 * p2 * X)

    K = np.arctan2(sinK, cosK)

    return np.array([nu, p1, p2, K, q1, q2])


def geqoe2cart(
    state: np.ndarray,
    mu: float,
    perturbation: PerturbationModel,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GEqOE state to Cartesian.

    Args:
        state: GEqOE [nu, p1, p2, K, q1, q2].
        mu: gravitational parameter in km^3/s^2.
        perturbation: PerturbationModel providing U_numeric.
        t: epoch time in seconds.

    Returns:
        (r_vec, v_vec): position (3,) in km, velocity (3,) in km/s.
    """
    nu, p1, p2, K, q1, q2 = state

    # Equinoctial frame (Eq. 37)
    q1s = q1**2
    q2s = q2**2
    q1q2 = q1 * q2
    gamma_inv = 1.0 / (1.0 + q1s + q2s)

    eX = gamma_inv * np.array([1.0 - q1s + q2s, 2.0 * q1q2, -2.0 * q1])
    eY = gamma_inv * np.array([2.0 * q1q2, 1.0 + q1s - q2s, 2.0 * q2])

    # Shape quantities (Eq. 21, 40)
    g2 = p1**2 + p2**2
    beta = np.sqrt(1.0 - g2)
    alpha = 1.0 / (1.0 + beta)
    a = (mu / nu**2) ** (1.0 / 3.0)

    # Position from K (Eq. 42)
    sinK = np.sin(K)
    cosK = np.cos(K)

    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1**2) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2**2) * sinK - p1)

    r_vec = X * eX + Y * eY

    # Distance and radial velocity (Eq. 31-32)
    r = a * (1.0 - p1 * sinK - p2 * cosK)
    rdot = np.sqrt(mu * a) / r * (p2 * sinK - p1 * cosK)

    # True longitude trig
    cosL = X / r
    sinL = Y / r

    # Generalized angular momentum (Eq. 23)
    c = (mu**2 / nu) ** (1.0 / 3.0) * beta

    # Physical angular momentum h from c and U (Eq. 44)
    U = perturbation.U_numeric(r_vec, t)
    h = np.sqrt(c**2 - 2.0 * r**2 * U)

    # Velocity components (Eq. 43)
    Xdot = rdot * cosL - (h / r) * sinL
    Ydot = rdot * sinL + (h / r) * cosL

    v_vec = Xdot * eX + Ydot * eY

    return r_vec, v_vec
