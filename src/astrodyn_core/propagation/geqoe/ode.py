"""GEqOE equations of motion for scipy numerical integration.

Provides the right-hand side of the J2-perturbed GEqOE ODE in normalized
units, suitable for ``scipy.integrate.solve_ivp``.  The formulas are
extracted directly from ``taylor_order_1.py`` (lines 46-143) as scalar
computation without derivative tracking.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.propagation.geqoe.utils import solve_kep_gen


def geqoe_rhs(t_norm: float, state: np.ndarray, j2: float) -> np.ndarray:
    """GEqOE equations of motion for the J2-perturbed problem.

    State vector: ``[nu, q1, q2, p1, p2, Lr]`` in normalized units
    (length = Re, time = sqrt(Re^3/mu), mu_norm = 1).

    Parameters
    ----------
    t_norm : float
        Normalized time (unused -- the system is autonomous).
    state : np.ndarray
        GEqOE state ``[nu, q1, q2, p1, p2, Lr]``.
    j2 : float
        J2 gravitational coefficient (dimensionless, positive).

    Returns
    -------
    np.ndarray
        Time derivatives ``d(state)/d(t_norm)``.
    """
    mu_norm = 1.0
    A = j2 / 2.0

    nu, q1, q2, p1, p2, Lr = state

    # Solve generalized Kepler equation (reduce Lr to [0, 2pi) for solver)
    Lr_mod = Lr % (2.0 * np.pi)
    K = solve_kep_gen(np.array([Lr_mod]), np.array([p1]), np.array([p2]))[0]
    sinK, cosK = np.sin(K), np.cos(K)

    q1s, q2s = q1**2, q2**2
    p1s, p2s = p1**2, p2**2
    beta = np.sqrt(1.0 - p1s - p2s)
    alpha = 1.0 / (1.0 + beta)

    # Semi-major axis and position geometry
    a = (mu_norm / nu**2) ** (1.0 / 3.0)
    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1s) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2s) * sinK - p1)

    r = a * (1.0 - p1 * sinK - p2 * cosK)
    cosL = X / r
    sinL = Y / r

    # Generalized angular momentum and J2 potential
    c = (mu_norm**2 / nu) ** (1.0 / 3.0) * beta
    zg = 2.0 * (Y * q2 - X * q1) / (r * (1.0 + q1s + q2s))
    U = -A / r**3 * (1.0 - 3.0 * zg**2)

    # Effective angular momentum
    h = np.sqrt(c**2 - 2.0 * r**2 * U)

    # Perturbation intermediates
    r2 = r**2
    delta = 1.0 - q1s - q2s
    hr3 = h * r**3
    I_val = 3.0 * A * zg * delta / hr3
    d = (h - c) / r2
    wh = I_val * zg

    # Auxiliary quantities
    fic = 1.0 / c
    xi1 = X / a + 2.0 * p2
    xi2 = Y / a + 2.0 * p1
    fialpha = 1.0 / alpha
    GAMMA = fialpha + alpha * (1.0 - r / a)

    # Equations of motion
    dnu = 0.0  # energy integral: exact constant of motion
    dq1 = -I_val * sinL
    dq2 = -I_val * cosL
    dp1 = p2 * (d - wh) - fic * xi1 * U
    dp2 = p1 * (wh - d) + fic * xi2 * U
    dLr = nu + d - wh - fic * GAMMA * U

    return np.array([dnu, dq1, dq2, dp1, dp2, dLr])


__all__ = ["geqoe_rhs"]
