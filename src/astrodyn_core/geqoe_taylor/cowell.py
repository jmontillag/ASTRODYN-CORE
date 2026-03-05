"""Cowell (Cartesian) J2 propagators for ground truth reference.

Two implementations:
  - propagate_cowell: scipy DOP853 (no extra dependencies)
  - propagate_cowell_heyoka: heyoka Taylor (highest accuracy)

Used to validate the GEqOE Taylor propagator independently of the
paper's Dromo reference.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from astrodyn_core.geqoe_taylor.constants import MU, A_J2


def _cowell_rhs(t: float, y: np.ndarray, mu: float, A: float) -> np.ndarray:
    """Cartesian equations of motion with J2 (for scipy).

    State: [x, y, z, vx, vy, vz] in km, km/s.
    """
    r_vec = y[:3]
    v_vec = y[3:]

    r = np.linalg.norm(r_vec)
    r2 = r * r
    r5 = r2 * r2 * r
    x, y_c, z = r_vec
    zhat2 = (z / r) ** 2

    # Two-body + J2 acceleration
    # a = -mu/r^3 * r_vec - grad(U)
    # where U = -A/r^3*(1-3*zhat^2) is the paper's disturbing potential (energy).
    # The perturbation acceleration is -grad(U), i.e., force = -dU/dr.
    # Standard J2: a_x = -mu*x/r^3 - 3A*x/r^5*(1-5*z^2/r^2)
    #              a_z = -mu*z/r^3 - 3A*z/r^5*(3-5*z^2/r^2)
    mu_r3 = mu / (r2 * r)
    coeff = 3.0 * A / r5
    factor_xy = 1.0 - 5.0 * zhat2
    factor_z = 3.0 - 5.0 * zhat2

    ax = -mu_r3 * x - coeff * x * factor_xy
    ay = -mu_r3 * y_c - coeff * y_c * factor_xy
    az = -mu_r3 * z - coeff * z * factor_z

    return np.array([v_vec[0], v_vec[1], v_vec[2], ax, ay, az])


def propagate_cowell(
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    mu: float = MU,
    A: float = A_J2,
    rtol: float = 1e-14,
    atol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Cartesian state with J2 using DOP853.

    Args:
        r0: initial position (3,) in km.
        v0: initial velocity (3,) in km/s.
        t_final: propagation duration in seconds.
        mu: gravitational parameter.
        A: J2 convenience constant (mu*J2*Re^2/2).
        rtol, atol: integrator tolerances.

    Returns:
        (r_final, v_final): final position and velocity.
    """
    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(
        _cowell_rhs,
        [0.0, t_final],
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        args=(mu, A),
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Cowell integration failed: {sol.message}")

    return sol.y[:3, -1], sol.y[3:, -1]


def _build_cowell_heyoka_system(mu_val: float, A_val: float):
    """Build heyoka Cowell ODE system for J2."""
    import heyoka as hy

    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    r2 = x * x + y * y + z * z
    r = hy.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3
    zhat2 = (z * z) / r2

    mu = mu_val
    A = A_val
    mu_r3 = mu / r3
    coeff = 3.0 * A / r5
    fxy = 1.0 - 5.0 * zhat2
    fz = 3.0 - 5.0 * zhat2

    ax = -mu_r3 * x - coeff * x * fxy
    ay = -mu_r3 * y - coeff * y * fxy
    az = -mu_r3 * z - coeff * z * fz

    return [
        (x, vx), (y, vy), (z, vz),
        (vx, ax), (vy, ay), (vz, az),
    ]


def propagate_cowell_heyoka(
    r0: np.ndarray,
    v0: np.ndarray,
    t_final: float,
    mu: float = MU,
    A: float = A_J2,
    tol: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate Cartesian state with J2 using heyoka Taylor integrator.

    This provides the tightest possible ground truth since heyoka uses
    adaptive-order Taylor series (typically order 20+) with LLVM-compiled AD.

    Args:
        r0: initial position (3,) in km.
        v0: initial velocity (3,) in km/s.
        t_final: propagation duration in seconds.
        mu: gravitational parameter.
        A: J2 convenience constant.
        tol: Taylor integrator tolerance.

    Returns:
        (r_final, v_final): final position and velocity.
    """
    import heyoka as hy

    sys = _build_cowell_heyoka_system(mu, A)
    ic = list(r0) + list(v0)

    ta = hy.taylor_adaptive(sys, ic, tol=tol)
    ta.propagate_until(t_final)

    return ta.state[:3].copy(), ta.state[3:].copy()
