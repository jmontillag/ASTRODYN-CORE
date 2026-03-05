"""GEqOE symbolic RHS construction for heyoka.

Builds the GEqOE equations of motion as heyoka symbolic expressions.
State: [nu, p1, p2, K, q1, q2].

For J2-only, the simplified equations from Section 7.1 of
Baù et al. (2021) are used:
  - dnu/dt = 0  (energy is a first integral)
  - 2U - r*Fr = -U
  - dE/dt = 0
"""

from __future__ import annotations

import heyoka as hy

from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel


def build_geqoe_system(
    perturbation: PerturbationModel,
    mu_val: float | None = None,
    use_par: bool = True,
):
    """Build the J2-only GEqOE ODE system as heyoka expressions.

    Args:
        perturbation: PerturbationModel providing U_expr.
        mu_val: gravitational parameter (km^3/s^2). Required if use_par=False.
        use_par: if True, mu and A_J2 are runtime parameters (hy.par[]).
                 if False, they are compiled as constants.

    Returns:
        sys: list of (var, rhs) tuples for heyoka.taylor_adaptive.
        state_vars: list of 6 heyoka variable expressions.
        par_map: dict mapping parameter names to par[] indices.
    """
    # State variables
    nu, p1, p2, K, q1, q2 = hy.make_vars("nu", "p1", "p2", "K", "q1", "q2")
    state_vars = [nu, p1, p2, K, q1, q2]

    # Parameters
    if use_par:
        mu = hy.par[0]
        A = hy.par[1]
        par_map = {"mu": 0, "A_J2": 1}
    else:
        from astrodyn_core.geqoe_taylor.constants import MU, A_J2
        mu = float(mu_val if mu_val is not None else MU)
        A = float(A_J2)
        par_map = {}

    # --- Intermediate quantities ---

    # Semi-major axis (Eq. 21)
    a = (mu / (nu * nu)) ** (1.0 / 3.0)

    # Shape parameters (Eq. 22, 40)
    g2 = p1 * p1 + p2 * p2
    beta = hy.sqrt(1.0 - g2)
    alpha = 1.0 / (1.0 + beta)

    # Position from K (Eq. 42)
    sinK = hy.sin(K)
    cosK = hy.cos(K)

    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)

    # Orbital distance (Eq. 31)
    r = a * (1.0 - p1 * sinK - p2 * cosK)

    # True longitude trig
    cosL = X / r
    sinL = Y / r

    # z-hat (Eq. 57)
    gamma_inv = 1.0 / (1.0 + q1 * q1 + q2 * q2)
    zhat = 2.0 * (Y * q2 - X * q1) * gamma_inv / r

    # Generalized angular momentum (Eq. 23)
    c = (mu * mu / nu) ** (1.0 / 3.0) * beta

    # Velocity: w = sqrt(mu/a), used in K_dot
    w = hy.sqrt(mu / a)

    # Disturbing potential U (Eq. 56): built via perturbation model
    # For J2: U = -A/r^3 * (1 - 3*zhat^2)
    # Compute symbolically using equinoctial z-hat directly
    r3 = r * r * r
    U_val = -A / r3 * (1.0 - 3.0 * zhat * zhat)

    # Physical angular momentum (Eq. 44)
    h = hy.sqrt(c * c - 2.0 * r * r * U_val)

    # h - c computed stably (Section 2.5 of plan): -2r^2 U / (h + c)
    h_minus_c = -2.0 * r * r * U_val / (h + c)

    # delta = 1 - q1^2 - q2^2 (related to inclination)
    delta = 1.0 - q1 * q1 - q2 * q2

    # I factor (Eq. 58)
    I_val = 3.0 * A * zhat * delta / (h * r3)

    # w_h = I * zhat
    w_h = I_val * zhat

    # d = (h - c) / r^2
    d = h_minus_c / (r * r)

    # --- Assemble ODEs (J2-only, Section 7.1) ---

    # nu_dot = 0 (energy integral)
    nu_dot = 0.0 * nu  # keep as expression for heyoka

    # p1_dot (Eq. 47 simplified)
    xi1 = X / a + 2.0 * p2
    p1_dot = p2 * (d - w_h) - (1.0 / c) * xi1 * U_val

    # p2_dot (Eq. 48 simplified)
    xi2 = Y / a + 2.0 * p1
    p2_dot = p1 * (w_h - d) + (1.0 / c) * xi2 * U_val

    # K_dot (Eq. 75 with E_dot=0, 2U-rFr=-U)
    # Coefficient of U is (1/c)*(1 + alpha*(1 - r/a))
    # Use 1-r/a = p1*sinK + p2*cosK for stability (Section 2.5)
    one_minus_r_over_a = p1 * sinK + p2 * cosK
    K_dot = (w / r + d - w_h
             - (1.0 / c) * (1.0 + alpha * one_minus_r_over_a) * U_val)

    # q1_dot (Eq. 50 simplified)
    q1_dot = -I_val * sinL

    # q2_dot (Eq. 51 simplified)
    q2_dot = -I_val * cosL

    sys = [
        (nu, nu_dot),
        (p1, p1_dot),
        (p2, p2_dot),
        (K, K_dot),
        (q1, q1_dot),
        (q2, q2_dot),
    ]

    return sys, state_vars, par_map
