"""Mean-element propagation for the Lara-Brouwer analytical theory.

At first order, the mean Delaunay elements evolve with constant rates
(secular for J2/J4, secular + long-period for J3/J5).  For the long-period
J3/J5 terms (which depend on omega), we use "frozen" rates evaluated at
the initial mean state — valid for arcs where omega doesn't change much.

For longer arcs, the ODE integrator re-evaluates J3/J5 rates at each step.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.integrate import solve_ivp

from .coordinates import (
    eccentric_to_true,
    solve_kepler,
)


# ---------------------------------------------------------------------------
#  J2 secular rates (exact first-order, Delaunay variables)
# ---------------------------------------------------------------------------

def secular_rates_j2(L, G, H, mu, Re, J2):
    """First-order J2 secular rates for Delaunay variables.

    Returns (dl_dt, dg_dt, dh_dt, dL_dt, dG_dt, dH_dt).
    """
    a = L**2 / mu
    eta = G / L
    p = a * eta**2
    cos_i = H / G
    n = np.sqrt(mu / a**3)
    gamma2 = J2 * Re**2 / (2.0 * p**2)

    dl_dt = n * (1.0 + 1.5 * gamma2 * eta * (3.0 * cos_i**2 - 1.0))
    dg_dt = n * 1.5 * gamma2 * (5.0 * cos_i**2 - 1.0)
    dh_dt = -3.0 * n * gamma2 * cos_i

    return dl_dt, dg_dt, dh_dt, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
#  J3-J5 secular/long-period rates via numerical averaging
# ---------------------------------------------------------------------------

def _orbit_averaged_Rn(a, e, inc, omega, mu, Re, Jn, n_deg, n_quad=64):
    """Compute <R_n> = (1/2pi) integral_0^{2pi} R_n dl using Gauss-Legendre."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    M_pts = np.pi * (nodes + 1.0)
    w_pts = np.pi * weights

    E_pts = solve_kepler(M_pts, e)
    f_pts = eccentric_to_true(E_pts, e)
    p = a * (1.0 - e**2)

    Rn_vals = np.empty(len(f_pts))
    for k, fk in enumerate(f_pts):
        r = p / (1.0 + e * np.cos(fk))
        sin_phi = np.sin(inc) * np.sin(omega + fk)
        coeffs = np.zeros(n_deg + 1)
        coeffs[n_deg] = 1.0
        Pn = legval(sin_phi, coeffs)
        Rn_vals[k] = -(mu / r) * Jn * (Re / r) ** n_deg * Pn

    return np.sum(w_pts * Rn_vals) / (2.0 * np.pi)


def compute_jn_rates(L, G, H, g, mu, Re, Jn, n_deg, n_quad=64):
    """Compute first-order Jn contribution to Delaunay rates via numerical averaging.

    Uses finite differences of the orbit-averaged disturbing function.
    Returns (dl_dt, dg_dt, dh_dt, dL_dt, dG_dt, dH_dt).
    """
    a = L**2 / mu
    e_sq = 1.0 - (G / L) ** 2
    e = np.sqrt(max(e_sq, 0.0))
    cos_i = np.clip(H / G, -1.0, 1.0)
    inc = np.arccos(cos_i)
    omega = g

    def avg(a_, e_, inc_, om_):
        return _orbit_averaged_Rn(a_, e_, inc_, om_, mu, Re, Jn, n_deg, n_quad)

    # Finite-difference step sizes
    eps_a = a * 1e-8
    eps_e = max(e * 1e-8, 1e-12)
    eps_i = 1e-8
    eps_om = 1e-8

    dRn_da = (avg(a + eps_a, e, inc, omega) - avg(a - eps_a, e, inc, omega)) / (2.0 * eps_a)
    dRn_de = (avg(a, e + eps_e, inc, omega) - avg(a, e - eps_e, inc, omega)) / (2.0 * eps_e)
    dRn_di = (avg(a, e, inc + eps_i, omega) - avg(a, e, inc - eps_i, omega)) / (2.0 * eps_i)
    dRn_dom = (avg(a, e, inc, omega + eps_om) - avg(a, e, inc, omega - eps_om)) / (2.0 * eps_om)

    # Chain rule to Delaunay partials
    da_dL = 2.0 * L / mu
    eta = np.sqrt(max(1.0 - e**2, 0.0))
    if e > 1e-12:
        de_dL = eta**2 / (e * L)
        de_dG = -eta / (e * L)
    else:
        de_dL = 0.0
        de_dG = 0.0

    sin_i = np.sin(inc)
    if abs(sin_i) > 1e-15:
        di_dG = cos_i / (G * sin_i)
        di_dH = -1.0 / (G * sin_i)
    else:
        di_dG = 0.0
        di_dH = 0.0

    dRn_dL = dRn_da * da_dL + dRn_de * de_dL
    dRn_dG = dRn_de * de_dG + dRn_di * di_dG
    dRn_dH = dRn_di * di_dH

    # Delaunay equations: dq/dt = dH/dp, dp/dt = -dH/dq
    dl_dt = dRn_dL
    dg_dt = dRn_dG
    dh_dt = dRn_dH
    dL_dt = 0.0
    dG_dt = -dRn_dom
    dH_dt = 0.0

    return dl_dt, dg_dt, dh_dt, dL_dt, dG_dt, dH_dt


# ---------------------------------------------------------------------------
#  Complete mean-element ODE (J2 analytical + J3-J5 frozen rates)
# ---------------------------------------------------------------------------

def compute_all_rates(y, mu, Re, j_coeffs):
    """Compute total Delaunay rates at state y = [ell, g, h, L, G, H]."""
    ell, g, h, L, G, H = y

    rates = np.array(secular_rates_j2(L, G, H, mu, Re, j_coeffs[2]), dtype=float)

    for n_deg in [3, 4, 5]:
        if n_deg in j_coeffs and j_coeffs[n_deg] != 0.0:
            r_n = compute_jn_rates(L, G, H, g, mu, Re, j_coeffs[n_deg], n_deg)
            rates += np.array(r_n, dtype=float)

    return rates


def propagate_mean_delaunay(y0, t_array, mu, Re, j_coeffs, rtol=1e-12, atol=1e-14):
    """Integrate the mean Delaunay ODE.

    Uses J2 analytical rates (recomputed at each step) plus J3-J5 rates
    frozen at the initial state (fast approximation valid at first order).

    Returns (N, 6) array of Delaunay states.
    """
    y0 = np.asarray(y0, dtype=float)

    # Pre-compute J3-J5 rates at initial state (frozen)
    ell0, g0, h0, L0, G0, H0 = y0
    frozen_rates = np.zeros(6)
    for n_deg in [3, 4, 5]:
        if n_deg in j_coeffs and j_coeffs[n_deg] != 0.0:
            r_n = compute_jn_rates(L0, G0, H0, g0, mu, Re, j_coeffs[n_deg], n_deg)
            frozen_rates += np.array(r_n, dtype=float)

    def rhs(t, y):
        ell, g, h, L, G, H = y
        j2_rates = np.array(secular_rates_j2(L, G, H, mu, Re, j_coeffs[2]), dtype=float)
        return j2_rates + frozen_rates

    sol = solve_ivp(
        rhs,
        (t_array[0], t_array[-1]),
        y0,
        method="DOP853",
        t_eval=t_array,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Mean-element integration failed: {sol.message}")

    return sol.y.T
