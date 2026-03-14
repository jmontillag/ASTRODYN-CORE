"""Mean-element propagation for the Lara-Brouwer analytical theory.

At first order, the mean Delaunay elements evolve with constant rates
(secular for J2, secular + long-period for J3/J5).  For the long-period
J3/J5 terms (which depend on omega), we use "frozen" rates evaluated at
the initial mean state — valid for arcs where omega doesn't change much.

Also provides ``secular_rates_brouwer`` with Brouwer second-order (J2² + J4)
secular corrections, though the propagation pipeline uses first-order J2
secular rates by default because without matching second-order short-period
corrections, the J2²/J4 secular terms degrade accuracy.  The correct J4
secular coefficients (verified against numerical quadrature) are documented
in ``secular_rates_brouwer`` for use in higher-order theories.

The preferred secular rate function for the {1+:2:1} theory is
``secular_rates_lara``, which derives rates from the totally reduced
averaged Hamiltonian through second order (Lara 2021, Eq. 14), computed
via central finite differences.  This is used by ``propagate_mean_delaunay``
by default.
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
#  Lara (2021) averaged Hamiltonian and second-order secular rates
# ---------------------------------------------------------------------------

def averaged_hamiltonian_H01(L, G, H, mu, Re):
    """First-order averaged Hamiltonian H₀,₁ (Lara 2021, page 5-6).

    H₀,₁ = H₀,₀ · (R⊕/p)² · η · (1 - 3s²/2)

    where H₀,₀ = -μ/(2a), p = a(1-e²) = aη², η = G/L, s = sin(i).
    """
    a = L**2 / mu
    eta = G / L
    p = a * eta**2
    s2 = 1.0 - (H / G) ** 2  # sin²(i) = 1 - cos²(i)
    H00 = -mu / (2.0 * a)
    return H00 * (Re / p) ** 2 * eta * (1.0 - 1.5 * s2)


def averaged_hamiltonian_H02(L, G, H, mu, Re):
    """Second-order averaged Hamiltonian H₀,₂ (Lara 2021, Eq. 14).

    H₀,₂ = H₀,₀ · (R⊕⁴/p⁴) · (3/(32η))
            · [5(7s⁴ - 16s² + 8) + η(6s² - 4)² + η²(5s⁴ + 8s² - 8)]
    """
    a = L**2 / mu
    eta = G / L
    p = a * eta**2
    s2 = 1.0 - (H / G) ** 2  # sin²(i)
    s4 = s2**2
    H00 = -mu / (2.0 * a)
    bracket = (5.0 * (7.0 * s4 - 16.0 * s2 + 8.0)
               + eta * (6.0 * s2 - 4.0) ** 2
               + eta**2 * (5.0 * s4 + 8.0 * s2 - 8.0))
    return H00 * (Re / p) ** 4 * (3.0 / (32.0 * eta)) * bracket


def total_averaged_hamiltonian(L, G, H, mu, Re, J2):
    """Total averaged Hamiltonian K = H₀,₀ + J₂·H₀,₁ + (J₂²/2)·H₀,₂.

    This is the completely reduced Hamiltonian through second order
    (Lara 2021, combining Eqs. on page 5-6 with Eq. 14).
    """
    # H₀,₀ = -μ/(2a) = -μ²/(2L²) since a = L²/μ
    H00 = -mu**2 / (2.0 * L**2)
    H01 = averaged_hamiltonian_H01(L, G, H, mu, Re)
    H02 = averaged_hamiltonian_H02(L, G, H, mu, Re)
    return H00 + J2 * H01 + 0.5 * J2**2 * H02


def secular_rates_lara(L, G, H, mu, Re, J2):
    """Second-order secular rates from Lara (2021) averaged Hamiltonian.

    Computes dl/dt = ∂K/∂L, dg/dt = ∂K/∂G, dh/dt = ∂K/∂H via central
    finite differences of the totally reduced Hamiltonian.

    Returns (dl_dt, dg_dt, dh_dt, 0, 0, 0).
    """
    eps_L = L * 1e-10
    eps_G = G * 1e-10
    eps_H = abs(H) * 1e-10 + 1e-15  # Avoid zero for equatorial

    K = total_averaged_hamiltonian

    dl_dt = (K(L + eps_L, G, H, mu, Re, J2)
             - K(L - eps_L, G, H, mu, Re, J2)) / (2.0 * eps_L)
    dg_dt = (K(L, G + eps_G, H, mu, Re, J2)
             - K(L, G - eps_G, H, mu, Re, J2)) / (2.0 * eps_G)
    dh_dt = (K(L, G, H + eps_H, mu, Re, J2)
             - K(L, G, H - eps_H, mu, Re, J2)) / (2.0 * eps_H)

    return dl_dt, dg_dt, dh_dt, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
#  Second-order secular rates: J2² + J4  (Brouwer / SGP4)
# ---------------------------------------------------------------------------

def secular_rates_brouwer(L, G, H, mu, Re, J2, J4):
    """Second-order (J2² + J4) secular rates for Delaunay variables.

    Includes first-order J2 rates plus second-order J2² corrections
    (from Brouwer 1959 / Spacetrack Report No. 3) and first-order J4
    secular contributions (from orbit-averaged R_4 disturbing function,
    verified numerically against Gauss-Legendre quadrature).

    The J4 secular coefficients are:
      dg: (15/16) - (105/8)*t² + (315/16)*t⁴
      dl: (15/32) - (15/4)*t² + (105/32)*t⁴  (times eta)
      dh: (45/16) - (105/16)*t²               (times theta)

    Returns (dl_dt, dg_dt, dh_dt, dL_dt, dG_dt, dH_dt).
    """
    a = L**2 / mu
    eta = G / L          # sqrt(1 - e²)
    eta2 = eta**2
    p = a * eta2
    theta = H / G        # cos(i)
    theta2 = theta**2
    theta4 = theta2**2
    n = np.sqrt(mu / a**3)

    gamma2 = J2 * Re**2 / (2.0 * p**2)
    gamma2_sq = gamma2**2

    # J4 prefactor: n * J4 * (Re/p)^4
    Re_over_p_4 = (Re / p) ** 4
    j4_pref = J4 * Re_over_p_4

    # Mean-motion rate (dl/dt) — first + second order J2² + first order J4
    dl_dt = n * (1.0
                 + 1.5 * gamma2 * eta * (3.0 * theta2 - 1.0)
                 + (3.0 / 16.0) * gamma2_sq * eta
                   * (13.0 - 78.0 * theta2 + 137.0 * theta4)
                 + j4_pref * eta
                   * (15.0 / 32.0 - (15.0 / 4.0) * theta2
                      + (105.0 / 32.0) * theta4))

    # Argument of perigee rate (dg/dt)
    dg_dt = n * (1.5 * gamma2 * (5.0 * theta2 - 1.0)
                 + (3.0 / 16.0) * gamma2_sq
                   * (7.0 - 114.0 * theta2 + 395.0 * theta4)
                 + j4_pref
                   * (15.0 / 16.0 - (105.0 / 8.0) * theta2
                      + (315.0 / 16.0) * theta4))

    # RAAN rate (dh/dt)
    dh_dt = n * (-3.0 * gamma2 * theta
                 + 1.5 * gamma2_sq * theta * (4.0 - 19.0 * theta2)
                 + j4_pref * theta
                   * (45.0 / 16.0 - (105.0 / 16.0) * theta2))

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
#  Complete mean-element ODE (J2 analytical + J3/J5 frozen rates)
# ---------------------------------------------------------------------------

def compute_all_rates(y, mu, Re, j_coeffs):
    """Compute total Delaunay rates at state y = [ell, g, h, L, G, H].

    Uses first-order J2 secular rates plus J3/J5 numerical rates.
    J4 is deliberately excluded: without matching J4 short-period
    corrections, the first-order J4 secular rate degrades accuracy.
    """
    ell, g, h, L, G, H = y

    rates = np.array(secular_rates_j2(L, G, H, mu, Re, j_coeffs[2]), dtype=float)

    for n_deg in [3, 5]:
        if n_deg in j_coeffs and j_coeffs[n_deg] != 0.0:
            r_n = compute_jn_rates(L, G, H, g, mu, Re, j_coeffs[n_deg], n_deg)
            rates += np.array(r_n, dtype=float)

    return rates


def propagate_mean_delaunay(y0, t_array, mu, Re, j_coeffs, rtol=1e-12, atol=1e-14,
                            dl_bv_correction=0.0):
    """Integrate the mean Delaunay ODE.

    Uses Lara (2021) second-order secular rates derived from the totally
    reduced averaged Hamiltonian (recomputed at each step) plus J3/J5
    rates frozen at the initial state (fast approximation valid at first
    order).  J4 secular rates are excluded because without matching J4
    short-period corrections they degrade accuracy.

    The optional ``dl_bv_correction`` adds a constant correction to
    dl/dt from the Breakwell-Vagners energy calibration (Lara 2021,
    Eq. 23).  This corrects the Keplerian part of the mean motion
    without modifying the mean elements themselves.

    Returns (N, 6) array of Delaunay states.
    """
    y0 = np.asarray(y0, dtype=float)

    # Pre-compute J3/J5 rates at initial state (frozen)
    ell0, g0, h0, L0, G0, H0 = y0
    frozen_rates = np.zeros(6)
    for n_deg in [3, 5]:
        if n_deg in j_coeffs and j_coeffs[n_deg] != 0.0:
            r_n = compute_jn_rates(L0, G0, H0, g0, mu, Re, j_coeffs[n_deg], n_deg)
            frozen_rates += np.array(r_n, dtype=float)

    # Pre-compute Lara second-order secular rates at initial state (frozen).
    # These are constant for a secular-only theory (L, G, H don't change
    # at first order; their rates are zero).
    lara_rates_0 = np.array(
        secular_rates_lara(L0, G0, H0, mu, Re, j_coeffs[2]), dtype=float)

    # BV correction goes into the dl/dt component only
    total_frozen = lara_rates_0 + frozen_rates
    total_frozen[0] += dl_bv_correction

    # Constant rates -> linear propagation (no ODE solver needed)
    dt = t_array - t_array[0]
    return y0[np.newaxis, :] + np.outer(dt, total_frozen)
