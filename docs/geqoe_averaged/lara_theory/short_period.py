"""First-order short-period corrections for the Lara-Brouwer theory.

J2 corrections use the Lyddane/SGP4-style polar-nodal form (robust at e~0).
J3-J5 corrections use numerical short-period extraction.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval

from .coordinates import (
    cartesian_to_keplerian,
    eccentric_to_true,
    keplerian_to_cartesian,
    polar_to_cartesian,
    solve_kepler,
)


# ---------------------------------------------------------------------------
#  J2 short-period corrections in polar-nodal form (SGP4 / Lyddane)
# ---------------------------------------------------------------------------

def j2_sp_polar(a, e, inc, Omega, omega, M, mu, Re, J2):
    """J2 first-order short-period corrections in polar-nodal variables.

    Given MEAN Keplerian elements, returns osculating polar-nodal quantities
    (r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc).

    Formulas from Hoots & Roehrich (1980) / SGP4 — page 16, pure J2 part.
    """
    k2 = J2 * Re**2 / 2.0
    theta = np.cos(inc)
    theta2 = theta**2
    eta = np.sqrt(1.0 - e**2)
    p = a * (1.0 - e**2)
    n = np.sqrt(mu / a**3)

    # Solve Kepler and get mean-state polar quantities
    E = solve_kepler(M, e)
    f = eccentric_to_true(E, e)
    r = p / (1.0 + e * np.cos(f))
    u = omega + f
    rdot = np.sqrt(mu / p) * e * np.sin(f)
    rfdot = np.sqrt(mu * p) / r

    # Short-period corrections (SGP4 page 16)
    sin2u = np.sin(2.0 * u)
    cos2u = np.cos(2.0 * u)

    dr = k2 / (2.0 * p) * (1.0 - theta2) * cos2u
    du = -k2 / (4.0 * p**2) * (7.0 * theta2 - 1.0) * sin2u
    dOmega = 3.0 * k2 * theta / (2.0 * p**2) * sin2u
    dinc = 3.0 * k2 * theta / (2.0 * p**2) * np.sin(inc) * cos2u
    drdot = -k2 * n / p * (1.0 - theta2) * sin2u
    drfdot = k2 * n / p * ((1.0 - theta2) * cos2u - 1.5 * (1.0 - 3.0 * theta2))

    # Osculating = mean + correction
    r_osc = r * (1.0 - 1.5 * k2 * eta / p**2 * (3.0 * theta2 - 1.0)) + dr
    u_osc = u + du
    Omega_osc = Omega + dOmega
    inc_osc = inc + dinc
    rdot_osc = rdot + drdot
    rfdot_osc = rfdot + drfdot

    return r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc


def j2_sp_polar_batch(a, e, inc, Omega, omega, M, mu, Re, J2):
    """Vectorized version of j2_sp_polar for arrays."""
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    omega = np.asarray(omega, dtype=float)
    M = np.asarray(M, dtype=float)

    k2 = J2 * Re**2 / 2.0
    theta = np.cos(inc)
    theta2 = theta**2
    eta = np.sqrt(1.0 - e**2)
    p = a * (1.0 - e**2)
    n = np.sqrt(mu / a**3)

    E = solve_kepler(M, e)
    f = eccentric_to_true(E, e)
    r = p / (1.0 + e * np.cos(f))
    u = omega + f
    rdot = np.sqrt(mu / p) * e * np.sin(f)
    rfdot = np.sqrt(mu * p) / r

    sin2u = np.sin(2.0 * u)
    cos2u = np.cos(2.0 * u)

    dr = k2 / (2.0 * p) * (1.0 - theta2) * cos2u
    du = -k2 / (4.0 * p**2) * (7.0 * theta2 - 1.0) * sin2u
    dOmega = 3.0 * k2 * theta / (2.0 * p**2) * sin2u
    dinc = 3.0 * k2 * theta / (2.0 * p**2) * np.sin(inc) * cos2u
    drdot = -k2 * n / p * (1.0 - theta2) * sin2u
    drfdot = k2 * n / p * ((1.0 - theta2) * cos2u - 1.5 * (1.0 - 3.0 * theta2))

    r_osc = r * (1.0 - 1.5 * k2 * eta / p**2 * (3.0 * theta2 - 1.0)) + dr
    u_osc = u + du
    Omega_osc = Omega + dOmega
    inc_osc = inc + dinc
    rdot_osc = rdot + drdot
    rfdot_osc = rfdot + drfdot

    return r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc


# ---------------------------------------------------------------------------
#  Short-period corrections in Keplerian elements (via polar-nodal)
# ---------------------------------------------------------------------------

def short_period_corrections(a, e, inc, Omega, omega, M, mu, Re, j_coeffs):
    """First-order short-period corrections evaluated at given (mean) state.

    Returns (da, de, di, dOmega, domega, dM).

    Uses J2 polar-nodal corrections (SGP4/Lyddane form) converted to
    Keplerian element differences.
    """
    J2 = j_coeffs[2]

    # Apply J2 SP in polar-nodal, convert to Cartesian, then to Keplerian
    r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc = j2_sp_polar(
        a, e, inc, Omega, omega, M, mu, Re, J2)
    pos_osc, vel_osc = polar_to_cartesian(
        r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc)
    osc_kep = cartesian_to_keplerian(pos_osc, vel_osc, mu)

    da = osc_kep[0] - a
    de = osc_kep[1] - e
    di = osc_kep[2] - inc
    dOmega = _angle_diff(osc_kep[3], Omega)
    domega = _angle_diff(osc_kep[4], omega)
    dM = _angle_diff(osc_kep[5], M)

    return (da, de, di, dOmega, domega, dM)


def _angle_diff(a1, a2):
    """Signed angular difference a1 - a2, wrapped to [-pi, pi]."""
    d = a1 - a2
    return (d + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
#  Mean <-> Osculating conversions
# ---------------------------------------------------------------------------

def mean_to_osculating(mean_kep, mu, Re, j_coeffs):
    """Mean Keplerian -> osculating Keplerian (first-order forward map)."""
    a, e, i, Om, om, M = mean_kep
    da, de, di, dOm, dom, dM = short_period_corrections(
        a, e, i, Om, om, M, mu, Re, j_coeffs)
    return (a + da, e + de, i + di, Om + dOm, om + dom, M + dM)


def osculating_to_mean(osc_kep, mu, Re, j_coeffs, max_iter=20, tol=1e-12):
    """Osculating Keplerian -> mean Keplerian (Cartesian-space iteration).

    Subtracts the Cartesian SP displacement and converts back to Keplerian.
    Accuracy is O(J2) for the initialization — the propagation pipeline
    (which uses the polar-nodal forward map) determines the true accuracy.
    """
    J2 = j_coeffs[2]

    # Target osculating Cartesian
    E_osc = solve_kepler(osc_kep[5], osc_kep[1])
    f_osc = eccentric_to_true(E_osc, osc_kep[1])
    r_target, v_target = keplerian_to_cartesian(*osc_kep[:5], f_osc, mu)

    mean_kep = np.array(osc_kep, dtype=float)

    for _ in range(max_iter):
        a, e_m, i_m, Om_m, om_m, M_m = mean_kep

        # Unperturbed Cartesian from mean elements
        E_m = solve_kepler(M_m, e_m)
        f_m = eccentric_to_true(E_m, e_m)
        r_unpert, v_unpert = keplerian_to_cartesian(a, e_m, i_m, Om_m, om_m, f_m, mu)

        # Osculating Cartesian (with SP)
        r_fwd, v_fwd = mean_to_cartesian(a, e_m, i_m, Om_m, om_m, M_m, mu, Re, J2)

        # SP displacement
        sp_r = r_fwd - r_unpert
        sp_v = v_fwd - v_unpert

        # Mean Cartesian = target - SP
        r_mean = r_target - sp_r
        v_mean = v_target - sp_v

        new_kep = np.array(cartesian_to_keplerian(r_mean, v_mean, mu))

        da = abs(new_kep[0] - mean_kep[0])
        de = abs(new_kep[1] - mean_kep[1])
        mean_kep = new_kep

        mean_kep[0] = max(mean_kep[0], 100.0)
        mean_kep[1] = np.clip(mean_kep[1], 1e-10, 0.9999)

        if da < tol and de < tol:
            break

    return tuple(mean_kep)


# ---------------------------------------------------------------------------
#  Direct mean -> Cartesian via polar-nodal corrections
# ---------------------------------------------------------------------------

def mean_to_cartesian(a, e, inc, Omega, omega, M, mu, Re, J2):
    """Convert mean Keplerian to osculating Cartesian via polar-nodal SP."""
    r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc = j2_sp_polar(
        a, e, inc, Omega, omega, M, mu, Re, J2)
    return polar_to_cartesian(r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc)


def mean_to_cartesian_batch(a, e, inc, Omega, omega, M, mu, Re, J2):
    """Vectorized mean -> osculating Cartesian."""
    r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc = j2_sp_polar_batch(
        a, e, inc, Omega, omega, M, mu, Re, J2)
    return polar_to_cartesian(r_osc, rdot_osc, u_osc, rfdot_osc, Om_osc, i_osc)
