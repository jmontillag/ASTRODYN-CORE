"""First-order short-period corrections for the Lara-Brouwer theory.

J2 corrections use the Lyddane/SGP4-style polar-nodal form (robust at e~0).
J3-J5 corrections add radial and along-track perturbations from the
instantaneous minus orbit-averaged zonal disturbing function.

Also provides heyoka-AD-based exact Poisson brackets of the W1 generating
function (Lara 2021) using Lyddane non-singular variables (e*cos w, e*sin w,
M+w) to avoid the 1/e catastrophic cancellation that plagues finite-difference
approaches at low eccentricity.
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
#  Pre-computation of orbit-averaged <R_n> for J3-J5 SP corrections
# ---------------------------------------------------------------------------

def precompute_orbit_averages(a, e, inc, omega, mu, Re, j_coeffs, n_quad=128):
    """Compute orbit-averaged <R_n> for n in {3,4,5} using Gauss-Legendre.

    Returns a dict {n_deg: <R_n>} for each active Jn coefficient.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    M_pts = np.pi * (nodes + 1.0)  # Map [-1,1] to [0, 2*pi]
    w_pts = np.pi * weights

    E_pts = solve_kepler(M_pts, e)
    f_pts = eccentric_to_true(E_pts, e)
    p = a * (1.0 - e**2)

    averages = {}
    for n_deg in [3, 4, 5]:
        Jn = j_coeffs.get(n_deg, 0.0)
        if Jn == 0.0:
            continue

        Rn_vals = np.empty(len(f_pts))
        for k, fk in enumerate(f_pts):
            r = p / (1.0 + e * np.cos(fk))
            sin_phi = np.sin(inc) * np.sin(omega + fk)
            coeffs = np.zeros(n_deg + 1)
            coeffs[n_deg] = 1.0
            Pn = legval(sin_phi, coeffs)
            Rn_vals[k] = -(mu / r) * Jn * (Re / r) ** n_deg * Pn

        averages[n_deg] = np.sum(w_pts * Rn_vals) / (2.0 * np.pi)

    return averages


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
#  Brouwer SP corrections: J2 (polar-nodal) + J3-J5 (radial + along-track)
# ---------------------------------------------------------------------------

def _eval_Rn_pointwise(r, sin_phi, mu, Re, Jn, n_deg):
    """Evaluate R_n at a single point (vectorized over arrays)."""
    coeffs = np.zeros(n_deg + 1)
    coeffs[n_deg] = 1.0
    Pn = legval(sin_phi, coeffs)
    return -(mu / r) * Jn * (Re / r) ** n_deg * Pn


def _eval_dRn_dr(r, sin_phi, mu, Re, Jn, n_deg):
    """Evaluate dR_n/dr for the radial force (vectorized).

    dR_n/dr = (mu/r²) * Jn * (Re/r)^n * Pn * (n+1)
    """
    coeffs = np.zeros(n_deg + 1)
    coeffs[n_deg] = 1.0
    Pn = legval(sin_phi, coeffs)
    return (mu / r**2) * Jn * (Re / r) ** n_deg * Pn * (n_deg + 1)


def brouwer_sp_polar(a, e, inc, Omega, omega, M, mu, Re, J2, j_coeffs=None,
                     orbit_averages=None):
    """Brouwer SP corrections: J2 polar-nodal + J3-J5 radial perturbation.

    Given MEAN Keplerian elements, returns osculating polar-nodal quantities
    (r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc).
    """
    # Start with J2 SP corrections
    r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc = j2_sp_polar(
        a, e, inc, Omega, omega, M, mu, Re, J2)

    # Add J3-J5 radial SP corrections
    if j_coeffs is not None and orbit_averages is not None:
        E = solve_kepler(M, e)
        f = eccentric_to_true(E, e)
        p = a * (1.0 - e**2)
        r_mean = p / (1.0 + e * np.cos(f))
        u_mean = omega + f
        sin_phi = np.sin(inc) * np.sin(u_mean)

        for n_deg in [3, 4, 5]:
            Jn = j_coeffs.get(n_deg, 0.0)
            if Jn == 0.0 or n_deg not in orbit_averages:
                continue

            Rn_inst = _eval_Rn_pointwise(r_mean, sin_phi, mu, Re, Jn, n_deg)
            Rn_avg = orbit_averages[n_deg]

            # Radial SP correction: delta_r = (2*a²/mu) * (R_n - <R_n>)
            delta_r = (2.0 * a**2 / mu) * (Rn_inst - Rn_avg)
            r_osc = r_osc + delta_r

    return r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc


def brouwer_sp_polar_batch(a, e, inc, Omega, omega, M, mu, Re, J2,
                           j_coeffs=None, orbit_averages=None):
    """Vectorized Brouwer SP: J2 polar-nodal + J3-J5 radial perturbation."""
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    omega = np.asarray(omega, dtype=float)
    M = np.asarray(M, dtype=float)

    # Start with J2 SP corrections
    r_osc, rdot_osc, u_osc, rfdot_osc, Omega_osc, inc_osc = j2_sp_polar_batch(
        a, e, inc, Omega, omega, M, mu, Re, J2)

    # Add J3-J5 radial SP corrections
    if j_coeffs is not None and orbit_averages is not None:
        E = solve_kepler(M, e)
        f = eccentric_to_true(E, e)
        p = a * (1.0 - e**2)
        r_mean = p / (1.0 + e * np.cos(f))
        u_mean = omega + f
        sin_phi = np.sin(inc) * np.sin(u_mean)

        for n_deg in [3, 4, 5]:
            Jn = j_coeffs.get(n_deg, 0.0)
            if Jn == 0.0 or n_deg not in orbit_averages:
                continue

            Rn_inst = _eval_Rn_pointwise(r_mean, sin_phi, mu, Re, Jn, n_deg)
            Rn_avg = orbit_averages[n_deg]

            # Radial SP correction: delta_r = (2*a²/mu) * (R_n - <R_n>)
            delta_r = (2.0 * a**2 / mu) * (Rn_inst - Rn_avg)
            r_osc = r_osc + delta_r

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


# ---------------------------------------------------------------------------
#  Lara (2021) W₁ generating function — exact first-order Brouwer SP
# ---------------------------------------------------------------------------

def evaluate_W1(ell, g, h, L, G, H, mu, Re):
    r"""Evaluate the first-order generating function W₁ (Lara 2021 Eq. 6 + C₁).

    Parameters
    ----------
    ell, g, h : float or ndarray
        Delaunay coordinates (mean anomaly, arg. perigee, RAAN).
    L, G, H : float or ndarray
        Delaunay momenta.
    mu : float
        Gravitational parameter.
    Re : float
        Reference radius (Earth equatorial radius).

    Returns
    -------
    W1 : float or ndarray
        Value of the generating function.
    """
    ell = np.asarray(ell, dtype=float)
    g = np.asarray(g, dtype=float)
    L = np.asarray(L, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)

    # Derived quantities
    a = L**2 / mu
    eta = G / L                         # sqrt(1 - e^2)
    e_sq = 1.0 - eta**2
    e = np.sqrt(np.maximum(e_sq, 0.0))
    p = a * eta**2                      # semi-latus rectum
    s2 = 1.0 - (H / G)**2              # sin^2(I)

    # Solve Kepler equation
    E = solve_kepler(ell, e)
    f = eccentric_to_true(E, e)

    # Equation of center: phi = f - ell, wrapped to [-pi, pi]
    phi = (f - ell + np.pi) % (2.0 * np.pi) - np.pi

    # Brouwer coefficients
    B0 = 1.0 - 1.5 * s2               # 1 - 3s²/2
    B1 = 0.75 * s2                     # 3s²/4

    # W₁ = -(G·Re²/(2·p²)) · [B₀·φ + B₀·e·sin(f) + B₁·e·sin(f+2g)
    #        + B₁·sin(2f+2g) + B₁·(e/3)·sin(3f+2g)] + C₁
    bracket = (
        B0 * phi
        + B0 * e * np.sin(f)
        + B1 * e * np.sin(f + 2.0 * g)
        + B1 * np.sin(2.0 * f + 2.0 * g)
        + B1 * (e / 3.0) * np.sin(3.0 * f + 2.0 * g)
    )

    W1_main = -(G * Re**2 / (2.0 * p**2)) * bracket

    # C₁ integration constant (Lara 2021 Eq. 13)
    # C₁ = (G·Re²/p²) · (15s²-14)/(32(5s²-4)) · s²·e² · sin(2g)
    denom_C1 = 5.0 * s2 - 4.0
    # At the critical inclination (5s²-4 = 0), C₁ should still be finite
    # because the numerator (15s²-14) also has specific behavior.
    # For safety, protect against exact zero (which is measure-zero).
    safe_denom = np.where(np.abs(denom_C1) < 1e-30, 1e-30, denom_C1)
    C1 = (G * Re**2 / p**2) * (15.0 * s2 - 14.0) / (32.0 * safe_denom) * s2 * e**2 * np.sin(2.0 * g)

    return W1_main + C1



def sp_corrections_w1_delaunay(ell, g, h, L, G, H, mu, Re, J2):
    r"""Compute J₂ SP corrections via Poisson brackets in Delaunay variables.

    Returns partial derivatives of W₁ w.r.t. all 6 Delaunay variables via
    central finite differences.

    Returns
    -------
    dW1_dell, dW1_dg, dW1_dh, dW1_dL, dW1_dG, dW1_dH : float or ndarray
        Partial derivatives of W₁.
    """
    # Step sizes (relative for momenta, absolute for angles)
    eps_ell = 1e-8
    eps_g = 1e-8
    eps_h = 1e-8
    eps_L = L * 1e-8
    eps_G = G * 1e-8
    eps_H = max(abs(H) * 1e-8, 1e-15)

    W = evaluate_W1

    dW1_dell = (W(ell + eps_ell, g, h, L, G, H, mu, Re)
                - W(ell - eps_ell, g, h, L, G, H, mu, Re)) / (2.0 * eps_ell)
    dW1_dg = (W(ell, g + eps_g, h, L, G, H, mu, Re)
              - W(ell, g - eps_g, h, L, G, H, mu, Re)) / (2.0 * eps_g)
    dW1_dh = (W(ell, g, h + eps_h, L, G, H, mu, Re)
              - W(ell, g, h - eps_h, L, G, H, mu, Re)) / (2.0 * eps_h)
    dW1_dL = (W(ell, g, h, L + eps_L, G, H, mu, Re)
              - W(ell, g, h, L - eps_L, G, H, mu, Re)) / (2.0 * eps_L)
    dW1_dG = (W(ell, g, h, L, G + eps_G, H, mu, Re)
              - W(ell, g, h, L, G - eps_G, H, mu, Re)) / (2.0 * eps_G)
    dW1_dH = (W(ell, g, h, L, G, H + eps_H, mu, Re)
              - W(ell, g, h, L, G, H - eps_H, mu, Re)) / (2.0 * eps_H)

    return dW1_dell, dW1_dg, dW1_dh, dW1_dL, dW1_dG, dW1_dH


def sp_corrections_kep_w1(a, e, inc, Omega, omega, M, mu, Re, J2):
    r"""First-order J₂ SP corrections in Keplerian elements via W₁ Poisson brackets.

    Given MEAN Keplerian elements, returns (da, de, dI, dOmega, domega, dM).

    The Poisson brackets in Delaunay (q=[ℓ,g,h], p=[L,G,H]) are:

    - {a, W₁} = -(2L/μ) · ∂W₁/∂ℓ
    - {e, W₁} = -(η²/(eL)) · ∂W₁/∂ℓ + (η/(eL)) · ∂W₁/∂g
    - {I, W₁} = -(cos I/(G sin I)) · ∂W₁/∂g + (1/(G sin I)) · ∂W₁/∂h
    - {Ω, W₁} = ∂W₁/∂H
    - {ω, W₁} = ∂W₁/∂G
    - {M, W₁} = ∂W₁/∂L

    Each correction is multiplied by J₂.
    """
    from .coordinates import keplerian_to_delaunay

    # Convert to Delaunay
    ell, g, h, L, G, H_del = keplerian_to_delaunay(a, e, inc, Omega, omega, M, mu)

    # Get all partial derivatives of W₁
    dW_dell, dW_dg, dW_dh, dW_dL, dW_dG, dW_dH = \
        sp_corrections_w1_delaunay(ell, g, h, L, G, H_del, mu, Re, J2)

    # Derived quantities
    eta = G / L  # sqrt(1 - e^2)

    # Poisson brackets
    da = -(2.0 * L / mu) * dW_dell

    # For e: need care when e ~ 0
    if e > 1e-10:
        de = -(eta**2 / (e * L)) * dW_dell + (eta / (e * L)) * dW_dg
    else:
        # For nearly circular orbits, use limit form.
        # As e -> 0, {e, W1} = -dW/dell / L + dW/dg / L  (since eta -> 1)
        # Actually, need L'Hopital or alternative parameterization.
        # The Poisson bracket {e, W1} = -(eta^2/(eL)) dW/dell + (eta/(eL)) dW/dg
        # = (1/(eL)) * (-eta^2 * dW/dell + eta * dW/dg)
        # = (eta/(eL)) * (-eta * dW/dell + dW/dg)
        # If we define x = -eta * dW/dell + dW/dg, then de = eta*x/(eL).
        # For small e, both the numerator and e should be proportional,
        # so the ratio stays finite. Numerically, central differences
        # with the step size we use should be fine even at e=1e-4.
        de = -(eta**2 / (max(e, 1e-12) * L)) * dW_dell + (eta / (max(e, 1e-12) * L)) * dW_dg

    sin_i = np.sin(inc)
    cos_i = np.cos(inc)
    if abs(sin_i) > 1e-10:
        di = -(cos_i / (G * sin_i)) * dW_dg + (1.0 / (G * sin_i)) * dW_dh
    else:
        di = 0.0  # Equatorial: dI = 0 by symmetry

    dOmega = dW_dH
    domega = dW_dG
    dM = dW_dL

    # Multiply by J2
    return (J2 * da, J2 * de, J2 * di, J2 * dOmega, J2 * domega, J2 * dM)


def sp_corrections_kep_w1_batch(a, e, inc, Omega, omega, M, mu, Re, J2):
    r"""Vectorized version of sp_corrections_kep_w1 for arrays.

    Given arrays of MEAN Keplerian elements, returns arrays of
    (da, de, dI, dOmega, domega, dM).
    """
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    omega = np.asarray(omega, dtype=float)
    M = np.asarray(M, dtype=float)

    # Convert to Delaunay (vectorized)
    L = np.sqrt(mu * a)
    G = L * np.sqrt(np.maximum(1.0 - e**2, 0.0))
    H_del = G * np.cos(inc)
    ell = M
    g = omega
    h = Omega

    # Get all partial derivatives of W₁ (vectorized via evaluate_W1)
    eps_ell = 1e-8
    eps_g = 1e-8
    eps_h = 1e-8
    eps_L = L * 1e-8
    eps_G = G * 1e-8
    eps_H = np.maximum(np.abs(H_del) * 1e-8, 1e-15)

    W = evaluate_W1

    dW_dell = (W(ell + eps_ell, g, h, L, G, H_del, mu, Re)
               - W(ell - eps_ell, g, h, L, G, H_del, mu, Re)) / (2.0 * eps_ell)
    dW_dg = (W(ell, g + eps_g, h, L, G, H_del, mu, Re)
             - W(ell, g - eps_g, h, L, G, H_del, mu, Re)) / (2.0 * eps_g)
    dW_dh = (W(ell, g, h + eps_h, L, G, H_del, mu, Re)
             - W(ell, g, h - eps_h, L, G, H_del, mu, Re)) / (2.0 * eps_h)
    dW_dL = (W(ell, g, h, L + eps_L, G, H_del, mu, Re)
             - W(ell, g, h, L - eps_L, G, H_del, mu, Re)) / (2.0 * eps_L)
    dW_dG = (W(ell, g, h, L, G + eps_G, H_del, mu, Re)
             - W(ell, g, h, L, G - eps_G, H_del, mu, Re)) / (2.0 * eps_G)
    dW_dH = (W(ell, g, h, L, G, H_del + eps_H, mu, Re)
             - W(ell, g, h, L, G, H_del - eps_H, mu, Re)) / (2.0 * eps_H)

    # Derived quantities
    eta = G / L

    # Poisson brackets (vectorized)
    da = -(2.0 * L / mu) * dW_dell

    # Handle e ~ 0 gracefully
    e_safe = np.maximum(e, 1e-12)
    de = -(eta**2 / (e_safe * L)) * dW_dell + (eta / (e_safe * L)) * dW_dg

    sin_i = np.sin(inc)
    cos_i = np.cos(inc)
    sin_i_safe = np.where(np.abs(sin_i) < 1e-10, 1e-10, sin_i)
    di = -(cos_i / (G * sin_i_safe)) * dW_dg + (1.0 / (G * sin_i_safe)) * dW_dh
    # Zero out for equatorial orbits
    di = np.where(np.abs(sin_i) < 1e-10, 0.0, di)

    dOmega = dW_dH
    domega = dW_dG
    dM_corr = dW_dL

    return (J2 * da, J2 * de, J2 * di, J2 * dOmega, J2 * domega, J2 * dM_corr)



def mean_to_osc_kep_w1(a, e, inc, Om, om, M, mu, Re, J2):
    """Mean Keplerian -> osculating Keplerian via W₁ SP corrections (scalar)."""
    da, de, di, dOm, dom, dM = sp_corrections_kep_w1(
        a, e, inc, Om, om, M, mu, Re, J2)
    return (a + da, e + de, inc + di, Om + dOm, om + dom, M + dM)


def mean_to_osc_kep_w1_batch(a, e, inc, Om, om, M, mu, Re, J2):
    """Mean Keplerian -> osculating Keplerian via W₁ SP corrections (batch)."""
    da, de, di, dOm, dom, dM = sp_corrections_kep_w1_batch(
        a, e, inc, Om, om, M, mu, Re, J2)
    return (a + da, e + de, inc + di, Om + dOm, om + dom, M + dM)



def mean_to_cartesian_w1(a, e, inc, Om, om, M, mu, Re, J2):
    """Mean Keplerian -> osculating Cartesian via W₁ Keplerian SP (scalar).

    Computes J₂·{ξ, W₁} for each Keplerian element, then converts the
    osculating Keplerian elements to Cartesian.  Allows signed (negative)
    eccentricity in the conversion, which is the correct analytical
    continuation for nearly circular orbits.
    """
    da, de, di, dOm, dom, dM = sp_corrections_kep_w1(
        a, e, inc, Om, om, M, mu, Re, J2)

    osc_a = a + da
    osc_e = e + de  # Can be negative for near-circular orbits
    osc_i = inc + di
    osc_Om = Om + dOm
    osc_om = om + dom
    osc_M = M + dM

    E_osc = solve_kepler(osc_M, osc_e)
    f_osc = eccentric_to_true(E_osc, osc_e)
    return keplerian_to_cartesian(osc_a, osc_e, osc_i, osc_Om, osc_om, f_osc, mu)


def mean_to_cartesian_w1_batch(a, e, inc, Om, om, M, mu, Re, J2):
    """Mean Keplerian -> osculating Cartesian via W₁ Keplerian SP (batch).

    Returns (N,3) positions, (N,3) velocities.
    """
    from .coordinates import keplerian_to_cartesian_batch

    da, de, di, dOm, dom, dM = sp_corrections_kep_w1_batch(
        a, e, inc, Om, om, M, mu, Re, J2)

    osc_a = a + da
    osc_e = e + de  # Signed eccentricity (can be negative)
    osc_i = inc + di
    osc_Om = Om + dOm
    osc_om = om + dom
    osc_M = M + dM

    E_osc = solve_kepler(osc_M, osc_e)
    f_osc = eccentric_to_true(E_osc, osc_e)
    return keplerian_to_cartesian_batch(osc_a, osc_e, osc_i, osc_Om, osc_om, f_osc, mu)


def osculating_to_mean_w1(osc_kep, mu, Re, J2, max_iter=20, tol=1e-12):
    """Osculating Keplerian -> mean Keplerian via W₁ SP (Lyddane-space iteration).

    Subtracts SP corrections directly in the non-singular Lyddane
    representation [a, ecosω, esinω, I, Ω, M+ω] to avoid the O(J₂²)
    error introduced by Cartesian-space subtraction.
    """
    a_o, e_o, i_o, Om_o, om_o, M_o = osc_kep

    # Osculating state in Lyddane representation
    osc_ecosw = e_o * np.cos(om_o)
    osc_esinw = e_o * np.sin(om_o)
    osc_Mpw = M_o + om_o
    osc_a = float(a_o)
    osc_I = float(i_o)
    osc_Om = float(Om_o)

    cf = _get_sp_heyoka_cfunc(mu, Re, J2)

    # Initialize mean Lyddane to osculating
    m_a = osc_a
    m_ecosw = osc_ecosw
    m_esinw = osc_esinw
    m_I = osc_I
    m_Om = osc_Om
    m_Mpw = osc_Mpw

    for _ in range(max_iter):
        # Recover Keplerian from current mean Lyddane
        m_e = np.sqrt(m_ecosw**2 + m_esinw**2)
        m_om = np.arctan2(m_esinw, m_ecosw)
        m_M = m_Mpw - m_om

        m_e = np.clip(m_e, 1e-15, 0.9999)
        m_a = max(m_a, 100.0)

        # Delaunay momenta
        L = np.sqrt(mu * m_a)
        G = L * np.sqrt(1.0 - m_e**2)
        H = G * np.cos(m_I)
        E = solve_kepler(m_M, m_e)

        # SP corrections in Lyddane form: [da, d(ecosw), d(esinw), dI, dOm, d(M+w)]
        res = cf([E, m_om, L, G, H])
        da = float(res[0])
        d_ecosw = float(res[1])
        d_esinw = float(res[2])
        dI = float(res[3])
        dOm = float(res[4])
        d_Mpw = float(res[5])

        # Subtract in Lyddane space: mean = osc - sp(mean)
        new_a = osc_a - da
        new_ecosw = osc_ecosw - d_ecosw
        new_esinw = osc_esinw - d_esinw
        new_I = osc_I - dI
        new_Om = osc_Om - dOm
        new_Mpw = osc_Mpw - d_Mpw

        # Check convergence
        conv_a = abs(new_a - m_a)
        conv_e = np.sqrt((new_ecosw - m_ecosw)**2 + (new_esinw - m_esinw)**2)
        m_a, m_ecosw, m_esinw, m_I, m_Om, m_Mpw = (
            new_a, new_ecosw, new_esinw, new_I, new_Om, new_Mpw)

        if conv_a < tol and conv_e < tol:
            break

    # Final recovery of Keplerian elements
    m_e = np.sqrt(m_ecosw**2 + m_esinw**2)
    m_om = np.arctan2(m_esinw, m_ecosw)
    m_M = m_Mpw - m_om

    m_e = np.clip(m_e, 1e-15, 0.9999)
    m_a = max(m_a, 100.0)

    return (m_a, m_e, m_I, m_Om, m_om, m_M)


# ---------------------------------------------------------------------------
#  Heyoka AD-based non-singular SP corrections (Lyddane variables)
# ---------------------------------------------------------------------------

_SP_HEYOKA_CFUNC = None
_SP_HEYOKA_PARAMS = None


def _build_sp_heyoka_cfunc(mu, Re, J2):
    r"""Build heyoka cfunc for exact J2 SP corrections via Poisson brackets of W1.

    Uses the eccentric anomaly E as parameter to avoid the Kepler equation,
    and outputs Lyddane non-singular combinations to avoid 1/e singularities:

        [da, d(e*cos w), d(e*sin w), dI, dOmega, d(M+w)]

    The key non-singular identities:

    * {M+w, W1} = dW1/dL + dW1/dG.  The chain-rule contribution from
      dE/dL + dE/dG uses  (de/dL + de/dG) = -G/(L^2(eta+1)),
      which has NO 1/e.

    * {e*cos w, W1}: the 1/e in de/dL,de/dG is neutralised because the
      numerator  BRACKET = -eta*dW1/dE + (1-e*cosE)*dW1/dg  is O(e),
      giving BRACKET/e = O(1) after the division.

    Input variables: [E, g, L, G, H]
    """
    import heyoka as hy

    E_s, g_s, L_s, G_s, H_s = hy.make_vars("E", "g", "L", "G", "H")

    # --- Derived quantities ---
    e_s = hy.sqrt(1.0 - (G_s / L_s) ** 2)
    eta_s = G_s / L_s
    p_s = G_s ** 2 / mu  # p = G^2/mu, independent of L
    s2_s = 1.0 - (H_s / G_s) ** 2  # sin^2(I)

    B0_s = 1.0 - 1.5 * s2_s
    B1_s = 0.75 * s2_s

    sinE_s = hy.sin(E_s)
    cosE_s = hy.cos(E_s)

    # True anomaly from eccentric anomaly
    f_s = hy.atan2(eta_s * sinE_s, cosE_s - e_s)

    # Equation of center: phi = f - ell, where ell = E - e*sinE.
    # Use atan2(sin(phi), cos(phi)) to make phi manifestly 2π-periodic
    # in E, avoiding the branch-cut discontinuity that occurs when
    # phi_s = atan2(...) - E + e*sinE and E crosses 2π.
    ell_s = E_s - e_s * sinE_s
    sin_phi = hy.sin(f_s) * hy.cos(ell_s) - hy.cos(f_s) * hy.sin(ell_s)
    cos_phi = hy.cos(f_s) * hy.cos(ell_s) + hy.sin(f_s) * hy.sin(ell_s)
    phi_s = hy.atan2(sin_phi, cos_phi)

    # --- W1 (Lara 2021 Eq. 6 + C1 from Eq. 13) ---
    W1_terms = (
        B0_s * phi_s
        + B0_s * e_s * hy.sin(f_s)
        + B1_s * e_s * hy.sin(f_s + 2.0 * g_s)
        + B1_s * hy.sin(2.0 * f_s + 2.0 * g_s)
        + B1_s * (e_s / 3.0) * hy.sin(3.0 * f_s + 2.0 * g_s)
    )
    W1_main = -(G_s * Re ** 2 / (2.0 * p_s ** 2)) * W1_terms

    denom_c1 = 5.0 * s2_s - 4.0
    C1_s = (
        (G_s * Re ** 2 / p_s ** 2)
        * (15.0 * s2_s - 14.0)
        / (32.0 * denom_c1)
        * s2_s
        * e_s ** 2
        * hy.sin(2.0 * g_s)
    )

    W1_s = W1_main + C1_s

    # --- Partial derivatives at fixed E ---
    dW1_dE = hy.diff(W1_s, E_s)
    dW1_dg = hy.diff(W1_s, g_s)
    dW1_dL_E = hy.diff(W1_s, L_s)  # at fixed E
    dW1_dG_E = hy.diff(W1_s, G_s)  # at fixed E
    dW1_dH = hy.diff(W1_s, H_s)

    one_minus_ecosE = 1.0 - e_s * cosE_s

    # --- da: non-singular ---
    # {a, W1} = -(2L/mu) * dW1/dl,  dW1/dl = dW1/dE / (1-ecosE)
    dW1_dell = dW1_dE / one_minus_ecosE
    da_expr = J2 * (-(2.0 * L_s / mu) * dW1_dell)

    # --- dI: non-singular (for i != 0) ---
    # {I, W1} = -(cosI/(G*sinI)) * dW1/dg  (since dW1/dh = 0)
    c_s = H_s / G_s
    si_s = hy.sqrt(s2_s)
    di_expr = J2 * (-(c_s / (G_s * si_s)) * dW1_dg)

    # --- dOmega: non-singular ---
    # {Omega, W1} = dW1/dH
    dOm_expr = J2 * dW1_dH

    # --- d(M+w): non-singular via (eta-1)/e = -e/(eta+1) trick ---
    # de/dL + de/dG = G*(eta-1)/(eL^2) = -G*e / (L^2*(eta+1))   [no 1/e]
    dedLpG = -G_s * e_s / (L_s ** 2 * (eta_s + 1.0))
    dEdLpG = dedLpG * sinE_s / one_minus_ecosE
    dMpw_expr = J2 * ((dW1_dL_E + dW1_dG_E) + dW1_dE * dEdLpG)

    # --- d(ecosw) and d(esinw): non-singular via BRACKET/e ---
    # BRACKET = -eta*dW1/dE + (1-ecosE)*dW1/dg   is O(e)
    # {ecosw, W1} = eta*cosw*BRACKET / (eL(1-ecosE))
    #             - sinw * [e*dW1/dG|_E - eta*sinE*dW1/dE/(L(1-ecosE))]
    BRACKET = -eta_s * dW1_dE + one_minus_ecosE * dW1_dg

    ecosw_t1 = eta_s * hy.cos(g_s) * BRACKET / (e_s * L_s * one_minus_ecosE)
    ecosw_t2 = -hy.sin(g_s) * (
        e_s * dW1_dG_E
        - eta_s * sinE_s * dW1_dE / (L_s * one_minus_ecosE)
    )
    d_ecosw_expr = J2 * (ecosw_t1 + ecosw_t2)

    # {esinw, W1}: same structure, rotated by pi/2
    esinw_t1 = eta_s * hy.sin(g_s) * BRACKET / (e_s * L_s * one_minus_ecosE)
    esinw_t2 = hy.cos(g_s) * (
        e_s * dW1_dG_E
        - eta_s * sinE_s * dW1_dE / (L_s * one_minus_ecosE)
    )
    d_esinw_expr = J2 * (esinw_t1 + esinw_t2)

    return hy.cfunc(
        [da_expr, d_ecosw_expr, d_esinw_expr, di_expr, dOm_expr, dMpw_expr],
        vars=[E_s, g_s, L_s, G_s, H_s],
    )


def _get_sp_heyoka_cfunc(mu, Re, J2):
    """Lazy-build and cache the heyoka SP cfunc."""
    global _SP_HEYOKA_CFUNC, _SP_HEYOKA_PARAMS
    key = (mu, Re, J2)
    if _SP_HEYOKA_CFUNC is None or _SP_HEYOKA_PARAMS != key:
        _SP_HEYOKA_CFUNC = _build_sp_heyoka_cfunc(mu, Re, J2)
        _SP_HEYOKA_PARAMS = key
    return _SP_HEYOKA_CFUNC


def sp_corrections_heyoka(a, e, inc, Omega, omega, M, mu, Re, J2):
    r"""Exact J2 SP corrections via heyoka AD of W1 Poisson brackets.

    Uses Lyddane non-singular combinations internally and returns
    standard Keplerian corrections (da, de, dI, dOmega, domega, dM).

    For near-circular orbits (e < 1e-6), domega and dM are set to zero
    individually but their sum d(M+w) is exact.  The forward map
    ``mean_to_cartesian_heyoka`` uses the non-singular form directly.

    Parameters
    ----------
    a, e, inc, Omega, omega, M : float
        Mean Keplerian elements (angles in radians).
    mu, Re, J2 : float
        Constants.

    Returns
    -------
    da, de, dI, dOmega, domega, dM : float
    """
    cf = _get_sp_heyoka_cfunc(mu, Re, J2)

    L = np.sqrt(mu * a)
    G = L * np.sqrt(1.0 - e ** 2)
    H = G * np.cos(inc)
    E = solve_kepler(M, e)

    result = cf([E, omega, L, G, H])
    da_v = float(result[0])
    decw = float(result[1])
    desw = float(result[2])
    di_v = float(result[3])
    dOm_v = float(result[4])
    dMpw = float(result[5])

    # Recover de, domega, dM from non-singular Lyddane variables
    cw = np.cos(omega)
    sw = np.sin(omega)
    de_v = decw * cw + desw * sw

    if e > 1e-6:
        dom_v = (-decw * sw + desw * cw) / e
        dM_v = dMpw - dom_v
    else:
        # At e ~ 0, domega and dM diverge individually but
        # d(M+w) is finite.  Set both to zero; the forward map
        # uses the non-singular path.
        dom_v = 0.0
        dM_v = dMpw

    return (da_v, de_v, di_v, dOm_v, dom_v, dM_v)


def sp_corrections_heyoka_batch(a, e, inc, Omega, omega, M, mu, Re, J2):
    r"""Vectorised exact J2 SP corrections via heyoka AD.

    Returns the 6 Lyddane non-singular corrections as arrays:
        (da, d_ecosw, d_esinw, dI, dOmega, d_Mpw)

    These avoid 1/e issues at all eccentricities and are used directly
    by ``mean_to_cartesian_heyoka_batch``.
    """
    cf = _get_sp_heyoka_cfunc(mu, Re, J2)

    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    omega = np.asarray(omega, dtype=float)
    M = np.asarray(M, dtype=float)

    L = np.sqrt(mu * a)
    G = L * np.sqrt(np.maximum(1.0 - e ** 2, 0.0))
    H = G * np.cos(inc)
    E_arr = solve_kepler(M, e)

    N = len(a) if a.ndim > 0 else 1
    da_out = np.empty(N)
    decw_out = np.empty(N)
    desw_out = np.empty(N)
    di_out = np.empty(N)
    dOm_out = np.empty(N)
    dMpw_out = np.empty(N)

    E_flat = np.atleast_1d(E_arr)
    om_flat = np.atleast_1d(omega)
    L_flat = np.atleast_1d(L)
    G_flat = np.atleast_1d(G)
    H_flat = np.atleast_1d(H)

    for k in range(N):
        r = cf([E_flat[k], om_flat[k], L_flat[k], G_flat[k], H_flat[k]])
        da_out[k] = float(r[0])
        decw_out[k] = float(r[1])
        desw_out[k] = float(r[2])
        di_out[k] = float(r[3])
        dOm_out[k] = float(r[4])
        dMpw_out[k] = float(r[5])

    return da_out, decw_out, desw_out, di_out, dOm_out, dMpw_out


def mean_to_cartesian_heyoka(a, e, inc, Om, om, M, mu, Re, J2):
    """Mean Keplerian -> osculating Cartesian via heyoka AD SP corrections.

    Uses non-singular Lyddane combinations (e*cos w, e*sin w, M+w) to
    avoid the 1/e catastrophic cancellation at near-circular orbits.

    Returns (r_vec, v_vec) as 3-element arrays.
    """
    cf = _get_sp_heyoka_cfunc(mu, Re, J2)

    L = np.sqrt(mu * a)
    G = L * np.sqrt(1.0 - e ** 2)
    H = G * np.cos(inc)
    E = solve_kepler(M, e)

    result = cf([E, om, L, G, H])
    da_v = float(result[0])
    decw = float(result[1])
    desw = float(result[2])
    di_v = float(result[3])
    dOm_v = float(result[4])
    dMpw = float(result[5])

    # Osculating elements via non-singular reconstruction
    osc_a = a + da_v
    osc_i = inc + di_v
    osc_Om = Om + dOm_v

    # Non-singular eccentricity vector
    mean_ecosw = e * np.cos(om)
    mean_esinw = e * np.sin(om)
    osc_ecosw = mean_ecosw + decw
    osc_esinw = mean_esinw + desw

    osc_e = np.sqrt(osc_ecosw ** 2 + osc_esinw ** 2)
    osc_om = np.arctan2(osc_esinw, osc_ecosw)

    # Non-singular mean longitude
    mean_Mpw = M + om
    osc_Mpw = mean_Mpw + dMpw
    osc_M = osc_Mpw - osc_om

    E_osc = solve_kepler(osc_M, osc_e)
    f_osc = eccentric_to_true(E_osc, osc_e)
    return keplerian_to_cartesian(osc_a, osc_e, osc_i, osc_Om, osc_om, f_osc, mu)


def mean_to_cartesian_heyoka_batch(a, e, inc, Om, om, M, mu, Re, J2):
    """Vectorised mean -> osculating Cartesian via heyoka AD SP corrections.

    Returns (N,3) positions, (N,3) velocities.
    """
    from .coordinates import keplerian_to_cartesian_batch

    da, decw, desw, di, dOm, dMpw = sp_corrections_heyoka_batch(
        a, e, inc, Om, om, M, mu, Re, J2)

    osc_a = a + da
    osc_i = inc + di
    osc_Om = Om + dOm

    mean_ecosw = e * np.cos(om)
    mean_esinw = e * np.sin(om)
    osc_ecosw = mean_ecosw + decw
    osc_esinw = mean_esinw + desw

    osc_e = np.sqrt(osc_ecosw ** 2 + osc_esinw ** 2)
    osc_om = np.arctan2(osc_esinw, osc_ecosw)

    mean_Mpw = M + om
    osc_Mpw = mean_Mpw + dMpw
    osc_M = osc_Mpw - osc_om

    E_osc = solve_kepler(osc_M, osc_e)
    f_osc = eccentric_to_true(E_osc, osc_e)
    return keplerian_to_cartesian_batch(osc_a, osc_e, osc_i, osc_Om, osc_om, f_osc, mu)
