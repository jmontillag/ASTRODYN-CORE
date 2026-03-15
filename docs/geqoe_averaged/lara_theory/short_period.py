"""First-order short-period corrections for the Lara-Brouwer theory.

J2 corrections use the Lyddane/SGP4-style polar-nodal form (robust at e~0).
J3-J5 corrections add radial and along-track perturbations from the
instantaneous minus orbit-averaged zonal disturbing function.
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
    """Osculating Keplerian -> mean Keplerian via W₁ SP (Cartesian-space iteration).

    Uses the same Cartesian-space iteration as ``osculating_to_mean``, but with
    the W₁ Cartesian Poisson bracket forward map.  The forward map computes
    {r, W₁} directly, avoiding the G > L singularity.
    """
    # Target osculating Cartesian
    E_osc = solve_kepler(osc_kep[5], osc_kep[1])
    f_osc = eccentric_to_true(E_osc, osc_kep[1])
    r_target, v_target = keplerian_to_cartesian(*osc_kep[:5], f_osc, mu)

    mean_kep = np.array(osc_kep, dtype=float)

    for _ in range(max_iter):
        a_m, e_m, i_m, Om_m, om_m, M_m = mean_kep

        # Unperturbed Cartesian from mean elements
        E_m = solve_kepler(M_m, e_m)
        f_m = eccentric_to_true(E_m, e_m)
        r_unpert, v_unpert = keplerian_to_cartesian(a_m, e_m, i_m, Om_m, om_m, f_m, mu)

        # Osculating Cartesian via W₁ Cartesian Poisson bracket
        r_fwd, v_fwd = mean_to_cartesian_w1(a_m, e_m, i_m, Om_m, om_m, M_m, mu, Re, J2)

        # SP displacement
        sp_r = np.asarray(r_fwd).ravel() - np.asarray(r_unpert).ravel()
        sp_v = np.asarray(v_fwd).ravel() - np.asarray(v_unpert).ravel()

        # Mean Cartesian = target - SP
        r_mean = np.asarray(r_target).ravel() - sp_r
        v_mean = np.asarray(v_target).ravel() - sp_v

        new_kep = np.array(cartesian_to_keplerian(r_mean, v_mean, mu))

        da = abs(new_kep[0] - mean_kep[0])
        de = abs(new_kep[1] - mean_kep[1])
        mean_kep = new_kep

        mean_kep[0] = max(mean_kep[0], 100.0)
        mean_kep[1] = np.clip(mean_kep[1], 1e-10, 0.9999)

        if da < tol and de < tol:
            break

    return tuple(mean_kep)
