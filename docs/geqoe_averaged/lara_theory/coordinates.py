"""Coordinate conversions for the Lara-Brouwer analytical theory.

Keplerian <-> Cartesian <-> Delaunay, Kepler equation solver.
All core functions accept scalar or ndarray inputs.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
#  Kepler equation
# ---------------------------------------------------------------------------

def solve_kepler(M, e, tol=1e-15, max_iter=50):
    """Solve M = E - e sin(E) via Newton-Raphson.

    Parameters
    ----------
    M : float or ndarray — mean anomaly [rad]
    e : float or ndarray — eccentricity
    """
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    scalar = M.ndim == 0 and e.ndim == 0
    M = np.atleast_1d(M)
    e = np.atleast_1d(e)

    # Smart starter
    E = np.where(e < 0.8, M.copy(), np.full_like(M, np.pi))

    for _ in range(max_iter):
        sinE = np.sin(E)
        cosE = np.cos(E)
        f_val = E - e * sinE - M
        f_prime = 1.0 - e * cosE
        dE = f_val / f_prime
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break

    return float(E.item()) if scalar else E


def eccentric_to_true(E, e):
    """Eccentric anomaly -> true anomaly (atan2-based)."""
    E = np.asarray(E, dtype=float)
    e = np.asarray(e, dtype=float)
    scalar = E.ndim == 0 and e.ndim == 0
    E = np.atleast_1d(E)
    e = np.atleast_1d(e)

    sinE = np.sin(E)
    cosE = np.cos(E)
    f = np.arctan2(np.sqrt(1.0 - e**2) * sinE, cosE - e)

    return float(f.item()) if scalar else f


def true_to_eccentric(f, e):
    """True anomaly -> eccentric anomaly."""
    f = np.asarray(f, dtype=float)
    e = np.asarray(e, dtype=float)
    scalar = f.ndim == 0 and e.ndim == 0
    f = np.atleast_1d(f)
    e = np.atleast_1d(e)

    sinf = np.sin(f)
    cosf = np.cos(f)
    E = np.arctan2(np.sqrt(np.maximum(1.0 - e**2, 0.0)) * sinf, e + cosf)

    return float(E.item()) if scalar else E


# ---------------------------------------------------------------------------
#  Cartesian <-> Keplerian
# ---------------------------------------------------------------------------

def cartesian_to_keplerian(r_vec, v_vec, mu):
    """Cartesian state -> Keplerian elements.

    Returns (a, e, i, Omega, omega, M)  — angles in radians.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    ecc = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = 0.5 * v**2 - mu / r
    a = -mu / (2.0 * energy)

    # Inclination
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # RAAN
    if n > 1e-15:
        Omega = np.arctan2(n_vec[1], n_vec[0]) % (2.0 * np.pi)
    else:
        Omega = 0.0

    # Argument of perigee
    if n > 1e-15 and ecc > 1e-15:
        cos_omega = np.dot(n_vec, e_vec) / (n * ecc)
        omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
        if e_vec[2] < 0.0:
            omega = 2.0 * np.pi - omega
    elif ecc > 1e-15:
        omega = np.arctan2(e_vec[1], e_vec[0]) % (2.0 * np.pi)
    else:
        omega = 0.0

    # True anomaly
    if ecc > 1e-15:
        cos_f = np.dot(e_vec, r_vec) / (ecc * r)
        f = np.arccos(np.clip(cos_f, -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0.0:
            f = 2.0 * np.pi - f
    elif n > 1e-15:
        cos_u = np.dot(n_vec, r_vec) / (n * r)
        u = np.arccos(np.clip(cos_u, -1.0, 1.0))
        if r_vec[2] < 0.0:
            u = 2.0 * np.pi - u
        f = (u - omega) % (2.0 * np.pi)
    else:
        f = np.arctan2(r_vec[1], r_vec[0]) % (2.0 * np.pi)

    # Mean anomaly
    E = true_to_eccentric(f, ecc)
    M = (E - ecc * np.sin(E)) % (2.0 * np.pi)

    return (a, ecc, inc, Omega, omega, M)


def keplerian_to_cartesian(a, e, i, Omega, omega, f, mu):
    """Keplerian elements -> Cartesian state.  Takes true anomaly f."""
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(f))

    # Perifocal frame
    r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0])
    sqrt_mu_p = np.sqrt(mu / p)
    v_pf = np.array([-sqrt_mu_p * np.sin(f), sqrt_mu_p * (e + np.cos(f)), 0.0])

    # Rotation: perifocal -> inertial
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])

    return R @ r_pf, R @ v_pf


def keplerian_to_cartesian_batch(a, e, inc, Om, om, f, mu):
    """Vectorized keplerian -> cartesian for arrays of elements.

    All inputs are 1-D arrays of length N.  Returns (N,3) positions and velocities.
    """
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    Om = np.asarray(Om, dtype=float)
    om = np.asarray(om, dtype=float)
    f = np.asarray(f, dtype=float)

    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(f))

    cf = np.cos(f)
    sf = np.sin(f)
    sqrt_mu_p = np.sqrt(mu / p)

    # Perifocal
    rx_pf = r_mag * cf
    ry_pf = r_mag * sf
    vx_pf = -sqrt_mu_p * sf
    vy_pf = sqrt_mu_p * (e + cf)

    # Rotation matrix elements
    cO, sO = np.cos(Om), np.sin(Om)
    cw, sw = np.cos(om), np.sin(om)
    ci, si = np.cos(inc), np.sin(inc)

    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R31 = sw * si
    R32 = cw * si

    pos = np.column_stack([
        R11 * rx_pf + R12 * ry_pf,
        R21 * rx_pf + R22 * ry_pf,
        R31 * rx_pf + R32 * ry_pf,
    ])
    vel = np.column_stack([
        R11 * vx_pf + R12 * vy_pf,
        R21 * vx_pf + R22 * vy_pf,
        R31 * vx_pf + R32 * vy_pf,
    ])
    return pos, vel


# ---------------------------------------------------------------------------
#  Keplerian <-> Delaunay
# ---------------------------------------------------------------------------

def keplerian_to_delaunay(a, e, i, Omega, omega, M, mu):
    """Returns (ell, g, h, L, G, H)."""
    L = np.sqrt(mu * a)
    G = L * np.sqrt(1.0 - e**2)
    H = G * np.cos(i)
    return (M, omega, Omega, L, G, H)


def delaunay_to_keplerian(ell, g, h, L, G, H, mu):
    """Returns (a, e, i, Omega, omega, M)."""
    a = L**2 / mu
    e_sq = 1.0 - (G / L) ** 2
    e = np.sqrt(np.maximum(e_sq, 0.0))
    cos_i = np.clip(H / G, -1.0, 1.0)
    i = np.arccos(cos_i)
    return (a, e, i, h, g, ell)


# ---------------------------------------------------------------------------
#  Polar-nodal quantities from Keplerian
# ---------------------------------------------------------------------------

def keplerian_to_polar(a, e, i, Omega, omega, M, mu):
    """Compute polar-nodal quantities from Keplerian elements.

    Returns (r, rdot, u, rfdot, Omega, inc) where u = omega + f.
    """
    E = solve_kepler(M, e)
    f = eccentric_to_true(E, e)

    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(f))
    u = omega + f

    # Radial and tangential velocities
    sqrt_mu_p = np.sqrt(mu / p)
    rdot = sqrt_mu_p * e * np.sin(f)           # dr/dt = sqrt(mu/p) * e sin f
    rfdot = np.sqrt(mu * p) / r                # r df/dt = h/r

    return r, rdot, u, rfdot, Omega, i


def polar_to_cartesian(r, rdot, u, rfdot, Omega, inc):
    """Convert polar-nodal (r, rdot, u, rfdot, Omega, inc) to Cartesian.

    Works for scalars or arrays (vectorised).
    """
    r = np.asarray(r, dtype=float)
    rdot = np.asarray(rdot, dtype=float)
    u = np.asarray(u, dtype=float)
    rfdot = np.asarray(rfdot, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    inc = np.asarray(inc, dtype=float)

    su, cu = np.sin(u), np.cos(u)
    sO, cO = np.sin(Omega), np.cos(Omega)
    si, ci = np.sin(inc), np.cos(inc)

    # Unit vectors U (radial) and V (tangential) in the orbital plane
    Ux = -sO * ci * su + cO * cu
    Uy = cO * ci * su + sO * cu
    Uz = si * su

    Vx = -sO * ci * cu - cO * su
    Vy = cO * ci * cu - sO * su
    Vz = si * cu

    if r.ndim == 0:
        pos = np.array([r * Ux, r * Uy, r * Uz])
        vel = np.array([rdot * Ux + rfdot * Vx,
                        rdot * Uy + rfdot * Vy,
                        rdot * Uz + rfdot * Vz])
    else:
        pos = np.column_stack([r * Ux, r * Uy, r * Uz])
        vel = np.column_stack([rdot * Ux + rfdot * Vx,
                               rdot * Uy + rfdot * Vy,
                               rdot * Uz + rfdot * Vz])
    return pos, vel
