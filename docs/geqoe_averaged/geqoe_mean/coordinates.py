"""Coordinate conversion utilities for GEqOE averaged theory.

Consolidates kepler_to_rv, rotation matrices, and rv_to_classical
which were duplicated across 6+ scripts.
"""

from __future__ import annotations

import numpy as np


def rot3(theta: float) -> np.ndarray:
    """Rotation matrix about the 3rd (z) axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot1(theta: float) -> np.ndarray:
    """Rotation matrix about the 1st (x) axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def kepler_to_rv(
    a_km: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    M_deg: float,
    mu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to Cartesian position and velocity vectors.

    Parameters
    ----------
    a_km : semi-major axis [km]
    e : eccentricity
    inc_deg : inclination [deg]
    raan_deg : RAAN [deg]
    argp_deg : argument of perigee [deg]
    M_deg : mean anomaly [deg]
    mu : gravitational parameter [km^3/s^2], defaults to Earth
    """
    if mu is None:
        from astrodyn_core.geqoe_taylor import MU
        mu = MU

    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    M = np.deg2rad(M_deg)

    # Newton-Raphson for Kepler's equation
    E = M if e < 0.8 else np.pi
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1.0e-14:
            break

    cE = np.cos(E)
    sE = np.sin(E)
    r_pf = np.array([a_km * (cE - e), a_km * np.sqrt(1.0 - e * e) * sE, 0.0])
    rm = a_km * (1.0 - e * cE)
    v_pf = np.sqrt(mu * a_km) / rm * np.array(
        [-sE, np.sqrt(1.0 - e * e) * cE, 0.0]
    )
    dcm = rot3(raan) @ rot1(inc) @ rot3(argp)
    return dcm @ r_pf, dcm @ v_pf


def rv_to_classical(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    mu: float | None = None,
) -> tuple[float, float, float]:
    """Cartesian state to classical Keplerian (a, e, i_deg)."""
    if mu is None:
        from astrodyn_core.geqoe_taylor import MU
        mu = MU

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    E_kep = 0.5 * v * v - mu / r
    a = -mu / (2.0 * E_kep)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    inc = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1, 1)))
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    ecc = np.linalg.norm(e_vec)
    return a, ecc, inc
