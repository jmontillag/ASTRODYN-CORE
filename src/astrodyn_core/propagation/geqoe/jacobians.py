"""Analytical Jacobians for Cartesian <-> GEqOE transformations.

Vectorized implementations that account for J2 perturbations.
"""

from typing import Tuple, Union

import numpy as np

from astrodyn_core.propagation.geqoe.conversion import BodyConstants
from astrodyn_core.propagation.geqoe.utils import solve_kep_gen


def get_pEqpY(
    t: float,
    y: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
) -> np.ndarray:
    """Compute the Jacobian d(Eq)/d(Y) from Cartesian to GEqOE.

    Args:
        t: Time in seconds (unused, retained for interface parity).
        y: Cartesian state(s) ``[rx, ry, rz, vx, vy, vz]`` in SI units.
        p: Body constants as ``BodyConstants`` or ``(J2, Re, mu)``.

    Returns:
        Jacobian tensor ``d(Eq)/d(Y)`` with shape ``(N, 6, 6)``.
    """
    if isinstance(p, BodyConstants):
        J2, Re, mu = p.j2, p.re, p.mu
    else:
        J2, Re, mu = p

    y = np.atleast_2d(y)
    L = Re
    T = (Re**3 / mu) ** 0.5
    mu_norm = 1.0
    A = J2 / 2.0

    rv = y[:, :3] / L
    rpv = y[:, 3:6] / L * T

    r2 = np.sum(rv**2, axis=1)
    r = np.sqrt(r2)
    rp = np.sum(rv * rpv, axis=1) / r
    v2 = np.sum(rpv**2, axis=1)
    r3 = r**3

    zg = rv[:, 2] / r
    U = -A / r3 * (1 - 3 * zg**2)

    er = rv / r[:, np.newaxis]
    ez = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])

    hv = np.cross(rv, rpv)
    h2 = np.sum(hv**2, axis=1)
    h = np.sqrt(h2)
    Ueff = h2 / (2 * r2) + U

    eh = hv / h[:, np.newaxis]
    ef = np.cross(eh, er)

    E_val = v2 / 2 - mu_norm / r + U
    nu_norm = (1 / mu_norm) * (-2 * E_val) ** 1.5

    ehez = np.dot(eh, ez)
    q1 = np.dot(eh, ex) / (1 + ehez)
    q2 = -np.dot(eh, ey) / (1 + ehez)

    q1s, q2s, q1q2 = q1**2, q2**2, q1 * q2
    eTerm = 1 / (1 + q1s + q2s)
    eX = eTerm[:, np.newaxis] * np.vstack([1 - q1s + q2s, 2 * q1q2, -2 * q1]).T
    eY = eTerm[:, np.newaxis] * np.vstack([2 * q1q2, 1 + q1s - q2s, 2 * q2]).T

    c = np.sqrt(2 * r2 * Ueff)
    gv = rp[:, np.newaxis] * er + c[:, np.newaxis] / r[:, np.newaxis] * ef
    g = (np.cross(gv, np.cross(rv, gv)) - mu_norm * er) / mu_norm

    p1 = np.sum(g * eY, axis=1)
    p2 = np.sum(g * eX, axis=1)
    p1s, p2s = p1**2, p2**2
    gs = p1s + p2s

    X = np.sum(rv * eX, axis=1)
    Y_val = np.sum(rv * eY, axis=1)

    beta = np.sqrt(1 - gs)
    alpha = 1 / (1 + beta)
    a = (mu_norm / nu_norm**2) ** (1 / 3)

    Xp = np.sum(rpv * eX, axis=1)
    Yp = np.sum(rpv * eY, axis=1)

    cosL = X / r
    sinL = Y_val / r
    rou = a * (1 - gs)

    # Derivative terms
    Lambda = Y_val * q2 - X * q1
    LAMBDA_p = Yp * q2 - Xp * q1
    epsi = p2 + cosL
    zeta = p1 + sinL
    gamma_ = 1 + q1s + q2s

    pU_r_term1 = -6 * zg[:, np.newaxis] * (ez - zg[:, np.newaxis] * er)
    pU_r_term2 = -3 * (1 - 3 * zg**2)[:, np.newaxis] * er
    pU_r = -A / r[:, np.newaxis] ** 4 * (pU_r_term1 + pU_r_term2)

    delta_0 = -3 / np.sqrt(mu_norm * a)[:, np.newaxis] * pU_r
    delta_1_scalar = r / (rou * mu_norm) * (r * zeta + rou * sinL)
    delta_1 = delta_1_scalar[:, np.newaxis] * pU_r
    delta_2_scalar = r / (rou * mu_norm) * (r * epsi + rou * cosL)
    delta_2 = delta_2_scalar[:, np.newaxis] * pU_r
    delta_3_scalar = r * rp / (c * mu_norm) * (rou + r) * alpha
    delta_3 = delta_3_scalar[:, np.newaxis] * pU_r

    # Position derivatives
    pnu_r = -3 * a * nu_norm / r2
    pnu_r = pnu_r[:, np.newaxis] * er + delta_0

    pp1_r_t1 = (zeta / r)[:, np.newaxis] * er
    pp1_r_t2 = (h / (c * r) * ((2 - c / h) * p2 + X / a))[:, np.newaxis] * ef
    pp1_r_t3 = (p2 * LAMBDA_p / h)[:, np.newaxis] * eh
    pp1_r = pp1_r_t1 - pp1_r_t2 - pp1_r_t3 + delta_1

    pp2_r_t1 = (epsi / r)[:, np.newaxis] * er
    pp2_r_t2 = (h / (c * r) * ((2 - c / h) * p1 + Y_val / a))[:, np.newaxis] * ef
    pp2_r_t3 = (p1 * LAMBDA_p / h)[:, np.newaxis] * eh
    pp2_r = pp2_r_t1 + pp2_r_t2 + pp2_r_t3 + delta_2

    pLr_r_t1 = (rp / (c * r) * (rou * alpha - r * beta))[:, np.newaxis] * er
    pLr_r_t2 = (h / (c * r) * (2 - c / h + alpha / a * (rou - r)))[:, np.newaxis] * ef
    pLr_r_t3 = (LAMBDA_p / h)[:, np.newaxis] * eh
    pLr_r = pLr_r_t1 - pLr_r_t2 - pLr_r_t3 + delta_3

    pq1_r = (-gamma_ * Yp / (2 * h))[:, np.newaxis] * eh
    pq2_r = (-gamma_ * Xp / (2 * h))[:, np.newaxis] * eh

    # Velocity derivatives
    pnu_rp = (-3 / np.sqrt(mu_norm * a))[:, np.newaxis] * rpv

    pp1_rp_t1 = (-c / mu_norm * cosL)[:, np.newaxis] * er
    pp1_rp_t2 = (h / mu_norm * (2 * sinL - rp / c * X))[:, np.newaxis] * ef
    pp1_rp_t3 = (Lambda * p2 / h)[:, np.newaxis] * eh
    pp1_rp = pp1_rp_t1 + pp1_rp_t2 + pp1_rp_t3

    pp2_rp_t1 = (-c / mu_norm * sinL)[:, np.newaxis] * er
    pp2_rp_t2 = (h / mu_norm * (2 * cosL + rp / c * Y_val))[:, np.newaxis] * ef
    pp2_rp_t3 = (Lambda * p1 / h)[:, np.newaxis] * eh
    pp2_rp = -pp2_rp_t1 + pp2_rp_t2 - pp2_rp_t3

    pLr_rp_t1 = (
        (c / (mu_norm * r) * alpha * (r - rou)) - 2 * r / np.sqrt(mu_norm * a)
    )[:, np.newaxis] * er
    pLr_rp_t2 = (h * rp / (c * mu_norm) * alpha * (rou + r))[:, np.newaxis] * ef
    pLr_rp_t3 = (Lambda / h)[:, np.newaxis] * eh
    pLr_rp = pLr_rp_t1 + pLr_rp_t2 + pLr_rp_t3

    pq1_rp = (gamma_ * Y_val / (2 * h))[:, np.newaxis] * eh
    pq2_rp = (gamma_ * X / (2 * h))[:, np.newaxis] * eh

    # Assemble Jacobian
    N = y.shape[0]
    pEqpY = np.zeros((N, 6, 6))

    pEqpY[:, 0, :3] = pnu_r / (L * T)
    pEqpY[:, 0, 3:] = pnu_rp / L
    pEqpY[:, 1, :3] = pq1_r / L
    pEqpY[:, 1, 3:] = pq1_rp * T / L
    pEqpY[:, 2, :3] = pq2_r / L
    pEqpY[:, 2, 3:] = pq2_rp * T / L
    pEqpY[:, 3, :3] = pp1_r / L
    pEqpY[:, 3, 3:] = pp1_rp * T / L
    pEqpY[:, 4, :3] = pp2_r / L
    pEqpY[:, 4, 3:] = pp2_rp * T / L
    pEqpY[:, 5, :3] = pLr_r / L
    pEqpY[:, 5, 3:] = pLr_rp * T / L

    return pEqpY


def get_pYpEq(
    t: float,
    y: np.ndarray,
    p: Union[BodyConstants, Tuple[float, float, float]],
) -> np.ndarray:
    """Compute the Jacobian d(Y)/d(Eq) from GEqOE to Cartesian.

    Args:
        t: Time in seconds (unused, retained for interface parity).
        y: GEqOE state(s) ``[nu, q1, q2, p1, p2, Lr]``.
        p: Body constants as ``BodyConstants`` or ``(J2, Re, mu)``.

    Returns:
        Jacobian tensor ``d(Y)/d(Eq)`` with shape ``(N, 6, 6)``.
    """
    if isinstance(p, BodyConstants):
        J2, Re, mu = p.j2, p.re, p.mu
    else:
        J2, Re, mu = p

    y = np.atleast_2d(y)
    L = Re
    T = (Re**3 / mu) ** 0.5
    mu_norm = 1.0
    A = J2 / 2.0

    nu_norm, q1, q2, p1, p2, Lr_in = (
        y[:, 0] * T,
        y[:, 1],
        y[:, 2],
        y[:, 3],
        y[:, 4],
        y[:, 5],
    )
    Lr = np.mod(Lr_in, 2 * np.pi)

    K = solve_kep_gen(Lr, p1, p2)
    sinK, cosK = np.sin(K), np.cos(K)

    q1s, q2s, q1q2 = q1**2, q2**2, q1 * q2
    gamma_ = 1 + q1s + q2s
    eTerm = 1 / gamma_
    eX = eTerm[:, np.newaxis] * np.vstack([1 - q1s + q2s, 2 * q1q2, -2 * q1]).T
    eY = eTerm[:, np.newaxis] * np.vstack([2 * q1q2, 1 + q1s - q2s, 2 * q2]).T

    p1s, p2s = p1**2, p2**2
    gs = p1s + p2s
    beta = np.sqrt(1 - gs)
    alpha = 1 / (1 + beta)
    a = (mu_norm / nu_norm**2) ** (1 / 3)

    X = a * (alpha * p1 * p2 * sinK + (1 - alpha * p1s) * cosK - p2)
    Y_val = a * (alpha * p1 * p2 * cosK + (1 - alpha * p2s) * sinK - p1)

    rv_norm = X[:, np.newaxis] * eX + Y_val[:, np.newaxis] * eY
    r = np.sqrt(np.sum(rv_norm**2, axis=1))
    rp = np.sqrt(mu_norm * a) / r * (p2 * sinK - p1 * cosK)

    cosL, sinL = X / r, Y_val / r
    c = (mu_norm**2 / nu_norm) ** (1 / 3) * beta
    rou = a * (1 - gs)

    z = rv_norm[:, 2]
    zg = z / r
    U = -A / r**3 * (1 - 3 * zg**2)
    h = np.sqrt(c**2 - 2 * r**2 * U)

    Xp = rp * cosL - h / r * sinL
    Yp = rp * sinL + h / r * cosL
    rpv_norm = Xp[:, np.newaxis] * eX + Yp[:, np.newaxis] * eY

    er = rv_norm / r[:, np.newaxis]
    hv = np.cross(rv_norm, rpv_norm)
    eh = hv / np.linalg.norm(hv, axis=1)[:, np.newaxis]
    ef = np.cross(eh, er)

    # Derivative terms
    v = rp[:, np.newaxis] * er + c[:, np.newaxis] / r[:, np.newaxis] * ef
    epsi, zeta = p2 + cosL, p1 + sinL

    # Position derivatives
    pr_nu = (-2 * r / (3 * nu_norm))[:, np.newaxis] * er
    pr_p1_s1 = -(alpha * rp / nu_norm * p2 + a * sinL)
    pr_p1_s2 = -a * ((a * alpha * beta / r + r / rou) * p2 + X / rou + cosL)
    pr_p1 = pr_p1_s1[:, np.newaxis] * er + pr_p1_s2[:, np.newaxis] * ef
    pr_p2_s1 = alpha * rp / nu_norm * p1 - a * cosL
    pr_p2_s2 = a * ((a * alpha * beta / r + r / rou) * p1 + Y_val / rou + sinL)
    pr_p2 = pr_p2_s1[:, np.newaxis] * er + pr_p2_s2[:, np.newaxis] * ef
    pr_Lr = (1 / nu_norm)[:, np.newaxis] * v
    pr_q1 = (-2 / gamma_)[:, np.newaxis] * (
        r[:, np.newaxis] * q2[:, np.newaxis] * ef + X[:, np.newaxis] * eh
    )
    pr_q2 = (2 / gamma_)[:, np.newaxis] * (
        r[:, np.newaxis] * q1[:, np.newaxis] * ef + Y_val[:, np.newaxis] * eh
    )

    # Perturbation potential derivatives
    ez = np.array([0.0, 0.0, 1.0])
    pU_r = (3 * A / r**4)[:, np.newaxis] * (
        2 * zg[:, np.newaxis] * ez + (1 - 5 * zg**2)[:, np.newaxis] * er
    )
    pU_nu = np.sum(pU_r * pr_nu, axis=1)
    pU_p1 = np.sum(pU_r * pr_p1, axis=1)
    pU_p2 = np.sum(pU_r * pr_p2, axis=1)
    pU_Lr = np.sum(pU_r * pr_Lr, axis=1)
    pU_q1 = np.sum(pU_r * pr_q1, axis=1)
    pU_q2 = np.sum(pU_r * pr_q2, axis=1)

    # Delta vectors
    delta_0 = (r / h * (2 / (3 * nu_norm) * U - pU_nu))[:, np.newaxis] * ef
    t1 = (a * (a * alpha * beta / r * p2 + cosL))[:, np.newaxis] * er + (
        alpha * rp / nu_norm * p2
    )[:, np.newaxis] * ef
    delta_1_s1 = ((h - c) / r**2)[:, np.newaxis] * t1
    delta_1_s2 = (
        1 / h * (2 * (alpha * rp / nu_norm * p2 + a * sinL) * U - r * pU_p1)
    )[:, np.newaxis] * ef
    delta_1 = delta_1_s1 + delta_1_s2
    t2 = (a * (a * alpha * beta / r * p1 + sinL))[:, np.newaxis] * er + (
        alpha * rp / nu_norm * p1
    )[:, np.newaxis] * ef
    delta_2_s1 = ((c - h) / r**2)[:, np.newaxis] * t2
    delta_2_s2 = (
        -1 / h * (2 * (alpha * rp / nu_norm * p1 - a * cosL) * U + r * pU_p2)
    )[:, np.newaxis] * ef
    delta_2 = delta_2_s1 + delta_2_s2
    t3 = (1 / nu_norm)[:, np.newaxis] * (
        (c / r)[:, np.newaxis] * er + rp[:, np.newaxis] * ef
    )
    delta_3 = ((c - h) / r**2)[:, np.newaxis] * t3 - (
        1 / h * (2 * rp / nu_norm * U + r * pU_Lr)
    )[:, np.newaxis] * ef
    delta_4 = (-r / h * pU_q1)[:, np.newaxis] * ef
    delta_5 = (-r / h * pU_q2)[:, np.newaxis] * ef

    # Velocity derivatives
    s = (h / r)[:, np.newaxis] * er - rp[:, np.newaxis] * ef
    prp_nu = (1 / (3 * nu_norm))[:, np.newaxis] * rpv_norm + delta_0
    prp_p1 = (
        (a / r * alpha * np.sqrt(mu_norm * a) / r * p2)[:, np.newaxis] * er
        - (a / r * (mu_norm / h * p1 + Xp))[:, np.newaxis] * ef
        + (a / rou * epsi)[:, np.newaxis] * s
        + delta_1
    )
    prp_p2 = (
        (-a / r * alpha * np.sqrt(mu_norm * a) / r * p1)[:, np.newaxis] * er
        - (a / r * (mu_norm / h * p2 - Yp))[:, np.newaxis] * ef
        - (a / rou * zeta)[:, np.newaxis] * s
        + delta_2
    )
    prp_Lr = (-mu_norm / (r**2 * nu_norm))[:, np.newaxis] * er + delta_3
    prp_q1 = (
        (2 / gamma_)[:, np.newaxis]
        * (q2[:, np.newaxis] * s - Xp[:, np.newaxis] * eh)
        + delta_4
    )
    prp_q2 = (
        (-2 / gamma_)[:, np.newaxis]
        * (q1[:, np.newaxis] * s - Yp[:, np.newaxis] * eh)
        + delta_5
    )

    # Assemble Jacobian
    N = y.shape[0]
    pYpEq = np.zeros((N, 6, 6))

    pYpEq[:, :3, 0] = pr_nu * L * T
    pYpEq[:, 3:, 0] = prp_nu * L

    pYpEq[:, :3, 1] = pr_q1 * L
    pYpEq[:, 3:, 1] = prp_q1 * L / T

    pYpEq[:, :3, 2] = pr_q2 * L
    pYpEq[:, 3:, 2] = prp_q2 * L / T

    pYpEq[:, :3, 3] = pr_p1 * L
    pYpEq[:, 3:, 3] = prp_p1 * L / T

    pYpEq[:, :3, 4] = pr_p2 * L
    pYpEq[:, 3:, 4] = prp_p2 * L / T

    pYpEq[:, :3, 5] = pr_Lr * L
    pYpEq[:, 3:, 5] = prp_Lr * L / T

    return pYpEq


__all__ = ["get_pEqpY", "get_pYpEq"]
