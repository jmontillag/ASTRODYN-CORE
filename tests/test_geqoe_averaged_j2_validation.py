"""Regression tests for the averaged J2 GEqOE formulation study.

These checks are intentionally narrow:
1. lock the corrected first-order Psi identity used by the short-period map;
2. verify that the full inverse map improves the secular-vs-mean match across
   representative initial anomalies.
"""

from __future__ import annotations

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor import J2, MU, RE, J2Perturbation, build_state_integrator, cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.integrator import propagate_grid
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen


def _rot3(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rot1(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _kepler_to_rv(
    a_km: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    mean_anomaly_deg: float,
    mu: float = MU,
) -> tuple[np.ndarray, np.ndarray]:
    i = np.deg2rad(i_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    M = np.deg2rad(mean_anomaly_deg)

    E = M if e < 0.8 else np.pi
    for _ in range(50):
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= delta
        if abs(delta) < 1.0e-14:
            break

    cosE = np.cos(E)
    sinE = np.sin(E)
    r_pf = np.array([a_km * (cosE - e), a_km * np.sqrt(1.0 - e * e) * sinE, 0.0], dtype=float)
    r_mag = a_km * (1.0 - e * cosE)
    v_pf = np.sqrt(mu * a_km) / r_mag * np.array(
        [-sinE, np.sqrt(1.0 - e * e) * cosE, 0.0],
        dtype=float,
    )

    dcm = _rot3(raan) @ _rot1(i) @ _rot3(argp)
    return dcm @ r_pf, dcm @ v_pf


def _secular_rates(state0: np.ndarray, j2: float = J2) -> tuple[float, float]:
    nu0 = state0[0]
    g0 = np.hypot(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    beta = np.sqrt(1.0 - g0 * g0)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    c_i = delta / gamma
    p = a * beta * beta
    omega_node = -1.5 * nu0 * j2 * (RE / p) ** 2 * c_i
    omega_peri = 0.75 * nu0 * j2 * (RE / p) ** 2 * (5.0 * c_i * c_i - 2.0 * c_i - 1.0)
    return omega_peri, omega_node


def _secular_solution(state0: np.ndarray, t_grid_s: np.ndarray, j2: float = J2) -> dict[str, np.ndarray]:
    _, p10, p20, _, q10, q20 = state0
    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    psi0 = np.arctan2(p10, p20)
    omega0 = np.arctan2(q10, q20)
    omega_peri, omega_node = _secular_rates(state0, j2)
    psi = psi0 + omega_peri * t_grid_s
    omega = omega0 + omega_node * t_grid_s
    return {
        "p1": g0 * np.sin(psi),
        "p2": g0 * np.cos(psi),
        "q1": Q0 * np.sin(omega),
        "q2": Q0 * np.cos(omega),
        "psi": psi,
        "Omega": omega,
    }


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _periodic_zero_mean_integral(prime: np.ndarray, x: np.ndarray) -> np.ndarray:
    raw = _cumulative_trapezoid(prime, x)
    return raw - np.trapezoid(raw, x) / (x[-1] - x[0])


def _interp_periodic(x: np.ndarray, y: np.ndarray, x_eval: float) -> float:
    period = x[-1] - x[0]
    wrapped = (x_eval - x[0]) % period + x[0]
    return float(np.interp(wrapped, x, y))


def _orbit_means(series: np.ndarray, n_orbits: int, samples_per_orbit: int) -> np.ndarray:
    trimmed = series[: n_orbits * samples_per_orbit]
    return trimmed.reshape(n_orbits, samples_per_orbit).mean(axis=1)


def _build_mean_state(state0: np.ndarray, corr: dict[str, float]) -> np.ndarray:
    g0 = np.hypot(state0[1], state0[2])
    psi0 = np.arctan2(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    Omega0 = np.arctan2(state0[4], state0[5])
    mean_state = state0.copy()
    g_mean = g0 - corr["g_eval"]
    psi_mean = psi0 - corr["Psi_eval"]
    Q_mean = Q0 - corr["Q_eval"]
    Omega_mean = Omega0 - corr["Omega_eval"]
    mean_state[1] = g_mean * np.sin(psi_mean)
    mean_state[2] = g_mean * np.cos(psi_mean)
    mean_state[4] = Q_mean * np.sin(Omega_mean)
    mean_state[5] = Q_mean * np.cos(Omega_mean)
    return mean_state


def _short_period_map(state0: np.ndarray, g_eval: float, j2: float = J2) -> dict[str, float]:
    nu0, p10, p20, _, q10, q20 = state0
    g0 = float(np.hypot(p10, p20))
    Q0 = float(np.hypot(q10, q20))
    psi0 = float(np.arctan2(p10, p20))
    omega0 = float(np.arctan2(q10, q20))

    beta = np.sqrt(1.0 - g0 * g0)
    alpha = 1.0 / (1.0 + beta)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    w = np.sqrt(MU / a)
    c = (MU * MU / nu0) ** (1.0 / 3.0) * beta
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    s_i = 2.0 * Q0 / gamma
    c_i = delta / gamma
    p = a * beta * beta
    A = MU * j2 * RE * RE / 2.0
    omega_peri, omega_node = _secular_rates(state0, j2)

    G = np.linspace(0.0, 2.0 * np.pi, 4097, dtype=float)
    cosG = np.cos(G)
    sinG = np.sin(G)
    sinK = np.sin(G + psi0)
    cosK = np.cos(G + psi0)
    X = a * (alpha * p10 * p20 * sinK + (1.0 - alpha * p10 * p10) * cosK - p20)
    Y = a * (alpha * p10 * p20 * cosK + (1.0 - alpha * p20 * p20) * sinK - p10)
    r = a * (1.0 - g0 * cosG)
    cosf = (cosG - g0) / (1.0 - g0 * cosG)
    sinf = beta * sinG / (1.0 - g0 * cosG)
    argp = psi0 - omega0
    sin_u = np.sin(argp) * cosf + np.cos(argp) * sinf
    cos_u = np.cos(argp) * cosf - np.sin(argp) * sinf

    U = -A / (r**3) * (1.0 - 3.0 * s_i * s_i * sin_u * sin_u)
    d = -U / c
    w_h = 3.0 * A * delta * (s_i * s_i) * sin_u * sin_u / (c * r**3)
    xi1 = X / a + 2.0 * p20
    xi2 = Y / a + 2.0 * p10
    p1_dot = p20 * (d - w_h) - (1.0 / c) * xi1 * U
    p2_dot = p10 * (w_h - d) + (1.0 / c) * xi2 * U
    psi_dot = (p20 * p1_dot - p10 * p2_dot) / (g0 * g0)

    dg_dG = (r / w) * (U / c) * beta * sinG
    dQ_dG = -6.0 * A * Q0 * c_i / (MU * beta * r**2) * sin_u * cos_u
    dOmega_dG = -6.0 * A * c_i / (MU * beta * r**2) * sin_u * sin_u
    dPsi_dG = (r / w) * psi_dot

    N_prime = dOmega_dG - (r / w) * omega_node
    P_prime = dPsi_dG - (r / w) * omega_peri
    g_corr = _periodic_zero_mean_integral(dg_dG, G)
    Q_corr = _periodic_zero_mean_integral(dQ_dG, G)
    Omega_corr = _periodic_zero_mean_integral(N_prime, G)
    Psi_corr = _periodic_zero_mean_integral(P_prime, G)

    return {
        "g_eval": _interp_periodic(G, g_corr, g_eval),
        "Q_eval": _interp_periodic(G, Q_corr, g_eval),
        "Omega_eval": _interp_periodic(G, Omega_corr, g_eval),
        "Psi_eval": _interp_periodic(G, Psi_corr, g_eval),
    }


def _mean_longitude_rate(state0: np.ndarray, j2: float = J2) -> float:
    nu0 = state0[0]
    g0 = np.hypot(state0[1], state0[2])
    Q0 = np.hypot(state0[4], state0[5])
    beta = np.sqrt(1.0 - g0 * g0)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    c_i = delta / gamma
    p = a * beta * beta
    return nu0 + 0.75 * nu0 * j2 * (RE / p) ** 2 * (
        beta * (3.0 * c_i * c_i - 1.0) + 5.0 * c_i * c_i - 2.0 * c_i - 1.0
    )


def _mean_longitude_map(state0: np.ndarray, j2: float = J2) -> tuple[np.ndarray, np.ndarray, float]:
    nu0, p10, p20, _, q10, q20 = state0
    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    beta = np.sqrt(1.0 - g0 * g0)
    alpha = 1.0 / (1.0 + beta)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    w = np.sqrt(MU / a)
    c = (MU * MU / nu0) ** (1.0 / 3.0) * beta
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    A = MU * j2 * RE * RE / 2.0

    K_grid = np.linspace(0.0, 2.0 * np.pi, 4097, dtype=float)
    sinK = np.sin(K_grid)
    cosK = np.cos(K_grid)
    X = a * (alpha * p10 * p20 * sinK + (1.0 - alpha * p10 * p10) * cosK - p20)
    Y = a * (alpha * p10 * p20 * cosK + (1.0 - alpha * p20 * p20) * sinK - p10)
    r = a * (1.0 - p10 * sinK - p20 * cosK)
    zhat = 2.0 * (Y * q20 - X * q10) / (gamma * r)

    U = -A / (r**3) * (1.0 - 3.0 * zhat * zhat)
    h = np.sqrt(c * c - 2.0 * r * r * U)
    d = (h - c) / (r * r)
    I_val = 3.0 * A * zhat * delta / (h * r**3)
    w_h = I_val * zhat

    p1_dot = p20 * (d - w_h) - (1.0 / c) * (X / a + 2.0 * p20) * U
    p2_dot = p10 * (w_h - d) + (1.0 / c) * (Y / a + 2.0 * p10) * U
    K_dot = w / r + d - w_h - (1.0 / c) * (1.0 + alpha * (1.0 - r / a)) * U
    L_dot = (r / a) * K_dot + p1_dot * cosK - p2_dot * sinK

    L_dot_avg = np.trapezoid(r * L_dot, K_grid) / (2.0 * np.pi * a)
    ell_prime = (r / w) * (L_dot - L_dot_avg)
    ell_corr = _periodic_zero_mean_integral(ell_prime, K_grid)
    return K_grid, ell_corr, float(L_dot_avg)


def _component_rms(reference: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> float:
    vals = []
    for key in ("p1", "p2", "q1", "q2"):
        vals.append(float(np.sqrt(np.mean((reference[key] - model[key]) ** 2))))
    return float(np.mean(vals))


def _make_molniya_state(mean_anomaly_deg: float) -> np.ndarray:
    rp_km = 6916.0
    e = 0.74
    a_km = rp_km / (1.0 - e)
    r0, v0 = _kepler_to_rv(a_km, e, 63.4, 30.0, 270.0, mean_anomaly_deg)
    return cart2geqoe(r0, v0, MU, J2Perturbation())


def test_short_period_psi_identity_matches_direct_geqoe_flow() -> None:
    state0 = _make_molniya_state(45.0)
    nu0, p10, p20, _, q10, q20 = state0
    g0 = np.hypot(p10, p20)
    Q0 = np.hypot(q10, q20)
    psi0 = np.arctan2(p10, p20)
    omega0 = np.arctan2(q10, q20)
    beta = np.sqrt(1.0 - g0 * g0)
    alpha = 1.0 / (1.0 + beta)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    w = np.sqrt(MU / a)
    c = (MU * MU / nu0) ** (1.0 / 3.0) * beta
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    s_i = 2.0 * Q0 / gamma
    A = MU * J2 * RE * RE / 2.0

    G = np.linspace(0.0, 2.0 * np.pi, 2049, dtype=float)
    cosG = np.cos(G)
    sinG = np.sin(G)
    sinK = np.sin(G + psi0)
    cosK = np.cos(G + psi0)
    X = a * (alpha * p10 * p20 * sinK + (1.0 - alpha * p10 * p10) * cosK - p20)
    Y = a * (alpha * p10 * p20 * cosK + (1.0 - alpha * p20 * p20) * sinK - p10)
    r = a * (1.0 - g0 * cosG)
    cosf = (cosG - g0) / (1.0 - g0 * cosG)
    sinf = beta * sinG / (1.0 - g0 * cosG)
    argp = psi0 - omega0
    sin_u = np.sin(argp) * cosf + np.cos(argp) * sinf

    U = -A / (r**3) * (1.0 - 3.0 * s_i * s_i * sin_u * sin_u)
    d = -U / c
    w_h = 3.0 * A * delta * (s_i * s_i) * sin_u * sin_u / (c * r**3)
    xi1 = X / a + 2.0 * p20
    xi2 = Y / a + 2.0 * p10
    p1_dot = p20 * (d - w_h) - (1.0 / c) * xi1 * U
    p2_dot = p10 * (w_h - d) + (1.0 / c) * xi2 * U
    psi_dot_exact = (p20 * p1_dot - p10 * p2_dot) / (g0 * g0)
    psi_dot_closed = d - w_h - (U / c) * (1.0 + cosG / g0)
    dpsi_dg_exact = (r / w) * psi_dot_exact
    dpsi_dg_closed = (r / w) * psi_dot_closed

    assert np.max(np.abs(dpsi_dg_exact - dpsi_dg_closed)) < 1.0e-12


@pytest.mark.parametrize("mean_anomaly_deg", [0.0, 45.0, 90.0])
def test_full_inverse_map_improves_component_means(mean_anomaly_deg: float) -> None:
    state0 = _make_molniya_state(mean_anomaly_deg)
    orbit_period_s = 2.0 * np.pi / state0[0]
    n_orbits = 20
    samples_per_orbit = 120
    t_grid = np.linspace(0.0, n_orbits * orbit_period_s, n_orbits * samples_per_orbit + 1, dtype=float)

    ta, _ = build_state_integrator(J2Perturbation(), state0, tol=1.0e-15, compact_mode=True)
    states = propagate_grid(ta, t_grid)

    centers = 0.5 * (
        t_grid[: n_orbits * samples_per_orbit : samples_per_orbit]
        + t_grid[samples_per_orbit : n_orbits * samples_per_orbit + samples_per_orbit : samples_per_orbit]
    )
    orbit_means = {
        "p1": _orbit_means(states[:, 1], n_orbits, samples_per_orbit),
        "p2": _orbit_means(states[:, 2], n_orbits, samples_per_orbit),
        "q1": _orbit_means(states[:, 4], n_orbits, samples_per_orbit),
        "q2": _orbit_means(states[:, 5], n_orbits, samples_per_orbit),
    }

    sec_naive = _secular_solution(state0, centers)
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    corr = _short_period_map(state0, G0)
    sec_full = _secular_solution(_build_mean_state(state0, corr), centers)

    naive_rms = _component_rms(orbit_means, sec_naive)
    full_rms = _component_rms(orbit_means, sec_full)

    assert full_rms < naive_rms * 0.2
    assert full_rms < 1.0e-5


def test_full_inverse_map_improves_phase_fit_for_nonsymmetric_case() -> None:
    state0 = _make_molniya_state(45.0)
    orbit_period_s = 2.0 * np.pi / state0[0]
    n_orbits = 20
    samples_per_orbit = 120
    t_grid = np.linspace(0.0, n_orbits * orbit_period_s, n_orbits * samples_per_orbit + 1, dtype=float)

    ta, _ = build_state_integrator(J2Perturbation(), state0, tol=1.0e-15, compact_mode=True)
    states = propagate_grid(ta, t_grid)

    centers = 0.5 * (
        t_grid[: n_orbits * samples_per_orbit : samples_per_orbit]
        + t_grid[samples_per_orbit : n_orbits * samples_per_orbit + samples_per_orbit : samples_per_orbit]
    )
    orbit_mean_psi = _orbit_means(np.unwrap(np.arctan2(states[:, 1], states[:, 2])), n_orbits, samples_per_orbit)
    orbit_mean_omega = _orbit_means(np.unwrap(np.arctan2(states[:, 4], states[:, 5])), n_orbits, samples_per_orbit)

    sec_naive = _secular_solution(state0, centers)
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    corr = _short_period_map(state0, G0)
    sec_full = _secular_solution(_build_mean_state(state0, corr), centers)

    psi_rms_naive = float(np.sqrt(np.mean((orbit_mean_psi - sec_naive["psi"]) ** 2)))
    psi_rms_full = float(np.sqrt(np.mean((orbit_mean_psi - sec_full["psi"]) ** 2)))
    omega_rms_naive = float(np.sqrt(np.mean((orbit_mean_omega - sec_naive["Omega"]) ** 2)))
    omega_rms_full = float(np.sqrt(np.mean((orbit_mean_omega - sec_full["Omega"]) ** 2)))

    assert psi_rms_full < psi_rms_naive * 0.05
    assert omega_rms_full < omega_rms_naive * 0.1


def test_slow_inverse_map_plus_true_k_recovers_cartesian_state() -> None:
    state0 = _make_molniya_state(45.0)
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    mean0 = _build_mean_state(state0, _short_period_map(state0, G0))

    orbit_period_s = 2.0 * np.pi / state0[0]
    t_grid = np.linspace(0.0, 10.0 * orbit_period_s, 401, dtype=float)
    ta, _ = build_state_integrator(J2Perturbation(), state0, tol=1.0e-15, compact_mode=True)
    states = propagate_grid(ta, t_grid)

    pos_err = []
    for ti, sosc in zip(t_grid[::20], states[::20]):
        sec = _secular_solution(mean0, np.array([ti]))
        ms = mean0.copy()
        ms[1] = sec["p1"][0]
        ms[2] = sec["p2"][0]
        ms[4] = sec["q1"][0]
        ms[5] = sec["q2"][0]

        G_eval = sosc[3] - np.arctan2(sosc[1], sosc[2])
        corr = _short_period_map(ms, G_eval)
        rec = ms.copy()
        g = np.hypot(ms[1], ms[2]) + corr["g_eval"]
        psi = np.arctan2(ms[1], ms[2]) + corr["Psi_eval"]
        Q = np.hypot(ms[4], ms[5]) + corr["Q_eval"]
        Omega = np.arctan2(ms[4], ms[5]) + corr["Omega_eval"]
        rec[1] = g * np.sin(psi)
        rec[2] = g * np.cos(psi)
        rec[4] = Q * np.sin(Omega)
        rec[5] = Q * np.cos(Omega)
        rec[3] = sosc[3]

        r_rec, _ = geqoe2cart(rec, MU, J2Perturbation())
        r_osc, _ = geqoe2cart(sosc, MU, J2Perturbation())
        pos_err.append(np.linalg.norm(r_rec - r_osc))

    assert float(np.mean(pos_err)) < 0.2
    assert float(np.max(pos_err)) < 0.5


def test_exact_frozen_state_mean_longitude_map_recovers_fast_phase() -> None:
    state0 = _make_molniya_state(45.0)
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    mean0 = _build_mean_state(state0, _short_period_map(state0, G0))

    K_grid0, ell_corr0, _ = _mean_longitude_map(mean0)
    L0_bar_exact = K_to_L(state0[3], state0[1], state0[2]) - _interp_periodic(K_grid0, ell_corr0, state0[3])
    L0_bar_classical = K_to_L(state0[3], mean0[1], mean0[2])

    orbit_period_s = 2.0 * np.pi / state0[0]
    t_grid = np.linspace(0.0, 10.0 * orbit_period_s, 401, dtype=float)
    ta, _ = build_state_integrator(J2Perturbation(), state0, tol=1.0e-15, compact_mode=True)
    states = propagate_grid(ta, t_grid)

    pos_err_classical = []
    pos_err_exact = []
    for ti, sosc in zip(t_grid[::20], states[::20]):
        sec = _secular_solution(mean0, np.array([ti]))
        ms = mean0.copy()
        ms[1] = sec["p1"][0]
        ms[2] = sec["p2"][0]
        ms[4] = sec["q1"][0]
        ms[5] = sec["q2"][0]

        Lbar_classical = L0_bar_classical + _mean_longitude_rate(mean0) * ti
        Kbar_classical = solve_kepler_gen(Lbar_classical, ms[1], ms[2])
        Gbar_classical = Kbar_classical - np.arctan2(ms[1], ms[2])
        corr_classical = _short_period_map(ms, Gbar_classical)
        g_classical = np.hypot(ms[1], ms[2]) + corr_classical["g_eval"]
        psi_classical = np.arctan2(ms[1], ms[2]) + corr_classical["Psi_eval"]
        Q_classical = np.hypot(ms[4], ms[5]) + corr_classical["Q_eval"]
        Omega_classical = np.arctan2(ms[4], ms[5]) + corr_classical["Omega_eval"]
        rec_classical = np.array(
            [
                ms[0],
                g_classical * np.sin(psi_classical),
                g_classical * np.cos(psi_classical),
                Kbar_classical,
                Q_classical * np.sin(Omega_classical),
                Q_classical * np.cos(Omega_classical),
            ],
            dtype=float,
        )

        K_grid, ell_corr, L_dot_avg = _mean_longitude_map(ms)
        Lbar_exact = L0_bar_exact + L_dot_avg * ti
        Kbar_exact = solve_kepler_gen(Lbar_exact, ms[1], ms[2])
        Gbar_exact = Kbar_exact - np.arctan2(ms[1], ms[2])
        corr_exact = _short_period_map(ms, Gbar_exact)
        g_exact = np.hypot(ms[1], ms[2]) + corr_exact["g_eval"]
        psi_exact = np.arctan2(ms[1], ms[2]) + corr_exact["Psi_eval"]
        Q_exact = np.hypot(ms[4], ms[5]) + corr_exact["Q_eval"]
        Omega_exact = np.arctan2(ms[4], ms[5]) + corr_exact["Omega_eval"]
        Losc_exact = Lbar_exact + _interp_periodic(K_grid, ell_corr, Kbar_exact)
        Kosc_exact = solve_kepler_gen(Losc_exact, g_exact * np.sin(psi_exact), g_exact * np.cos(psi_exact))
        rec_exact = np.array(
            [
                ms[0],
                g_exact * np.sin(psi_exact),
                g_exact * np.cos(psi_exact),
                Kosc_exact,
                Q_exact * np.sin(Omega_exact),
                Q_exact * np.cos(Omega_exact),
            ],
            dtype=float,
        )

        r_classical, _ = geqoe2cart(rec_classical, MU, J2Perturbation())
        r_exact, _ = geqoe2cart(rec_exact, MU, J2Perturbation())
        r_osc, _ = geqoe2cart(sosc, MU, J2Perturbation())
        pos_err_classical.append(np.linalg.norm(r_classical - r_osc))
        pos_err_exact.append(np.linalg.norm(r_exact - r_osc))

    assert float(np.mean(pos_err_exact)) < 0.1
    assert float(np.max(pos_err_exact)) < 0.2
    assert float(np.mean(pos_err_exact)) < float(np.mean(pos_err_classical)) * 2.0e-3
