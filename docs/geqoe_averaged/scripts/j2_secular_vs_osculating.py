#!/usr/bin/env python
"""Compare the closed secular J2 GEqOE model against the full J2 integration.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/j2_secular_vs_osculating.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOC_DIR = Path(__file__).resolve().parents[1]
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from astrodyn_core.geqoe_taylor import (
    J2,
    MU,
    RE,
    J2Perturbation,
    build_state_integrator,
    cart2geqoe,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

from geqoe_mean.coordinates import kepler_to_rv as _kepler_to_rv

FIG_DIR = DOC_DIR / "figures"


def _secular_solution(state0: np.ndarray, t_grid_s: np.ndarray) -> dict[str, np.ndarray]:
    nu0, p10, p20, _, q10, q20 = state0
    g0 = float(np.hypot(p10, p20))
    Q0 = float(np.hypot(q10, q20))
    psi0 = float(np.arctan2(p10, p20))
    omega0 = float(np.arctan2(q10, q20))

    beta = np.sqrt(1.0 - g0 * g0)
    alpha = 1.0 / (1.0 + beta)
    a = (MU / (nu0 * nu0)) ** (1.0 / 3.0)
    gamma = 1.0 + Q0 * Q0
    delta = 1.0 - Q0 * Q0
    cos_i = delta / gamma
    p = a * beta * beta

    omega_node = -1.5 * nu0 * J2 * (RE / p) ** 2 * cos_i
    omega_peri = 0.75 * nu0 * J2 * (RE / p) ** 2 * (
        5.0 * cos_i * cos_i - 2.0 * cos_i - 1.0
    )

    psi = psi0 + omega_peri * t_grid_s
    omega = omega0 + omega_node * t_grid_s

    return {
        "nu": np.full_like(t_grid_s, nu0),
        "p1": g0 * np.sin(psi),
        "p2": g0 * np.cos(psi),
        "q1": Q0 * np.sin(omega),
        "q2": Q0 * np.cos(omega),
        "g": np.full_like(t_grid_s, g0),
        "Q": np.full_like(t_grid_s, Q0),
        "psi": psi,
        "Omega": omega,
        "omega_node": np.full_like(t_grid_s, omega_node),
        "omega_peri": np.full_like(t_grid_s, omega_peri),
    }


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def _periodic_zero_mean_integral(prime: np.ndarray, x: np.ndarray) -> np.ndarray:
    raw = _cumulative_trapezoid(prime, x)
    mean_value = np.trapezoid(raw, x) / (x[-1] - x[0])
    return raw - mean_value


def _interp_periodic(x: np.ndarray, y: np.ndarray, x_eval: float) -> float:
    period = x[-1] - x[0]
    wrapped = (x_eval - x[0]) % period + x[0]
    return float(np.interp(wrapped, x, y))


def _first_order_j2_short_period_map(state0: np.ndarray, g_eval: float) -> dict[str, np.ndarray | float]:
    nu0, p10, p20, K0, q10, q20 = state0
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
    A = MU * J2 * RE * RE / 2.0

    omega_node = -1.5 * nu0 * J2 * (RE / p) ** 2 * c_i
    omega_peri = 0.75 * nu0 * J2 * (RE / p) ** 2 * (
        5.0 * c_i * c_i - 2.0 * c_i - 1.0
    )

    G = np.linspace(0.0, 2.0 * np.pi, 4097, dtype=float)
    cosG = np.cos(G)
    sinG = np.sin(G)
    p1 = p10
    p2 = p20
    sinK = np.sin(G + psi0)
    cosK = np.cos(G + psi0)
    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)
    r = a * (1.0 - g0 * cosG)
    cosf = (cosG - g0) / (1.0 - g0 * cosG)
    sinf = beta * sinG / (1.0 - g0 * cosG)
    argp = psi0 - omega0
    sin_u = np.sin(argp) * cosf + np.cos(argp) * sinf
    cos_u = np.cos(argp) * cosf - np.sin(argp) * sinf

    U = -A / (r**3) * (1.0 - 3.0 * s_i * s_i * sin_u * sin_u)
    d = -U / c
    w_h = 3.0 * A * delta * (s_i * s_i) * sin_u * sin_u / (c * r**3)
    xi1 = X / a + 2.0 * p2
    xi2 = Y / a + 2.0 * p1
    p1_dot = p2 * (d - w_h) - (1.0 / c) * xi1 * U
    p2_dot = p1 * (w_h - d) + (1.0 / c) * xi2 * U
    # Derive Psi from the exact p1/p2 GEqOE flow to avoid algebra mistakes.
    psi_dot = (p2 * p1_dot - p1 * p2_dot) / (g0 * g0)

    dg_dG = (r / w) * (U / c) * beta * sinG
    dQ_dG = -6.0 * A * Q0 * c_i / (MU * beta * r**2) * sin_u * cos_u
    dOmega_dG = -6.0 * A * c_i / (MU * beta * r**2) * sin_u * sin_u
    dPsi_dG = (r / w) * psi_dot

    N_prime = dOmega_dG - (r / w) * omega_node
    P_prime = dPsi_dG - (r / w) * omega_peri

    G_corr = _periodic_zero_mean_integral(dg_dG, G)
    Q_corr = _periodic_zero_mean_integral(dQ_dG, G)
    Omega_corr = _periodic_zero_mean_integral(N_prime, G)
    Psi_corr = _periodic_zero_mean_integral(P_prime, G)

    return {
        "G_grid": G,
        "g_corr": G_corr,
        "Q_corr": Q_corr,
        "Omega_corr": Omega_corr,
        "Psi_corr": Psi_corr,
        "g_eval": _interp_periodic(G, G_corr, g_eval),
        "Q_eval": _interp_periodic(G, Q_corr, g_eval),
        "Omega_eval": _interp_periodic(G, Omega_corr, g_eval),
        "Psi_eval": _interp_periodic(G, Psi_corr, g_eval),
    }


def _orbit_means(series: np.ndarray, n_orbits: int, samples_per_orbit: int) -> np.ndarray:
    trimmed = series[: n_orbits * samples_per_orbit]
    return trimmed.reshape(n_orbits, samples_per_orbit).mean(axis=1)


def _build_mean_state(
    state0: np.ndarray,
    *,
    g_mean: float | None = None,
    psi_mean: float | None = None,
    Q_mean: float | None = None,
    Omega_mean: float | None = None,
) -> np.ndarray:
    mean_state = state0.copy()
    g0 = float(np.hypot(state0[1], state0[2]))
    psi0 = float(np.arctan2(state0[1], state0[2]))
    Q0 = float(np.hypot(state0[4], state0[5]))
    Omega0 = float(np.arctan2(state0[4], state0[5]))

    g_use = g0 if g_mean is None else g_mean
    psi_use = psi0 if psi_mean is None else psi_mean
    Q_use = Q0 if Q_mean is None else Q_mean
    Omega_use = Omega0 if Omega_mean is None else Omega_mean

    mean_state[1] = g_use * np.sin(psi_use)
    mean_state[2] = g_use * np.cos(psi_use)
    mean_state[4] = Q_use * np.sin(Omega_use)
    mean_state[5] = Q_use * np.cos(Omega_use)
    return mean_state


def _normalized_rms(reference: np.ndarray, model: np.ndarray) -> float:
    scale = max(1.0e-15, np.max(np.abs(model)))
    return float(np.sqrt(np.mean((reference - model) ** 2)) / scale)


def main() -> None:
    rp_km = 6916.0
    e = 0.74
    a_km = rp_km / (1.0 - e)
    i_deg = 63.4
    raan_deg = 30.0
    argp_deg = 270.0
    mean_anomaly_deg = 45.0

    r0, v0 = _kepler_to_rv(a_km, e, i_deg, raan_deg, argp_deg, mean_anomaly_deg)
    state0 = cart2geqoe(r0, v0, MU, J2Perturbation())

    orbit_period_s = 2.0 * np.pi / state0[0]
    n_orbits = 40
    samples_per_orbit = 240
    t_grid = np.linspace(
        0.0,
        n_orbits * orbit_period_s,
        n_orbits * samples_per_orbit + 1,
        dtype=float,
    )

    ta, _ = build_state_integrator(
        J2Perturbation(),
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )
    states = propagate_grid(ta, t_grid)
    secular = _secular_solution(state0, t_grid)

    g0 = np.hypot(state0[1], state0[2])
    psi0 = np.arctan2(state0[1], state0[2])
    G0 = state0[3] - psi0
    corr = _first_order_j2_short_period_map(state0, G0)
    Q0 = np.hypot(state0[4], state0[5])
    Omega0 = np.arctan2(state0[4], state0[5])
    gQ_mean_state = _build_mean_state(
        state0,
        g_mean=g0 - corr["g_eval"],
        Q_mean=Q0 - corr["Q_eval"],
    )
    full_mean_state = _build_mean_state(
        state0,
        g_mean=g0 - corr["g_eval"],
        psi_mean=psi0 - corr["Psi_eval"],
        Q_mean=Q0 - corr["Q_eval"],
        Omega_mean=Omega0 - corr["Omega_eval"],
    )
    secular_gQ = _secular_solution(gQ_mean_state, t_grid)
    secular_full = _secular_solution(full_mean_state, t_grid)

    g = np.hypot(states[:, 1], states[:, 2])
    Q = np.hypot(states[:, 4], states[:, 5])
    psi = np.unwrap(np.arctan2(states[:, 1], states[:, 2]))
    omega = np.unwrap(np.arctan2(states[:, 4], states[:, 5]))

    t_days = t_grid / 86400.0
    centers = 0.5 * (
        t_grid[: n_orbits * samples_per_orbit : samples_per_orbit]
        + t_grid[samples_per_orbit : n_orbits * samples_per_orbit + samples_per_orbit : samples_per_orbit]
    )
    centers_days = centers / 86400.0

    orbit_mean_data = {
        "p1": _orbit_means(states[:, 1], n_orbits, samples_per_orbit),
        "p2": _orbit_means(states[:, 2], n_orbits, samples_per_orbit),
        "q1": _orbit_means(states[:, 4], n_orbits, samples_per_orbit),
        "q2": _orbit_means(states[:, 5], n_orbits, samples_per_orbit),
        "g": _orbit_means(g, n_orbits, samples_per_orbit),
        "Q": _orbit_means(Q, n_orbits, samples_per_orbit),
        "psi": _orbit_means(psi, n_orbits, samples_per_orbit),
        "Omega": _orbit_means(omega, n_orbits, samples_per_orbit),
    }

    component_fig, component_axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, name, index in zip(
        component_axes.ravel(),
        ("p1", "p2", "q1", "q2"),
        (1, 2, 4, 5),
        strict=True,
    ):
        ax.plot(
            t_days,
            states[:, index],
            color="tab:blue",
            lw=0.8,
            alpha=0.8,
            label="Osculating (full J2)",
        )
        ax.plot(
            t_days,
            secular[name],
            color="tab:red",
            lw=1.5,
            ls="--",
            label="Naive secular model",
        )
        ax.plot(
            t_days,
            secular_gQ[name],
            color="tab:green",
            lw=1.2,
            ls="-.",
            label=r"Numerical inverse map ($g,Q$ only)",
        )
        ax.plot(
            t_days,
            secular_full[name],
            color="tab:orange",
            lw=1.2,
            ls=":",
            label=r"Numerical inverse map (full)",
        )
        ax.plot(
            centers_days,
            orbit_mean_data[name],
            "o",
            color="black",
            ms=2.5,
            label="Per-orbit mean",
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    component_axes[0, 0].legend(loc="best", fontsize=8)
    component_axes[1, 0].set_xlabel("Time [days]")
    component_axes[1, 1].set_xlabel("Time [days]")
    component_fig.suptitle("J2 GEqOE: osculating components vs secular closure")
    component_fig.tight_layout()
    component_path = FIG_DIR / "j2_secular_vs_osculating_components.png"
    component_fig.savefig(component_path, dpi=180)
    plt.close(component_fig)

    diag_fig, diag_axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    diag_axes[0, 0].plot(
        t_days,
        psi,
        color="tab:blue",
        lw=0.8,
        alpha=0.8,
        label="Osculating phase",
    )
    diag_axes[0, 0].plot(
        t_days,
        secular["psi"],
        color="tab:red",
        lw=1.5,
        ls="--",
        label="Secular phase",
    )
    diag_axes[0, 0].plot(
        t_days,
        secular_gQ["psi"],
        color="tab:green",
        lw=1.2,
        ls="-.",
        label=r"Inverse map ($g,Q$ only)",
    )
    diag_axes[0, 0].plot(
        t_days,
        secular_full["psi"],
        color="tab:orange",
        lw=1.2,
        ls=":",
        label="Inverse map (full)",
    )
    diag_axes[0, 0].plot(
        centers_days,
        orbit_mean_data["psi"],
        "o",
        color="black",
        ms=2.5,
        label="Per-orbit mean",
    )
    diag_axes[0, 0].set_ylabel(r"$\Psi$ [rad]")
    diag_axes[0, 0].grid(True, alpha=0.25)
    diag_axes[0, 0].legend(loc="best", fontsize=8)

    diag_axes[0, 1].plot(t_days, omega, color="tab:blue", lw=0.8, alpha=0.8)
    diag_axes[0, 1].plot(t_days, secular["Omega"], color="tab:red", lw=1.5, ls="--")
    diag_axes[0, 1].plot(t_days, secular_gQ["Omega"], color="tab:green", lw=1.2, ls="-.")
    diag_axes[0, 1].plot(t_days, secular_full["Omega"], color="tab:orange", lw=1.2, ls=":")
    diag_axes[0, 1].plot(centers_days, orbit_mean_data["Omega"], "o", color="black", ms=2.5)
    diag_axes[0, 1].set_ylabel(r"$\Omega$ [rad]")
    diag_axes[0, 1].grid(True, alpha=0.25)

    g_residual = g - secular["g"]
    Q_residual = Q - secular["Q"]
    g_residual_corrected = g - secular_gQ["g"]
    Q_residual_corrected = Q - secular_gQ["Q"]
    diag_axes[1, 0].plot(t_days, g_residual, color="tab:blue", lw=0.8, alpha=0.8)
    diag_axes[1, 0].plot(t_days, np.zeros_like(t_grid), color="tab:red", lw=1.5, ls="--")
    diag_axes[1, 0].plot(t_days, g_residual_corrected, color="tab:green", lw=1.2, ls="-.")
    diag_axes[1, 0].plot(
        centers_days,
        orbit_mean_data["g"] - secular["g"][0],
        "o",
        color="black",
        ms=2.5,
    )
    diag_axes[1, 0].set_ylabel(r"$g-g_{\mathrm{sec}}$")
    diag_axes[1, 0].set_xlabel("Time [days]")
    diag_axes[1, 0].grid(True, alpha=0.25)

    diag_axes[1, 1].plot(t_days, Q_residual, color="tab:blue", lw=0.8, alpha=0.8)
    diag_axes[1, 1].plot(t_days, np.zeros_like(t_grid), color="tab:red", lw=1.5, ls="--")
    diag_axes[1, 1].plot(t_days, Q_residual_corrected, color="tab:green", lw=1.2, ls="-.")
    diag_axes[1, 1].plot(
        centers_days,
        orbit_mean_data["Q"] - secular["Q"][0],
        "o",
        color="black",
        ms=2.5,
    )
    diag_axes[1, 1].set_ylabel(r"$Q-Q_{\mathrm{sec}}$")
    diag_axes[1, 1].set_xlabel("Time [days]")
    diag_axes[1, 1].grid(True, alpha=0.25)

    diag_fig.suptitle("J2 GEqOE: phase and magnitude diagnostics")
    diag_fig.tight_layout()
    diag_path = FIG_DIR / "j2_secular_vs_osculating_diagnostics.png"
    diag_fig.savefig(diag_path, dpi=180)
    plt.close(diag_fig)

    sec_centers = _secular_solution(state0, centers)
    sec_centers_gQ = _secular_solution(gQ_mean_state, centers)
    sec_centers_full = _secular_solution(full_mean_state, centers)
    print("Per-orbit RMS mismatch against naive secular initialization:")
    for name in ("p1", "p2", "q1", "q2"):
        print(f"  {name}: {_normalized_rms(orbit_mean_data[name], sec_centers[name]):.6e}")
    print(r"Per-orbit RMS mismatch against numerical inverse map ($g,Q$ only):")
    for name in ("p1", "p2", "q1", "q2"):
        print(f"  {name}: {_normalized_rms(orbit_mean_data[name], sec_centers_gQ[name]):.6e}")
    print(r"Per-orbit RMS mismatch against full numerical inverse map ($g,Q,\Psi,\Omega$):")
    for name in ("p1", "p2", "q1", "q2"):
        print(f"  {name}: {_normalized_rms(orbit_mean_data[name], sec_centers_full[name]):.6e}")
    print("Phase RMS mismatch against naive secular initialization:")
    psi_scale = abs(sec_centers["psi"][-1] - sec_centers["psi"][0])
    omega_scale = abs(sec_centers["Omega"][-1] - sec_centers["Omega"][0])
    print(
        f"  psi:   {np.sqrt(np.mean((orbit_mean_data['psi'] - sec_centers['psi']) ** 2)) / psi_scale:.6e}"
    )
    print(
        f"  Omega: {np.sqrt(np.mean((orbit_mean_data['Omega'] - sec_centers['Omega']) ** 2)) / omega_scale:.6e}"
    )
    print(r"Phase RMS mismatch against numerical inverse map ($g,Q$ only):")
    print(
        f"  psi:   {np.sqrt(np.mean((orbit_mean_data['psi'] - sec_centers_gQ['psi']) ** 2)) / psi_scale:.6e}"
    )
    print(
        f"  Omega: {np.sqrt(np.mean((orbit_mean_data['Omega'] - sec_centers_gQ['Omega']) ** 2)) / omega_scale:.6e}"
    )
    print(r"Phase RMS mismatch against full numerical inverse map ($g,Q,\Psi,\Omega$):")
    print(
        f"  psi:   {np.sqrt(np.mean((orbit_mean_data['psi'] - sec_centers_full['psi']) ** 2)) / psi_scale:.6e}"
    )
    print(
        f"  Omega: {np.sqrt(np.mean((orbit_mean_data['Omega'] - sec_centers_full['Omega']) ** 2)) / omega_scale:.6e}"
    )
    print(f"  g bias (naive init): {orbit_mean_data['g'][0] - secular['g'][0]:.6e}")
    print(f"  Q bias (naive init): {orbit_mean_data['Q'][0] - secular['Q'][0]:.6e}")
    print(f"  g bias (g,Q only):   {orbit_mean_data['g'][0] - secular_gQ['g'][0]:.6e}")
    print(f"  Q bias (g,Q only):   {orbit_mean_data['Q'][0] - secular_gQ['Q'][0]:.6e}")
    print(f"  g bias (full map):   {orbit_mean_data['g'][0] - secular_full['g'][0]:.6e}")
    print(f"  Q bias (full map):   {orbit_mean_data['Q'][0] - secular_full['Q'][0]:.6e}")
    print(f"  Applied map at t0: Delta g = {corr['g_eval']:.6e}")
    print(f"  Applied map at t0: Delta Q = {corr['Q_eval']:.6e}")
    print(f"  Applied map at t0: Delta Psi = {corr['Psi_eval']:.6e}")
    print(f"  Applied map at t0: Delta Omega = {corr['Omega_eval']:.6e}")
    print(f"Wrote {component_path}")
    print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
