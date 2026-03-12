"""Frozen-state numerical averaging and Fourier model fitting for GEqOE zonal drift."""

from __future__ import annotations

import numpy as np

from astrodyn_core.geqoe_taylor import J2, J3, J4, J5, MU, RE, ZonalPerturbation


def frozen_state(a_km: float, e: float, inc_deg: float, raan_deg: float, argp_deg: float) -> np.ndarray:
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    nu = np.sqrt(MU / a_km**3)
    Q = np.tan(inc / 2.0)
    Psi = raan + argp
    Omega = raan
    return np.array([nu, e * np.sin(Psi), e * np.cos(Psi), 0.0, Q * np.sin(Omega), Q * np.cos(Omega)], dtype=float)


def avg_slow_drift(state: np.ndarray, pert: ZonalPerturbation, samples: int = 4097) -> dict[str, float]:
    nu, p1, p2, _, q1, q2 = state
    g = np.hypot(p1, p2)
    Q = np.hypot(q1, q2)
    beta = np.sqrt(1.0 - g * g)
    alpha = 1.0 / (1.0 + beta)
    a = (MU / (nu * nu)) ** (1.0 / 3.0)
    c = (MU * MU / nu) ** (1.0 / 3.0) * beta
    gamma = 1.0 + Q * Q
    delta = 1.0 - Q * Q

    K = np.linspace(0.0, 2.0 * np.pi, samples, dtype=float)
    sinK = np.sin(K)
    cosK = np.cos(K)
    X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
    Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)
    r = a * (1.0 - p1 * sinK - p2 * cosK)
    zhat = 2.0 * (Y * q2 - X * q1) / (gamma * r)

    U, dU_dzhat, euler = pert.zonal_quantities(r, zhat)
    h = np.sqrt(c * c - 2.0 * r * r * U)
    d = (h - c) / (r * r)
    F_h = -dU_dzhat * delta / (gamma * r)
    w_X = (X / h) * F_h
    w_Y = (Y / h) * F_h
    w_h = w_X * q1 - w_Y * q2

    p1_dot = p2 * (d - w_h) + (1.0 / c) * (X / a + 2.0 * p2) * euler
    p2_dot = p1 * (w_h - d) - (1.0 / c) * (Y / a + 2.0 * p1) * euler
    q1_dot = 0.5 * gamma * w_Y
    q2_dot = 0.5 * gamma * w_X

    scale = 1.0 / (2.0 * np.pi * a)
    p1_bar = scale * np.trapezoid(r * p1_dot, K)
    p2_bar = scale * np.trapezoid(r * p2_dot, K)
    q1_bar = scale * np.trapezoid(r * q1_dot, K)
    q2_bar = scale * np.trapezoid(r * q2_dot, K)

    return {
        "g_dot": float((p1 * p1_bar + p2 * p2_bar) / g),
        "Q_dot": float((q1 * q1_bar + q2 * q2_bar) / Q),
        "Psi_dot": float((p2 * p1_bar - p1 * p2_bar) / (g * g)),
        "Omega_dot": float((q2 * q1_bar - q1 * q2_bar) / (Q * Q)),
    }


def _basis_columns(omega: np.ndarray, n_max: int, family: str) -> tuple[np.ndarray, list[str]]:
    cols = []
    names = []
    if family == "mag":
        for m in range(1, n_max + 1, 2):
            cols.append(np.cos(m * omega))
            names.append(f"cos({m}w)")
        for m in range(2, n_max + 1, 2):
            cols.append(np.sin(m * omega))
            names.append(f"sin({m}w)")
    elif family == "ang":
        cols.append(np.ones_like(omega))
        names.append("1")
        for m in range(1, n_max + 1, 2):
            cols.append(np.sin(m * omega))
            names.append(f"sin({m}w)")
        for m in range(2, n_max + 1, 2):
            cols.append(np.cos(m * omega))
            names.append(f"cos({m}w)")
    else:
        raise ValueError(f"Unknown family: {family}")
    return np.column_stack(cols), names


def fit_total_order_model(omega: np.ndarray, values: np.ndarray, n_max: int, family: str) -> tuple[np.ndarray, list[str], float]:
    A, names = _basis_columns(omega, n_max, family)
    coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
    resid = values - A @ coeffs
    denom = max(np.max(np.abs(values)), 1.0e-30)
    rel_rms = float(np.sqrt(np.mean(resid * resid)) / denom)
    return coeffs, names, rel_rms


def evaluate_total_order_model(omega: np.ndarray, coeffs: np.ndarray, n_max: int, family: str) -> np.ndarray:
    A, _ = _basis_columns(omega, n_max, family)
    return A @ coeffs


def project_total_order_model(
    a_km: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    j_coeffs: dict[int, float],
    omega_grid: np.ndarray,
    samples_k: int = 4097,
) -> dict[str, tuple[np.ndarray, list[str], float]]:
    pert = ZonalPerturbation(j_coeffs, mu=MU, re=RE)
    n_max = max(j_coeffs)
    series = {name: [] for name in ("g_dot", "Q_dot", "Psi_dot", "Omega_dot")}
    for omega in omega_grid:
        state = frozen_state(a_km, e, inc_deg, raan_deg, np.rad2deg(omega))
        drift = avg_slow_drift(state, pert, samples=samples_k)
        for name in series:
            series[name].append(drift[name])

    out = {}
    for name, vals in series.items():
        family = "mag" if name in ("g_dot", "Q_dot") else "ang"
        out[name] = fit_total_order_model(omega_grid, np.asarray(vals, dtype=float), n_max, family)
    return out


def _summarize_coeffs(coeffs: np.ndarray, names: list[str], rel_cutoff: float = 1.0e-3) -> str:
    amp = np.max(np.abs(coeffs))
    if amp < 1.0e-30:
        return "0"
    keep = [f"{name}={coef:+.3e}" for name, coef in zip(names, coeffs) if abs(coef) >= rel_cutoff * amp]
    return ", ".join(keep) if keep else "0"


def main() -> None:
    a_km = 16000.0
    e = 0.35
    inc_deg = 50.0
    raan_deg = 25.0
    omega_grid = np.linspace(0.0, 2.0 * np.pi, 97, dtype=float)[:-1]

    cases = {
        "J2+J3": {2: J2, 3: J3},
        "J2+J3+J4": {2: J2, 3: J3, 4: J4},
        "J2+J3+J4+J5": {2: J2, 3: J3, 4: J4, 5: J5},
    }

    print("=" * 72)
    print("Finite Fourier averaged zonal GEqOE model")
    print("=" * 72)
    print(f"Reference orbit: a={a_km:.1f} km, e={e:.2f}, i={inc_deg:.1f} deg, RAAN={raan_deg:.1f} deg")
    print("Model basis:")
    print("  g_dot, Q_dot     -> cos(odd*omega) + sin(even*omega)")
    print("  Psi_dot, Omega_dot -> 1 + sin(odd*omega) + cos(even*omega)\n")

    for label, coeffs_map in cases.items():
        n_max = max(coeffs_map)
        print("-" * 72)
        print(f"{label}  (N={n_max})")
        fitted = project_total_order_model(a_km, e, inc_deg, raan_deg, coeffs_map, omega_grid)
        for name in ("g_dot", "Q_dot", "Psi_dot", "Omega_dot"):
            coeffs, names, rel_rms = fitted[name]
            print(f"  {name:9s} rel fit rms={rel_rms:.3e}  coeffs: {_summarize_coeffs(coeffs, names)}")
        print()


if __name__ == "__main__":
    main()
