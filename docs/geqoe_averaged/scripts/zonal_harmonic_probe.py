#!/usr/bin/env python
"""Probe the harmonic structure of the averaged GEqOE zonal drift.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/zonal_harmonic_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astrodyn_core.geqoe_taylor import J2, J3, J4, J5, MU, RE, ZonalPerturbation


def _frozen_state(a_km: float, e: float, inc_deg: float, raan_deg: float, argp_deg: float) -> np.ndarray:
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    nu = np.sqrt(MU / a_km**3)
    g = e
    Q = np.tan(inc / 2.0)
    Psi = raan + argp
    Omega = raan
    return np.array([nu, g * np.sin(Psi), g * np.cos(Psi), 0.0, Q * np.sin(Omega), Q * np.cos(Omega)], dtype=float)


def _avg_slow_drift(state: np.ndarray, pert: ZonalPerturbation, samples: int = 4097) -> dict[str, float]:
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
        "p1_dot": float(p1_bar),
        "p2_dot": float(p2_bar),
        "q1_dot": float(q1_bar),
        "q2_dot": float(q2_bar),
        "g_dot": float((p1 * p1_bar + p2 * p2_bar) / g),
        "Q_dot": float((q1 * q1_bar + q2 * q2_bar) / Q),
        "Psi_dot": float((p2 * p1_bar - p1 * p2_bar) / (g * g)),
        "Omega_dot": float((q2 * q1_bar - q1 * q2_bar) / (Q * Q)),
    }


def _fit_fourier(omega: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, list[str], float]:
    cols = [np.ones_like(omega)]
    names = ["1"]
    for m in range(1, degree + 1):
        cols.append(np.cos(m * omega))
        cols.append(np.sin(m * omega))
        names.append(f"cos({m}w)")
        names.append(f"sin({m}w)")
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coeffs
    denom = max(np.max(np.abs(y)), 1.0e-30)
    return coeffs, names, float(np.sqrt(np.mean(resid * resid)) / denom)


def _summarize_modes(coeffs: np.ndarray, names: list[str], rel_cutoff: float = 1.0e-3) -> str:
    amp = np.max(np.abs(coeffs))
    if amp < 1.0e-30:
        return "0"
    keep = [f"{name}={coef:+.3e}" for name, coef in zip(names, coeffs) if abs(coef) >= rel_cutoff * amp]
    return ", ".join(keep) if keep else "0"


def main() -> None:
    # Use a generic, non-symmetric orbit so angle-dependent terms are visible.
    a_km = 16000.0
    e = 0.35
    inc_deg = 50.0
    raan_deg = 25.0
    omega_grid = np.linspace(0.0, 2.0 * np.pi, 73, dtype=float)[:-1]

    cases = {
        2: {2: J2},
        3: {3: J3},
        4: {4: J4},
        5: {5: J5},
    }
    variables = ("g_dot", "Q_dot", "Psi_dot", "Omega_dot")

    print("=" * 72)
    print("Averaged GEqOE zonal harmonic probe")
    print("=" * 72)
    print(f"Reference orbit: a={a_km:.1f} km, e={e:.2f}, i={inc_deg:.1f} deg, RAAN={raan_deg:.1f} deg")
    print("Model: frozen-state first-order K-averaged slow drift from the exact zonal RHS\n")

    for degree, coeffs in cases.items():
        pert = ZonalPerturbation(coeffs, mu=MU, re=RE)
        series = {name: [] for name in variables}
        for omega in omega_grid:
            state = _frozen_state(a_km, e, inc_deg, raan_deg, np.rad2deg(omega))
            drift = _avg_slow_drift(state, pert)
            for name in variables:
                series[name].append(drift[name])

        print("-" * 72)
        print(f"Degree n={degree}")
        for name in variables:
            y = np.asarray(series[name], dtype=float)
            coeff_fit, names, rel_rms = _fit_fourier(omega_grid, y, degree)
            print(f"  {name:9s} rel fit rms={rel_rms:.3e}  modes: {_summarize_modes(coeff_fit, names)}")
        print()


if __name__ == "__main__":
    main()
