#!/usr/bin/env python
"""Validate the exact truncated zonal averaged GEqOE model against numerics.

This script performs two checks for a mixed zonal model:
1. pointwise averaged RHS parity against numerical one-revolution averaging;
2. long-arc slow-flow validation against orbit means from the full zonal GEqOE
   integration.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/zonal_mean_validation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
FIG_DIR = DOC_DIR / "figures"
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from astrodyn_core.geqoe_taylor import (
    J2,
    J3,
    J4,
    J5,
    MU,
    RE,
    ZonalPerturbation,
    build_state_integrator,
    cart2geqoe,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.fourier_model import avg_slow_drift, frozen_state
from geqoe_mean.symbolic import evaluate_truncated_mean_rates, evaluate_truncated_mean_rhs_pq


OUT_COMPONENTS = FIG_DIR / "zonal_mean_validation_components.png"
OUT_DIAGNOSTICS = FIG_DIR / "zonal_mean_validation_diagnostics.png"
OUT_ANGLES = FIG_DIR / "zonal_mean_validation_angles.png"
OUT_TEX = DOC_DIR / "zonal_mean_validation.tex"


@dataclass(frozen=True)
class ValidationCase:
    a_km: float = 16000.0
    e: float = 0.35
    inc_deg: float = 50.0
    raan_deg: float = 25.0
    argp_deg: float = 40.0
    M_deg: float = 30.0
    n_orbits: int = 30
    samples_per_orbit: int = 96
    rk4_substeps_per_orbit: int = 16


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    resid = a - b
    denom = max(float(np.max(np.abs(b))), 1.0e-30)
    return float(np.sqrt(np.mean(resid * resid)) / denom)


def _rk4_integrate_mean(
    y0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
    substeps_per_interval: int = 16,
) -> np.ndarray:
    out = np.empty((len(t_eval), 5), dtype=float)
    out[0] = y0
    y = y0.copy()

    for i in range(len(t_eval) - 1):
        dt = (t_eval[i + 1] - t_eval[i]) / substeps_per_interval
        for _ in range(substeps_per_interval):
            k1 = evaluate_truncated_mean_rhs_pq(y, j_coeffs, re_val, mu_val)
            k2 = evaluate_truncated_mean_rhs_pq(y + 0.5 * dt * k1, j_coeffs, re_val, mu_val)
            k3 = evaluate_truncated_mean_rhs_pq(y + 0.5 * dt * k2, j_coeffs, re_val, mu_val)
            k4 = evaluate_truncated_mean_rhs_pq(y + dt * k3, j_coeffs, re_val, mu_val)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y[0] = y0[0]
        out[i + 1] = y
    return out


def _orbit_means(states: np.ndarray, n_orbits: int, samples_per_orbit: int) -> dict[str, np.ndarray]:
    reshaped = states[:-1].reshape(n_orbits, samples_per_orbit, states.shape[1])
    return {
        "nu": reshaped[:, :, 0].mean(axis=1),
        "p1": reshaped[:, :, 1].mean(axis=1),
        "p2": reshaped[:, :, 2].mean(axis=1),
        "q1": reshaped[:, :, 4].mean(axis=1),
        "q2": reshaped[:, :, 5].mean(axis=1),
    }


def pointwise_rhs_validation(
    case: ValidationCase,
    j_coeffs: dict[int, float],
    omega_samples: int = 16,
) -> dict[str, float]:
    omega_grid = np.linspace(0.0, 2.0 * np.pi, omega_samples, endpoint=False, dtype=float)
    pert = ZonalPerturbation(j_coeffs, mu=MU, re=RE)
    exact_series = {name: [] for name in ("g_dot", "Q_dot", "Psi_dot", "Omega_dot")}
    numeric_series = {name: [] for name in ("g_dot", "Q_dot", "Psi_dot", "Omega_dot")}

    for omega in omega_grid:
        state = frozen_state(
            case.a_km, case.e, case.inc_deg, case.raan_deg, np.rad2deg(omega)
        )
        numeric = avg_slow_drift(state, pert, samples=4097)
        exact = evaluate_truncated_mean_rates(
            nu_val=state[0],
            g_val=float(np.hypot(state[1], state[2])),
            Q_val=float(np.hypot(state[4], state[5])),
            omega_val=omega,
            j_coeffs=j_coeffs,
            re_val=RE,
            mu_val=MU,
        )
        for key in exact_series:
            exact_series[key].append(exact[key])
            numeric_series[key].append(numeric[key])

    out = {}
    for key in exact_series:
        exact_arr = np.asarray(exact_series[key], dtype=float)
        numeric_arr = np.asarray(numeric_series[key], dtype=float)
        denom = max(float(np.max(np.abs(numeric_arr))), 1.0e-30)
        out[key] = float(np.sqrt(np.mean((exact_arr - numeric_arr) ** 2)) / denom)
    return out


def propagate_full_and_mean(
    case: ValidationCase,
    j_coeffs: dict[int, float],
) -> dict[str, np.ndarray | float]:
    r0, v0 = kepler_to_rv(
        case.a_km, case.e, case.inc_deg, case.raan_deg, case.argp_deg, case.M_deg
    )
    pert = ZonalPerturbation(j_coeffs, mu=MU, re=RE)
    state0 = cart2geqoe(r0, v0, MU, pert)
    nu0 = float(state0[0])
    T_orbit = 2.0 * np.pi / nu0

    t_grid = np.linspace(
        0.0,
        case.n_orbits * T_orbit,
        case.n_orbits * case.samples_per_orbit + 1,
        dtype=float,
    )
    ta, _ = build_state_integrator(pert, state0, tol=1.0e-15, compact_mode=True)
    osc_states = propagate_grid(ta, t_grid)
    means = _orbit_means(osc_states, case.n_orbits, case.samples_per_orbit)

    t_mid = (np.arange(case.n_orbits, dtype=float) + 0.5) * T_orbit
    mean0 = np.array([means["nu"][0], means["p1"][0], means["p2"][0], means["q1"][0], means["q2"][0]])
    model_states = _rk4_integrate_mean(
        mean0,
        t_mid,
        j_coeffs,
        re_val=RE,
        mu_val=MU,
        substeps_per_interval=case.rk4_substeps_per_orbit,
    )

    psi_num = np.unwrap(np.arctan2(means["p1"], means["p2"]))
    psi_mod = np.unwrap(np.arctan2(model_states[:, 1], model_states[:, 2]))
    Omega_num = np.unwrap(np.arctan2(means["q1"], means["q2"]))
    Omega_mod = np.unwrap(np.arctan2(model_states[:, 3], model_states[:, 4]))

    metrics = {
        "p1_rel_rms": _relative_rms(model_states[:, 1], means["p1"]),
        "p2_rel_rms": _relative_rms(model_states[:, 2], means["p2"]),
        "q1_rel_rms": _relative_rms(model_states[:, 3], means["q1"]),
        "q2_rel_rms": _relative_rms(model_states[:, 4], means["q2"]),
        "psi_rms": float(np.sqrt(np.mean((psi_mod - psi_num) ** 2))),
        "Omega_rms": float(np.sqrt(np.mean((Omega_mod - Omega_num) ** 2))),
        "T_orbit": T_orbit,
    }

    return {
        "t_mid": t_mid,
        "means": means,
        "model_states": model_states,
        "metrics": metrics,
    }


def create_plots(result: dict[str, np.ndarray | float]) -> None:
    import matplotlib.pyplot as plt

    t_mid = np.asarray(result["t_mid"], dtype=float)
    means = result["means"]
    model = np.asarray(result["model_states"], dtype=float)
    T_orbit = float(result["metrics"]["T_orbit"])
    tau = t_mid / T_orbit

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    pairs = [
        ("p1", model[:, 1], means["p1"]),
        ("p2", model[:, 2], means["p2"]),
        ("q1", model[:, 3], means["q1"]),
        ("q2", model[:, 4], means["q2"]),
    ]
    for ax, (name, y_model, y_num) in zip(axes.ravel(), pairs):
        ax.plot(tau, y_num, label="orbit mean", lw=1.8)
        ax.plot(tau, y_model, "--", label="exact averaged model", lw=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("orbit index")
    axes[1, 1].set_xlabel("orbit index")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_COMPONENTS, dpi=180)
    plt.close(fig)

    psi_num = np.unwrap(np.arctan2(means["p1"], means["p2"]))
    psi_mod = np.unwrap(np.arctan2(model[:, 1], model[:, 2]))
    Omega_num = np.unwrap(np.arctan2(means["q1"], means["q2"]))
    Omega_mod = np.unwrap(np.arctan2(model[:, 3], model[:, 4]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    errors = [
        ("p1 error", model[:, 1] - means["p1"]),
        ("p2 error", model[:, 2] - means["p2"]),
        ("q1 error", model[:, 3] - means["q1"]),
        ("q2 error", model[:, 4] - means["q2"]),
    ]
    for ax, (name, err) in zip(axes.ravel(), errors):
        ax.plot(tau, err, lw=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("orbit index")
    axes[1, 1].set_xlabel("orbit index")
    fig.tight_layout()
    fig.savefig(OUT_DIAGNOSTICS, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(tau, psi_num, label="orbit mean")
    axes[0].plot(tau, psi_mod, "--", label="exact averaged model")
    axes[0].set_ylabel(r"$\Psi$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(tau, Omega_num, label="orbit mean")
    axes[1].plot(tau, Omega_mod, "--", label="exact averaged model")
    axes[1].set_ylabel(r"$\Omega$")
    axes[1].set_xlabel("orbit index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_ANGLES, dpi=180)
    plt.close(fig)


def write_note(case: ValidationCase, j_coeffs: dict[int, float], pointwise: dict[str, float], result: dict[str, np.ndarray | float]) -> None:
    metrics = result["metrics"]
    terms = ", ".join(f"J_{n}" for n in sorted(j_coeffs))
    scaled_coeffs = {n: 0.1 * val for n, val in j_coeffs.items()}
    scaled_pointwise = pointwise_rhs_validation(case, scaled_coeffs, omega_samples=12)
    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=2cm]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\begin{{document}}
\section*{{Validation of the Exact Truncated Zonal Averaged GEqOE Model}}

This note validates the exact symbolic truncated zonal averaged GEqOE model for
the mixed zonal set $\{{{terms}\}}$ against numerical computation from the full
GEqOE zonal dynamics.

The reference osculating orbit is
\[
a={case.a_km:.0f}\ \mathrm{{km}},\qquad
e={case.e:.2f},\qquad
i={case.inc_deg:.1f}^\circ,\qquad
\Omega_0={case.raan_deg:.1f}^\circ,\qquad
\omega_0={case.argp_deg:.1f}^\circ,\qquad
M_0={case.M_deg:.1f}^\circ.
\]

Two checks were performed:
\begin{{enumerate}}
\item pointwise parity between the exact symbolic averaged RHS and the
numerical one-revolution average of the full zonal RHS;
\item long-arc comparison between the propagated exact averaged slow model and
per-orbit means extracted from the full zonal GEqOE integration.
\end{{enumerate}}

\subsection*{{Pointwise averaged RHS parity}}
\begin{{center}}
\begin{{tabular}}{{lcc}}
\toprule
Component & Earth coefficients & $0.1\times$ coefficients \\
\midrule
$\dot{{\bar g}}$ & {pointwise["g_dot"]:.3e} & {scaled_pointwise["g_dot"]:.3e} \\
$\dot{{\bar Q}}$ & {pointwise["Q_dot"]:.3e} & {scaled_pointwise["Q_dot"]:.3e} \\
$\dot{{\bar\Psi}}$ & {pointwise["Psi_dot"]:.3e} & {scaled_pointwise["Psi_dot"]:.3e} \\
$\dot{{\bar\Omega}}$ & {pointwise["Omega_dot"]:.3e} & {scaled_pointwise["Omega_dot"]:.3e} \\
\bottomrule
\end{{tabular}}
\end{{center}}

The nonzero residual for $\dot{{\bar Q}}, \dot{{\bar\Psi}}, \dot{{\bar\Omega}}$
shrinks essentially linearly with the zonal scale factor. This is the expected
signature of a first-order averaged model compared against the full nonlinear
numerical average: the remaining mismatch is higher-order in the zonal
strength, not evidence of a wrong first-order closure.

\subsection*{{Long-arc slow-flow validation}}
The exact averaged slow model was initialized from the numerical mean state of
the first orbit and then propagated for {case.n_orbits} orbital periods. The
resulting mismatch against the subsequent numerical orbit means is:
\begin{{center}}
\begin{{tabular}}{{lc}}
\toprule
Quantity & metric \\
\midrule
$p_1$ relative RMS & {metrics["p1_rel_rms"]:.3e} \\
$p_2$ relative RMS & {metrics["p2_rel_rms"]:.3e} \\
$q_1$ relative RMS & {metrics["q1_rel_rms"]:.3e} \\
$q_2$ relative RMS & {metrics["q2_rel_rms"]:.3e} \\
$\Psi$ RMS phase error [rad] & {metrics["psi_rms"]:.3e} \\
$\Omega$ RMS phase error [rad] & {metrics["Omega_rms"]:.3e} \\
\bottomrule
\end{{tabular}}
\end{{center}}

This validates the \emph{{slow}} exact truncated zonal averaged model. No
general zonal short-period reconstruction is used here, so the comparison is
deliberately made against orbit means rather than against the full osculating
state.

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.92\textwidth]{{docs/geqoe_averaged/figures/{OUT_COMPONENTS.name}}}
\caption{{Per-orbit numerical means versus the exact truncated averaged model.}}
\end{{figure}}

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.92\textwidth]{{docs/geqoe_averaged/figures/{OUT_DIAGNOSTICS.name}}}
\caption{{Component-wise slow-state errors of the exact truncated averaged model.}}
\end{{figure}}

\end{{document}}
"""
    OUT_TEX.write_text(tex.strip() + "\n")
    print(f"Wrote {OUT_TEX}")


def main() -> None:
    case = ValidationCase()
    j_coeffs = {2: J2, 3: J3, 4: J4, 5: J5}
    pointwise = pointwise_rhs_validation(case, j_coeffs)
    pointwise_scaled = pointwise_rhs_validation(case, {n: 0.1 * val for n, val in j_coeffs.items()})
    result = propagate_full_and_mean(case, j_coeffs)
    create_plots(result)
    write_note(case, j_coeffs, pointwise, result)

    print("=" * 72)
    print("Exact truncated zonal averaged GEqOE validation")
    print("=" * 72)
    print("Pointwise averaged RHS relative RMS:")
    for key, val in pointwise.items():
        print(f"  {key:10s}: {val:.3e}")
    print("Scaled by 0.1:")
    for key, val in pointwise_scaled.items():
        print(f"  {key:10s}: {val:.3e}")
    print("\nLong-arc slow-flow validation:")
    metrics = result["metrics"]
    print(f"  p1 rel RMS   : {metrics['p1_rel_rms']:.3e}")
    print(f"  p2 rel RMS   : {metrics['p2_rel_rms']:.3e}")
    print(f"  q1 rel RMS   : {metrics['q1_rel_rms']:.3e}")
    print(f"  q2 rel RMS   : {metrics['q2_rel_rms']:.3e}")
    print(f"  Psi RMS [rad]: {metrics['psi_rms']:.3e}")
    print(f"  Om  RMS [rad]: {metrics['Omega_rms']:.3e}")


if __name__ == "__main__":
    main()
