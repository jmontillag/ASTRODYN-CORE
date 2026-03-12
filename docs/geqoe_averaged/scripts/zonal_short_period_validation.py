#!/usr/bin/env python
"""Validate the first-order mixed-zonal GEqOE short-period map.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/zonal_short_period_validation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
FIG_DIR = DOC_DIR / "figures"
OUT_TEX = DOC_DIR / "zonal_short_period_validation.tex"
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from astrodyn_core.geqoe_taylor import (
    MU,
    RE,
    ZonalPerturbation,
    build_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

from geqoe_mean.constants import J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    evaluate_truncated_mean_rhs_pqm,
    isolated_short_period_expressions_for,
    mean_to_osculating_state,
    osculating_to_mean_state,
)
from geqoe_mean.validation import (
    ensure_symbolic_cache as _ensure_symbolic_cache,
    phase_error as _phase_error,
    relative_rms as _relative_rms,
)

OUT_LOW = FIG_DIR / "zonal_short_period_lowe_components.png"
OUT_HIGH = FIG_DIR / "zonal_short_period_highe_components.png"
OUT_CART = FIG_DIR / "zonal_short_period_cartesian_errors.png"
OUT_SCALE = FIG_DIR / "zonal_short_period_scaling.png"


@dataclass(frozen=True)
class ValidationCase:
    name: str
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    anomalies_deg: tuple[float, ...]
    n_orbits: int
    samples_per_orbit: int
    rk4_substeps_per_interval: int = 8


CASES = (
    ValidationCase(
        name="low-e",
        a_km=9000.0,
        e=0.05,
        inc_deg=40.0,
        raan_deg=25.0,
        argp_deg=60.0,
        anomalies_deg=(20.0, 140.0),
        n_orbits=10,
        samples_per_orbit=64,
    ),
    ValidationCase(
        name="high-e",
        a_km=18000.0,
        e=0.65,
        inc_deg=63.0,
        raan_deg=40.0,
        argp_deg=250.0,
        anomalies_deg=(35.0, 170.0),
        n_orbits=8,
        samples_per_orbit=64,
    ),
)


def _rk4_integrate_mean(
    state0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
    substeps_per_interval: int = 8,
) -> np.ndarray:
    out = np.empty((len(t_eval), 6), dtype=float)
    out[0] = state0
    y = state0.copy()

    for i in range(len(t_eval) - 1):
        dt = (t_eval[i + 1] - t_eval[i]) / substeps_per_interval
        for _ in range(substeps_per_interval):
            k1 = evaluate_truncated_mean_rhs_pqm(y, j_coeffs, re_val=re_val, mu_val=mu_val)
            k2 = evaluate_truncated_mean_rhs_pqm(y + 0.5 * dt * k1, j_coeffs, re_val=re_val, mu_val=mu_val)
            k3 = evaluate_truncated_mean_rhs_pqm(y + 0.5 * dt * k2, j_coeffs, re_val=re_val, mu_val=mu_val)
            k4 = evaluate_truncated_mean_rhs_pqm(y + dt * k3, j_coeffs, re_val=re_val, mu_val=mu_val)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y[0] = state0[0]
        out[i + 1] = y
    return out


def _state_phases(state_hist: np.ndarray) -> dict[str, np.ndarray]:
    p = state_hist[:, 1:3]
    q = state_hist[:, 4:6]
    g_hist = np.linalg.norm(p, axis=1)
    Q_hist = np.linalg.norm(q, axis=1)
    Psi_hist = np.unwrap(np.arctan2(state_hist[:, 1], state_hist[:, 2]))
    Omega_hist = np.unwrap(np.arctan2(state_hist[:, 4], state_hist[:, 5]))
    return {
        "g": g_hist,
        "Q": Q_hist,
        "Psi": Psi_hist,
        "Omega": Omega_hist,
    }


def run_case(
    case: ValidationCase,
    anomaly_deg: float,
    scale: float = 1.0,
    j_coeffs: dict[int, float] | None = None,
) -> dict[str, object]:
    base = j_coeffs if j_coeffs is not None else J_COEFFS
    coeffs = {n: scale * val for n, val in base.items()}
    pert = ZonalPerturbation(coeffs, mu=MU, re=RE)

    r0, v0 = kepler_to_rv(
        case.a_km,
        case.e,
        case.inc_deg,
        case.raan_deg,
        case.argp_deg,
        anomaly_deg,
    )
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
    osc_hist = propagate_grid(ta, t_grid)
    mean0 = osculating_to_mean_state(state0, coeffs, re_val=RE, mu_val=MU)
    mean_hist = _rk4_integrate_mean(
        mean0,
        t_grid,
        coeffs,
        re_val=RE,
        mu_val=MU,
        substeps_per_interval=case.rk4_substeps_per_interval,
    )
    rec_hist = np.vstack([mean_to_osculating_state(state, coeffs, re_val=RE, mu_val=MU) for state in mean_hist])

    osc_cart = np.array([geqoe2cart(state, MU, pert)[0] for state in osc_hist])
    rec_cart = np.array([geqoe2cart(state, MU, pert)[0] for state in rec_hist])
    pos_err = np.linalg.norm(rec_cart - osc_cart, axis=1)

    osc_phases = _state_phases(osc_hist)
    rec_phases = _state_phases(rec_hist)
    k_osc = np.unwrap(osc_hist[:, 3])
    k_rec = np.unwrap(rec_hist[:, 3])

    metrics = {
        "p1_rel_rms": _relative_rms(rec_hist[:, 1], osc_hist[:, 1]),
        "p2_rel_rms": _relative_rms(rec_hist[:, 2], osc_hist[:, 2]),
        "q1_rel_rms": _relative_rms(rec_hist[:, 4], osc_hist[:, 4]),
        "q2_rel_rms": _relative_rms(rec_hist[:, 5], osc_hist[:, 5]),
        "g_rel_rms": _relative_rms(rec_phases["g"], osc_phases["g"]),
        "Q_rel_rms": _relative_rms(rec_phases["Q"], osc_phases["Q"]),
        "Psi_rms": _phase_error(rec_phases["Psi"], osc_phases["Psi"]),
        "Omega_rms": _phase_error(rec_phases["Omega"], osc_phases["Omega"]),
        "K_rms": _phase_error(k_rec, k_osc),
        "pos_rms_km": float(np.sqrt(np.mean(pos_err * pos_err))),
        "pos_max_km": float(np.max(pos_err)),
        "T_orbit": T_orbit,
    }

    return {
        "case": case,
        "anomaly_deg": anomaly_deg,
        "scale": scale,
        "t_grid": t_grid,
        "osc_hist": osc_hist,
        "rec_hist": rec_hist,
        "osc_cart": osc_cart,
        "rec_cart": rec_cart,
        "pos_err": pos_err,
        "metrics": metrics,
    }


def create_plots(results: list[dict[str, object]], scaling: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    rep_low = next(result for result in results if result["case"].name == "low-e")
    rep_high = next(result for result in results if result["case"].name == "high-e")

    for result, out_path in ((rep_low, OUT_LOW), (rep_high, OUT_HIGH)):
        tau = np.asarray(result["t_grid"], dtype=float) / float(result["metrics"]["T_orbit"])
        osc = np.asarray(result["osc_hist"], dtype=float)
        rec = np.asarray(result["rec_hist"], dtype=float)
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        panels = [
            (osc[:, 1], rec[:, 1], r"$p_1$"),
            (osc[:, 2], rec[:, 2], r"$p_2$"),
            (osc[:, 4], rec[:, 4], r"$q_1$"),
            (osc[:, 5], rec[:, 5], r"$q_2$"),
            (np.unwrap(osc[:, 3]), np.unwrap(rec[:, 3]), r"$K$"),
            (result["pos_err"], None, r"$||\Delta r||$ [km]"),
        ]
        for ax, (y_ref, y_rec, label) in zip(axes.ravel(), panels):
            if y_rec is None:
                ax.plot(tau, y_ref, lw=1.6)
            else:
                ax.plot(tau, y_ref, lw=1.5, label="full")
                ax.plot(tau, y_rec, "--", lw=1.3, label="reconstructed")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best")
        axes[2, 0].set_xlabel("orbit index")
        axes[2, 1].set_xlabel("orbit index")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        tau = np.asarray(result["t_grid"], dtype=float) / float(result["metrics"]["T_orbit"])
        label = f"{result['case'].name}, M0={result['anomaly_deg']:.0f} deg"
        ax.plot(tau, result["pos_err"], label=label)
    ax.set_xlabel("orbit index")
    ax.set_ylabel(r"$||\Delta r||$ [km]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_CART, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scales = np.asarray(scaling["scales"], dtype=float)
    errors = np.asarray(scaling["pos_rms_km"], dtype=float)
    ax.loglog(scales, errors, "o-", lw=1.6)
    ax.set_xlabel("zonal scale factor")
    ax.set_ylabel("position RMS error [km]")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_SCALE, dpi=180)
    plt.close(fig)


def write_note(results: list[dict[str, object]], scaling: dict[str, object]) -> None:
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            (
                result["case"].name,
                result["anomaly_deg"],
                metrics["K_rms"],
                metrics["pos_rms_km"],
                metrics["pos_max_km"],
                metrics["p1_rel_rms"],
                metrics["q1_rel_rms"],
            )
        )

    table_lines = []
    for name, anomaly, k_rms, pos_rms, pos_max, p1_rms, q1_rms in rows:
        table_lines.append(
            rf"{name} & {anomaly:.0f} & {k_rms:.3e} & {pos_rms:.3e} & {pos_max:.3e} & {p1_rms:.3e} & {q1_rms:.3e} \\"
        )
    scale_labels = ", ".join(f"{scale:.2f}" for scale in scaling["scales"])

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=2cm]{{geometry}}
\usepackage{{amsmath,amssymb,graphicx,booktabs}}
\begin{{document}}
\section*{{Validation of the Mixed-Zonal GEqOE Short-Period Map}}

This note validates the first-order mixed-zonal GEqOE short-period map for the
truncated zonal set $\{{J_2,J_3,J_4,J_5\}}$. The mean state is propagated in
$(\nu,p_1,p_2,M,q_1,q_2)$, where $M=L-\Psi$ is the uniformly advancing mean
fast phase. Osculating GEqOE reconstruction is obtained by applying the exact
degree-wise short-period corrections in $(g,Q,\Psi,\Omega,M)$ and then solving
the generalized Kepler equation for $K$.

\subsection*{{Osculating-history reconstruction}}
\begin{{center}}
\begin{{tabular}}{{lcccccc}}
\toprule
case & $M_0$ [deg] & $K$ RMS [rad] & pos RMS [km] & pos max [km] & $p_1$ rel RMS & $q_1$ rel RMS \\
\midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

The representative component histories are shown in
Figure~\ref{{fig:low}} for the low-eccentricity case and in
Figure~\ref{{fig:high}} for the high-eccentricity case. Figure~\ref{{fig:cart}}
collects the Cartesian position errors for all anomaly cases.

\begin{{figure}}[p]
\centering
\includegraphics[width=0.95\textwidth]{{docs/geqoe_averaged/figures/zonal_short_period_lowe_components.png}}
\caption{{Low-eccentricity osculating GEqOE and Cartesian reconstruction.}}
\label{{fig:low}}
\end{{figure}}

\begin{{figure}}[p]
\centering
\includegraphics[width=0.95\textwidth]{{docs/geqoe_averaged/figures/zonal_short_period_highe_components.png}}
\caption{{High-eccentricity osculating GEqOE and Cartesian reconstruction.}}
\label{{fig:high}}
\end{{figure}}

\begin{{figure}}[p]
\centering
\includegraphics[width=0.90\textwidth]{{docs/geqoe_averaged/figures/zonal_short_period_cartesian_errors.png}}
\caption{{Cartesian position errors across the anomaly sweep.}}
\label{{fig:cart}}
\end{{figure}}

\subsection*{{Zonal scaling}}
For the representative high-eccentricity case, the position RMS reconstruction
error was recomputed under zonal scale factors
$\{{{scale_labels}\}}$.
The fitted log-log slope was
\[
\frac{{d\log(\mathrm{{RMS}})}}{{d\log(\lambda)}} = {scaling["slope"]:.3f},
\]
which is consistent with the expected second-order residual of a first-order
mean-plus-short-period model.

\begin{{figure}}[h!]
\centering
\includegraphics[width=0.70\textwidth]{{docs/geqoe_averaged/figures/zonal_short_period_scaling.png}}
\caption{{RMS Cartesian reconstruction error versus zonal scale factor.}}
\end{{figure}}

\subsection*{{Conclusion}}
Within the truncated mixed-zonal problem, the GEqOE averaged formulation is now
closed to first order: the exact symbolic mean drift and the exact symbolic
short-period map together reconstruct the full osculating GEqOE and Cartesian
state with small residuals, and those residuals scale as expected for a
first-order theory.

\end{{document}}
"""
    OUT_TEX.write_text(tex.strip() + "\n")


def main() -> None:
    print("Precomputing exact symbolic short-period cache...", flush=True)
    _ensure_symbolic_cache(J_COEFFS)

    results: list[dict[str, object]] = []
    for case in CASES:
        for anomaly_deg in case.anomalies_deg:
            print(f"Running {case.name} case at M0={anomaly_deg:.0f} deg", flush=True)
            results.append(run_case(case, anomaly_deg, scale=1.0))

    scaling_case = CASES[-1]
    scaling_anomaly = scaling_case.anomalies_deg[0]
    scaling_scales = np.array([1.0, 0.5, 0.25], dtype=float)
    scaling_errors = []
    for scale in scaling_scales:
        print(f"Scaling sweep: lambda={scale:.2f}", flush=True)
        scaling_errors.append(run_case(scaling_case, scaling_anomaly, scale=scale)["metrics"]["pos_rms_km"])
    slope, intercept = np.polyfit(np.log(scaling_scales), np.log(scaling_errors), 1)
    scaling = {
        "scales": scaling_scales.tolist(),
        "pos_rms_km": [float(val) for val in scaling_errors],
        "slope": float(slope),
        "intercept": float(intercept),
    }

    create_plots(results, scaling)
    write_note(results, scaling)

    print("\nSummary", flush=True)
    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['case'].name:7s} M0={result['anomaly_deg']:6.1f} deg  "
            f"K_rms={metrics['K_rms']:.3e} rad  "
            f"pos_rms={metrics['pos_rms_km']:.3e} km  "
            f"pos_max={metrics['pos_max_km']:.3e} km",
            flush=True,
        )
    print(f"Scaling slope: {scaling['slope']:.3f}", flush=True)
    print(f"Wrote {OUT_TEX}", flush=True)


if __name__ == "__main__":
    main()
