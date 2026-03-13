#!/usr/bin/env python
"""Arc-length convergence study: error vs number of orbits.

Confirms O(ε²t) secular error accumulation for the first-order theory.

Metric: position error at t = N·T_orbit (endpoint). At integer-orbit epochs
the fast phase returns to the same value, so the short-period contribution
is ~constant across all N and only the secular drift grows.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/arc_length_convergence.py
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
    mean_to_osculating_state,
    osculating_to_mean_state,
)
from geqoe_mean.validation import (
    ensure_symbolic_cache as _ensure_symbolic_cache,
    rk4_integrate_mean,
)

OUT_FIG = FIG_DIR / "arc_length_convergence.png"


@dataclass(frozen=True)
class ConvergenceCase:
    name: str
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    anomaly_deg: float
    samples_per_orbit: int = 64
    rk4_substeps: int = 8


CASES = (
    ConvergenceCase(
        name="low-e LEO",
        a_km=9000.0,
        e=0.05,
        inc_deg=40.0,
        raan_deg=25.0,
        argp_deg=60.0,
        anomaly_deg=20.0,
    ),
    ConvergenceCase(
        name="high-e HEO",
        a_km=18000.0,
        e=0.65,
        inc_deg=63.0,
        raan_deg=40.0,
        argp_deg=250.0,
        anomaly_deg=35.0,
    ),
)

N_ORBITS_LIST = [2, 5, 10, 20, 40, 80, 160]


def run_convergence_point(
    case: ConvergenceCase,
    n_orbits: int,
    j_coeffs: dict[int, float],
) -> dict[str, float]:
    """Run one case at a given arc length and return error metrics."""
    coeffs = j_coeffs
    pert = ZonalPerturbation(coeffs, mu=MU, re=RE)

    r0, v0 = kepler_to_rv(
        case.a_km, case.e, case.inc_deg,
        case.raan_deg, case.argp_deg, case.anomaly_deg,
    )
    state0 = cart2geqoe(r0, v0, MU, pert)
    nu0 = float(state0[0])
    T_orbit = 2.0 * np.pi / nu0
    t_grid = np.linspace(
        0.0,
        n_orbits * T_orbit,
        n_orbits * case.samples_per_orbit + 1,
        dtype=float,
    )

    ta, _ = build_state_integrator(pert, state0, tol=1.0e-15, compact_mode=True)
    osc_hist = propagate_grid(ta, t_grid)
    mean0 = osculating_to_mean_state(state0, coeffs, re_val=RE, mu_val=MU)
    mean_hist = rk4_integrate_mean(
        mean0, t_grid, coeffs,
        re_val=RE, mu_val=MU,
        substeps=case.rk4_substeps,
    )
    rec_hist = np.vstack([
        mean_to_osculating_state(state, coeffs, re_val=RE, mu_val=MU)
        for state in mean_hist
    ])

    osc_cart = np.array([geqoe2cart(state, MU, pert)[0] for state in osc_hist])
    rec_cart = np.array([geqoe2cart(state, MU, pert)[0] for state in rec_hist])
    pos_err = np.linalg.norm(rec_cart - osc_cart, axis=1)

    # Endpoint error: at t = N*T, fast phase returns to ~same value,
    # so SP contribution is ~constant; only secular drift varies with N.
    endpoint_err = float(pos_err[-1])

    # Per-orbit-boundary errors (at t = k*T for k = 1, ..., N)
    orbit_indices = np.arange(1, n_orbits + 1) * case.samples_per_orbit
    orbit_boundary_errs = pos_err[orbit_indices]
    last_orbit_mean_err = float(np.mean(
        pos_err[-case.samples_per_orbit:]
    ))

    return {
        "endpoint_km": endpoint_err,
        "rms_km": float(np.sqrt(np.mean(pos_err * pos_err))),
        "last_orbit_mean_km": last_orbit_mean_err,
    }


def create_plot(
    results: dict[str, list[dict[str, float]]],
    slopes: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    n_arr = np.array(N_ORBITS_LIST, dtype=float)

    for case in CASES:
        errors = np.array([r["endpoint_km"] for r in results[case.name]])
        label = f"{case.name} (slope={slopes[case.name]:.2f})"
        ax.loglog(n_arr, errors, "o-", lw=1.6, label=label)

    # Reference slope-1 line
    n_ref = np.array([N_ORBITS_LIST[0], N_ORBITS_LIST[-1]], dtype=float)
    e_min = min(r["endpoint_km"] for res in results.values() for r in res)
    ref_line = e_min * (n_ref / n_ref[0])
    ax.loglog(n_ref, ref_line, "k--", lw=0.8, alpha=0.5, label="slope = 1 reference")

    ax.set_xlabel("Number of orbital periods")
    ax.set_ylabel("Position error at endpoint [km]")
    ax.set_title("Arc-length convergence (endpoint metric)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


def main() -> None:
    print("Precomputing exact symbolic short-period cache...", flush=True)
    _ensure_symbolic_cache(J_COEFFS)

    results: dict[str, list[dict[str, float]]] = {}
    slopes: dict[str, float] = {}

    for case in CASES:
        case_results = []
        for n_orbits in N_ORBITS_LIST:
            print(f"  {case.name}: {n_orbits} orbits ...", end=" ", flush=True)
            metrics = run_convergence_point(case, n_orbits, J_COEFFS)
            case_results.append(metrics)
            print(
                f"endpoint = {metrics['endpoint_km']:.4e} km  "
                f"rms = {metrics['rms_km']:.4e} km",
                flush=True,
            )
        results[case.name] = case_results

        log_n = np.log(np.array(N_ORBITS_LIST, dtype=float))
        log_e = np.log(np.array([r["endpoint_km"] for r in case_results]))
        slope, _ = np.polyfit(log_n, log_e, 1)
        slopes[case.name] = float(slope)
        print(f"  {case.name}: fitted endpoint slope = {slope:.3f}", flush=True)

    create_plot(results, slopes)

    print("\nSummary:")
    for case in CASES:
        print(f"  {case.name}: endpoint slope = {slopes[case.name]:.3f}")


if __name__ == "__main__":
    main()
