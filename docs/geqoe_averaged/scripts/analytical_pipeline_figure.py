#!/usr/bin/env python
"""Generate the orbital element decomposition figure.

Shows how osculating Keplerian elements oscillate rapidly while the
analytical mean elements are smooth/constant — demonstrating what
the averaging transformation accomplishes.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/analytical_pipeline_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

FIG_DIR = DOC_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

from astrodyn_core.geqoe_taylor import (
    MU, RE,
    ZonalPerturbation,
    cart2geqoe,
)
from astrodyn_core.geqoe_taylor.cowell import (
    _build_cowell_heyoka_general_system,
    _build_par_values,
)

from geqoe_mean.constants import J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv, rv_to_classical
from geqoe_mean.short_period import (
    isolated_short_period_expressions_for,
    osculating_to_mean_state,
)
from geqoe_mean.validation import (
    ensure_symbolic_cache,
    rk4_integrate_mean,
)


def compute_data():
    """Compute osculating Keplerian elements (from Cowell truth) and mean elements."""
    a_km, ecc, inc_deg = 6878.0, 0.001, 51.6
    r0, v0 = kepler_to_rv(a_km, ecc, inc_deg, 30.0, 45.0, 0.0)
    pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)

    T_orb = 2 * np.pi * np.sqrt(a_km**3 / MU)
    n_orbits = 5
    t_grid = np.linspace(0, n_orbits * T_orb, 600)
    t_hours = t_grid / 3600.0

    # --- 1. Cowell truth → osculating classical Keplerian elements ---
    print("  Cowell truth (positions + velocities)...")
    import heyoka as hy
    sys_cow, _, par_map = _build_cowell_heyoka_general_system(
        pert, mu_val=MU, use_par=True, time_origin=0.0)
    ta = hy.taylor_adaptive(
        sys_cow, list(r0) + list(v0),
        tol=1e-15, compact_mode=True,
        pars=_build_par_values(pert, par_map),
    )

    n_pts = len(t_grid)
    a_osc = np.empty(n_pts)
    e_osc = np.empty(n_pts)
    i_osc = np.empty(n_pts)

    a_osc[0], e_osc[0], i_osc[0] = rv_to_classical(r0, v0)
    for k in range(1, n_pts):
        ta.propagate_until(t_grid[k])
        a_osc[k], e_osc[k], i_osc[k] = rv_to_classical(
            ta.state[:3], ta.state[3:6])

    # --- 2. Mean GEqOE → approximate mean classical elements ---
    print("  Mean GEqOE propagation...")
    ensure_symbolic_cache(J_COEFFS)

    state0_osc = cart2geqoe(r0, v0, MU, pert)
    mean0 = osculating_to_mean_state(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
    mean = rk4_integrate_mean(mean0, t_grid, J_COEFFS)

    # Mean classical approximations:
    # a_mean ≈ (mu / nu_mean^2)^(1/3) — nearly constant
    # e_mean ≈ g_mean = ||(p1, p2)|| — slowly varying
    # i_mean ≈ 2*arctan(Q_mean) — slowly varying
    a_mean = (MU / mean[:, 0] ** 2) ** (1.0 / 3.0)
    e_mean = np.sqrt(mean[:, 1] ** 2 + mean[:, 2] ** 2)
    i_mean = 2.0 * np.degrees(np.arctan(
        np.sqrt(mean[:, 4] ** 2 + mean[:, 5] ** 2)))

    return {
        "t_hours": t_hours,
        "a_osc": a_osc, "a_mean": a_mean,
        "e_osc": e_osc, "e_mean": e_mean,
        "i_osc": i_osc, "i_mean": i_mean,
        "T_orb_min": T_orb / 60.0,
    }


def make_figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = d["t_hours"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True,
                             gridspec_kw={"hspace": 0.10})

    C_OSC = "#2563EB"
    C_MEAN = "#EA580C"
    C_FILL = "#BFDBFE"

    for ax in axes:
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Panel 1: Semi-major axis ---
    ax = axes[0]
    ax.plot(t, d["a_osc"], color=C_OSC, lw=0.7, label="Osculating")
    ax.plot(t, d["a_mean"], color=C_MEAN, lw=1.4, ls="--", label="Mean")
    ax.fill_between(t, d["a_mean"], d["a_osc"], alpha=0.15, color=C_FILL)
    ax.set_ylabel("Semi-major axis [km]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95,
              edgecolor="#D1D5DB")

    # --- Panel 2: Eccentricity ---
    ax = axes[1]
    ax.plot(t, d["e_osc"], color=C_OSC, lw=0.7)
    ax.plot(t, d["e_mean"], color=C_MEAN, lw=1.4, ls="--")
    ax.fill_between(t, d["e_mean"], d["e_osc"], alpha=0.15, color=C_FILL)
    ax.set_ylabel("Eccentricity", fontsize=10)

    # --- Panel 3: Inclination ---
    ax = axes[2]
    ax.plot(t, d["i_osc"], color=C_OSC, lw=0.7)
    ax.plot(t, d["i_mean"], color=C_MEAN, lw=1.4, ls="--")
    ax.fill_between(t, d["i_mean"], d["i_osc"], alpha=0.15, color=C_FILL)
    ax.set_ylabel("Inclination [deg]", fontsize=10)
    ax.set_xlabel("Time [hours]", fontsize=10)

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.09)

    out = FIG_DIR / "analytical_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("Computing orbital element decomposition...")
    data = compute_data()

    da = data["a_osc"] - data["a_mean"]
    de = data["e_osc"] - data["e_mean"]
    di = data["i_osc"] - data["i_mean"]
    print(f"  Semi-major axis SP oscillation: "
          f"{da.max():+.2f} / {da.min():+.2f} km")
    print(f"  Eccentricity SP oscillation:    "
          f"{de.max():+.6f} / {de.min():+.6f}")
    print(f"  Inclination SP oscillation:     "
          f"{di.max():+.4f} / {di.min():+.4f} deg")

    print("Generating figure...")
    make_figure(data)
    print("Done.")


if __name__ == "__main__":
    main()
