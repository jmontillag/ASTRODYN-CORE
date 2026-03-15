#!/usr/bin/env python
"""GEqOE short-period round-trip initialization error diagnostic.

Measures the O(J²) truncation error of the first-order direct (non-iterative)
osc→mean→osc inversion across the full orbital parameter space.

Produces a single 2×2 figure with four parameter-plane heatmaps.

Run:
  cd docs/geqoe_averaged && conda run -n astrodyn-core-env python scripts/roundtrip_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
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
    cart2geqoe, geqoe2cart,
)

from geqoe_mean.constants import J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.validation import ensure_symbolic_cache
from geqoe_mean.short_period import (
    osculating_to_mean_state_equinoctial,
    mean_to_osculating_state_equinoctial,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
R_EARTH = 6378.137  # WGS-84 equatorial radius [km]
PERIGEE_FLOOR = R_EARTH + 200.0  # km
PERT = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)


# ---------------------------------------------------------------------------
#  GEqOE round-trip: Cartesian → GEqOE osc → mean → osc → Cartesian
# ---------------------------------------------------------------------------
def _geqoe_roundtrip(r0, v0):
    """GEqOE osc→mean→osc round-trip. Returns position error [km]."""
    state_osc = cart2geqoe(r0, v0, MU, PERT)
    state_mean = osculating_to_mean_state_equinoctial(state_osc, J_COEFFS,
                                                       re_val=RE, mu_val=MU)
    state_rt = mean_to_osculating_state_equinoctial(state_mean, J_COEFFS,
                                                     re_val=RE, mu_val=MU)
    r_rt, _ = geqoe2cart(state_rt, MU, PERT)
    return np.linalg.norm(r_rt - r0)


# ---------------------------------------------------------------------------
#  Single grid-point evaluation
# ---------------------------------------------------------------------------
def _evaluate_point(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg):
    """Evaluate one grid point. Returns error in km or NaN."""
    if a_km * (1.0 - e) < PERIGEE_FLOOR:
        return np.nan

    try:
        r0, v0 = kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg)
        return _geqoe_roundtrip(r0, v0)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
#  Grid definitions
# ---------------------------------------------------------------------------
DEFAULTS = dict(a_km=7500, e=0.05, inc_deg=55.0, raan_deg=30.0,
                argp_deg=90.0, M0_deg=0.0)

GRIDS = [
    # (grid_func_name, ax1_vals, ax2_vals, xlabel, ylabel, param_map)
    ("e vs i",
     lambda: np.logspace(np.log10(0.001), np.log10(0.7), 30),
     lambda: np.linspace(5.0, 175.0, 30),
     "e", "i [deg]",
     lambda ax1, ax2: {"e": ax1, "inc_deg": ax2}),
    ("a vs i",
     lambda: np.logspace(np.log10(6600), np.log10(42000), 30),
     lambda: np.linspace(5.0, 175.0, 30),
     "a [km]", "i [deg]",
     lambda ax1, ax2: {"a_km": ax1, "inc_deg": ax2}),
    ("a vs e",
     lambda: np.logspace(np.log10(6600), np.log10(42000), 30),
     lambda: np.logspace(np.log10(0.001), np.log10(0.7), 30),
     "a [km]", "e",
     lambda ax1, ax2: {"a_km": ax1, "e": ax2}),
    (r"$M_0$ vs e",
     lambda: np.linspace(0.0, 350.0, 24),
     lambda: np.logspace(np.log10(0.001), np.log10(0.5), 25),
     r"$M_0$ [deg]", "e",
     lambda ax1, ax2: {"M0_deg": ax1, "e": ax2}),
]


def _run_grid(label, ax1_vals, ax2_vals, param_map_fn):
    """Run a 2D grid. Returns (n2, n1) error array in km."""
    n1, n2 = len(ax1_vals), len(ax2_vals)
    grid = np.full((n2, n1), np.nan)
    total = n1 * n2

    t0 = time.time()
    done = 0
    for j, ax2 in enumerate(ax2_vals):
        for i, ax1 in enumerate(ax1_vals):
            params = dict(DEFAULTS)
            params.update(param_map_fn(ax1, ax2))
            grid[j, i] = _evaluate_point(**params)

            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    [{done}/{total}] {elapsed:.0f}s, ~{eta:.0f}s left")

    return grid


# ---------------------------------------------------------------------------
#  Plotting — single 2×2 figure
# ---------------------------------------------------------------------------
def _plot_combined(results):
    """Plot all four grids in a single 2×2 figure with shared colorbar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Collect all valid log10(m) values for shared color limits
    all_log_m = []
    for _, _, _, _, grid_km in results:
        valid = grid_km[np.isfinite(grid_km)]
        if len(valid):
            all_log_m.append(np.log10(valid * 1e3))
    all_log_m = np.concatenate(all_log_m)
    vmin = np.nanpercentile(all_log_m, 1)
    vmax = np.nanpercentile(all_log_m, 99)

    mesh = None
    for idx, (label, xlabel, ylabel, (ax1_vals, ax2_vals), grid_km) in enumerate(results):
        ax = axes.flat[idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.log10(grid_km * 1e3)  # log10(metres)

        x_log = "km" in xlabel or xlabel == "e"
        y_log = ylabel == "e"

        mesh = ax.pcolormesh(ax1_vals, ax2_vals, data,
                             cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                             shading="nearest")
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=11)

        # Critical inclination lines
        if ylabel == "i [deg]":
            ax.axhline(63.435, color="white", ls="--", lw=1, alpha=0.7)
            ax.axhline(180 - 63.435, color="white", ls="--", lw=1, alpha=0.7)

        # Perigee constraint boundary
        if xlabel == "a [km]" and ylabel == "e":
            a_bnd = np.linspace(ax1_vals[0], ax1_vals[-1], 200)
            e_bnd = 1.0 - PERIGEE_FLOOR / a_bnd
            mask = (e_bnd > ax2_vals[0]) & (e_bnd < ax2_vals[-1])
            if np.any(mask):
                ax.plot(a_bnd[mask], e_bnd[mask], "k--", lw=1.5, alpha=0.6)
        elif xlabel == "e" and ylabel == "i [deg]":
            e_max = 1.0 - PERIGEE_FLOOR / DEFAULTS["a_km"]
            if e_max < ax1_vals[-1]:
                ax.axvline(e_max, color="k", ls="--", lw=1.5, alpha=0.6)

    # Shared colorbar
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.03, aspect=30)
    cbar.set_label(r"$\log_{10}$(round-trip position error [m])")

    fig.suptitle(
        "GEqOE Initialization Truncation Error: osc → mean → osc (J2–J5)",
        fontsize=13, y=0.98)

    out_path = FIG_DIR / "roundtrip_geqoe.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


# ---------------------------------------------------------------------------
#  JSON helper
# ---------------------------------------------------------------------------
def _to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 60)
    print("  GEqOE Round-Trip Initialization Error")
    print("=" * 60)

    t_start = time.time()

    print("\nPre-building symbolic short-period cache...")
    ensure_symbolic_cache(J_COEFFS)

    results = []  # [(label, xlabel, ylabel, (ax1, ax2), grid_km)]
    all_cache = {}

    for label, ax1_fn, ax2_fn, xlabel, ylabel, param_map_fn in GRIDS:
        ax1_vals = ax1_fn()
        ax2_vals = ax2_fn()
        n1, n2 = len(ax1_vals), len(ax2_vals)
        print(f"\n  {label} ({n1}x{n2} = {n1*n2} points)")
        grid_km = _run_grid(label, ax1_vals, ax2_vals, param_map_fn)

        valid = grid_km[np.isfinite(grid_km)]
        if len(valid):
            print(f"    median: {np.median(valid)*1e3:.3f} m, "
                  f"max: {np.max(valid)*1e3:.3f} m, "
                  f"failures: {np.sum(~np.isfinite(grid_km))}")

        results.append((label, xlabel, ylabel, (ax1_vals, ax2_vals), grid_km))
        all_cache[label] = {
            "ax1_vals": ax1_vals,
            "ax2_vals": ax2_vals,
            "geqoe_err_km": grid_km,
        }

    _plot_combined(results)

    # Save JSON
    cache_path = DOC_DIR / "roundtrip_geqoe_results.json"
    cache_path.write_text(
        json.dumps(_to_serializable(all_cache), indent=2),
        encoding="utf-8",
    )
    print(f"Results cached to {cache_path}")

    elapsed = time.time() - t_start
    print(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
