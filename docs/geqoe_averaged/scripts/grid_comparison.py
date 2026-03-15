#!/usr/bin/env python
"""Grid-based error heatmap comparison: GEqOE vs Lara-Brouwer.

Produces:
  - Topex validation figure (time-series RSS, confirming Lara W₁ match)
  - 4 grid heatmap figures (e-vs-i, a-vs-i, a-vs-e, M₀-vs-e)
  - JSON results cache for re-plotting without re-running

Run:
  cd docs/geqoe_averaged && conda run -n astrodyn-core-env python scripts/grid_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
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
    cart2geqoe,
)
from astrodyn_core.geqoe_taylor.cowell import (
    _build_cowell_heyoka_general_system,
    _build_par_values,
)

from geqoe_mean.constants import J2, J3, J4, J5, J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.validation import (
    compute_position_errors,
    ensure_symbolic_cache,
    rk4_integrate_mean,
)
from geqoe_mean.short_period import (
    mean_to_osculating_state_equinoctial_batch,
    osculating_to_mean_state_equinoctial,
)
from geqoe_mean.batch_conversions import geqoe2cart_zonal_batch

from lara_theory.propagator import LaraBrouwerPropagator

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
R_EARTH = 6378.137  # WGS-84 equatorial radius [km]
PERIGEE_FLOOR = R_EARTH + 200.0  # km

# Cache the perturbation model and symbolic expressions once
PERT = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)


# ---------------------------------------------------------------------------
#  Cowell truth propagator
# ---------------------------------------------------------------------------
def _build_cowell(r0, v0, tol=1e-15):
    import heyoka as hy
    sys, _, par_map = _build_cowell_heyoka_general_system(
        PERT, mu_val=MU, use_par=True, time_origin=0.0)
    ic = list(r0) + list(v0)
    par_values = _build_par_values(PERT, par_map)
    return hy.taylor_adaptive(sys, ic, tol=tol, compact_mode=True, pars=par_values)


def _propagate_cowell(ta, t_grid):
    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        ta.propagate_until(t)
        positions[i] = ta.state[:3]
    return positions


# ---------------------------------------------------------------------------
#  GEqOE mean + SP propagation
# ---------------------------------------------------------------------------
def _propagate_geqoe(r0, v0, t_grid, substeps=8):
    state0_osc = cart2geqoe(r0, v0, MU, PERT)
    mean0 = osculating_to_mean_state_equinoctial(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
    mean_hist = rk4_integrate_mean(mean0, t_grid, J_COEFFS, substeps=substeps)
    osc_rec = mean_to_osculating_state_equinoctial_batch(mean_hist, J_COEFFS, re_val=RE, mu_val=MU)
    positions, _ = geqoe2cart_zonal_batch(osc_rec, MU, PERT)
    return positions


# ---------------------------------------------------------------------------
#  Lara-Brouwer propagation
# ---------------------------------------------------------------------------
def _propagate_lara(r0, v0, t_grid, j_coeffs=None, use_w1_sp=False):
    jc = j_coeffs or J_COEFFS
    prop = LaraBrouwerPropagator(MU, RE, jc, use_w1_sp=use_w1_sp)
    prop.initialize(r0, v0, 0.0)
    positions, _ = prop.propagate(t_grid)
    return positions


# ---------------------------------------------------------------------------
#  Single grid-point evaluation
# ---------------------------------------------------------------------------
def _evaluate_point(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg,
                    n_orbits=50, samples_per_orbit=64):
    """Evaluate one grid point. Returns (geqoe_rms_km, lara_rms_km) or NaN on failure."""
    # Perigee check
    if a_km * (1.0 - e) < PERIGEE_FLOOR:
        return np.nan, np.nan

    try:
        r0, v0 = kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg)
    except Exception:
        return np.nan, np.nan

    # Build time grid from orbital period
    try:
        state0 = cart2geqoe(r0, v0, MU, PERT)
        nu0 = float(state0[0])
        T_orbit = 2.0 * np.pi / nu0
        t_grid = np.linspace(0, n_orbits * T_orbit,
                             n_orbits * samples_per_orbit + 1)
    except Exception:
        return np.nan, np.nan

    # Cowell truth
    try:
        ta = _build_cowell(r0, v0)
        truth = _propagate_cowell(ta, t_grid)
    except Exception:
        return np.nan, np.nan

    # GEqOE
    geqoe_rms = np.nan
    try:
        geqoe_pos = _propagate_geqoe(r0, v0, t_grid)
        err = compute_position_errors(truth, geqoe_pos, "GEqOE")
        geqoe_rms = err["pos_rms_km"]
    except Exception:
        pass

    # Lara
    lara_rms = np.nan
    try:
        lara_pos = _propagate_lara(r0, v0, t_grid)
        err = compute_position_errors(truth, lara_pos, "Lara")
        lara_rms = err["pos_rms_km"]
    except Exception:
        pass

    return geqoe_rms, lara_rms


# ---------------------------------------------------------------------------
#  Grid definitions
# ---------------------------------------------------------------------------
def _make_grid_ecc_inc(n=30):
    """Eccentricity vs inclination grid."""
    e_vals = np.logspace(np.log10(0.001), np.log10(0.7), n)
    i_vals = np.linspace(5.0, 175.0, n)
    return e_vals, i_vals, "e", "i [deg]", "grid_ecc_inc"


def _make_grid_sma_inc(n=30):
    """Semi-major axis vs inclination grid."""
    a_vals = np.logspace(np.log10(6600), np.log10(42000), n)
    i_vals = np.linspace(5.0, 175.0, n)
    return a_vals, i_vals, "a [km]", "i [deg]", "grid_sma_inc"


def _make_grid_sma_ecc(n=30):
    """Semi-major axis vs eccentricity grid."""
    a_vals = np.logspace(np.log10(6600), np.log10(42000), n)
    e_vals = np.logspace(np.log10(0.001), np.log10(0.7), n)
    return a_vals, e_vals, "a [km]", "e", "grid_sma_ecc"


def _make_grid_M0_ecc(n_M=24, n_e=25):
    """Initial mean anomaly vs eccentricity grid."""
    M_vals = np.linspace(0.0, 350.0, n_M)
    e_vals = np.logspace(np.log10(0.001), np.log10(0.5), n_e)
    return M_vals, e_vals, r"$M_0$ [deg]", "e", "grid_M0_ecc"


# Defaults when parameter is not on axis
DEFAULTS = dict(a_km=7500, e=0.05, inc_deg=55.0, raan_deg=30.0,
                argp_deg=90.0, M0_deg=0.0)


def _run_grid(grid_func, progress_label=""):
    """Run a full 2D grid and return (ax1_vals, ax2_vals, geqoe_grid, lara_grid, name)."""
    grid_result = grid_func()
    ax1_vals, ax2_vals, xlabel, ylabel, name = grid_result
    n1, n2 = len(ax1_vals), len(ax2_vals)
    geqoe_grid = np.full((n2, n1), np.nan)
    lara_grid = np.full((n2, n1), np.nan)

    total = n1 * n2
    print(f"\n{'='*60}")
    print(f"  Grid: {name} ({n1}x{n2} = {total} points)")
    print(f"{'='*60}")

    t0 = time.time()
    done = 0
    for j, ax2 in enumerate(ax2_vals):
        for i, ax1 in enumerate(ax1_vals):
            # Map axis values to orbital parameters
            params = dict(DEFAULTS)
            if "ecc_inc" in name:
                params["e"] = ax1
                params["inc_deg"] = ax2
            elif "sma_inc" in name:
                params["a_km"] = ax1
                params["inc_deg"] = ax2
            elif "sma_ecc" in name:
                params["a_km"] = ax1
                params["e"] = ax2
            elif "M0_ecc" in name:
                params["M0_deg"] = ax1
                params["e"] = ax2

            g_rms, l_rms = _evaluate_point(**params)
            geqoe_grid[j, i] = g_rms
            lara_grid[j, i] = l_rms

            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  Grid {name} done in {elapsed:.0f}s")
    return ax1_vals, ax2_vals, geqoe_grid, lara_grid, xlabel, ylabel, name


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def _plot_heatmap_pair(ax1_vals, ax2_vals, geqoe_grid, lara_grid,
                       xlabel, ylabel, name):
    """Plot side-by-side heatmaps with shared colorbar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Take log10 of RMS values (handle NaN and zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_log = np.log10(geqoe_grid)
        l_log = np.log10(lara_grid)

    # Combine for shared color limits
    all_vals = np.concatenate([g_log[np.isfinite(g_log)],
                               l_log[np.isfinite(l_log)]])
    if len(all_vals) == 0:
        print(f"  WARNING: no valid data for {name}, skipping plot")
        return
    vmin = np.nanpercentile(all_vals, 2)
    vmax = np.nanpercentile(all_vals, 98)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Determine if axes should be log-scaled
    x_log = "km" in xlabel or xlabel == "e"
    y_log = ylabel == "e"

    for ax, data, title in [(ax1, g_log, "GEqOE mean+SP"),
                             (ax2, l_log, "Lara-Brouwer")]:
        mesh = ax.pcolormesh(ax1_vals, ax2_vals, data,
                             cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                             shading="nearest")
        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_title(title)

        # Critical inclination lines (only where inclination is on an axis)
        if ylabel == "i [deg]":
            ax.axhline(63.435, color="white", ls="--", lw=1, alpha=0.7)
            ax.axhline(180 - 63.435, color="white", ls="--", lw=1, alpha=0.7)
        if xlabel == "i [deg]":
            ax.axvline(63.435, color="white", ls="--", lw=1, alpha=0.7)
            ax.axvline(180 - 63.435, color="white", ls="--", lw=1, alpha=0.7)

        # Perigee constraint boundary
        if "sma_ecc" in name:
            # a(1-e) = PERIGEE_FLOOR => e = 1 - PERIGEE_FLOOR/a
            a_boundary = np.linspace(ax1_vals[0], ax1_vals[-1], 200)
            e_boundary = 1.0 - PERIGEE_FLOOR / a_boundary
            mask = (e_boundary > ax2_vals[0]) & (e_boundary < ax2_vals[-1])
            if np.any(mask):
                ax.plot(a_boundary[mask], e_boundary[mask],
                        "k--", lw=1.5, alpha=0.6, label="perigee floor")
        elif "ecc_inc" in name:
            # For e-vs-i grid: a fixed, so perigee = a*(1-e) > floor
            # => e < 1 - floor/a
            e_max = 1.0 - PERIGEE_FLOOR / DEFAULTS["a_km"]
            if e_max < ax1_vals[-1]:
                ax.axvline(e_max, color="k", ls="--", lw=1.5, alpha=0.6)

    ax1.set_ylabel(ylabel)

    cbar = fig.colorbar(mesh, ax=[ax1, ax2], shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log_{10}$(position RMS [km])")

    fig.suptitle("Position Error: 50 orbits, J2-J5", fontsize=13, y=1.02)

    out_path = FIG_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
#  Topex validation figure
# ---------------------------------------------------------------------------
def run_topex_validation():
    """Produce time-series validation plot for the Topex orbit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  Topex orbit validation")
    print("=" * 60)

    # Topex orbit parameters (Lara 2021, page 15)
    a_km = 7707.270
    e = 0.0001
    inc_deg = 66.04
    raan_deg = 180.001
    argp_deg = 270.0
    M0_deg = 180.0

    r0, v0 = kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg)

    # 30 days, 64 samples/orbit
    state0 = cart2geqoe(r0, v0, MU, PERT)
    nu0 = float(state0[0])
    T_orbit = 2.0 * np.pi / nu0
    n_days = 30
    n_orbits = int(n_days * 86400.0 / T_orbit) + 1
    t_grid = np.linspace(0, n_days * 86400.0, n_orbits * 64 + 1)

    # Cowell J2-only truth (for W₁ validation)
    j_coeffs_j2 = {2: J2}
    pert_j2 = ZonalPerturbation(j_coeffs_j2, mu=MU, re=RE)
    ta_j2 = _build_cowell.__wrapped__(r0, v0, pert_j2) if hasattr(_build_cowell, '__wrapped__') else None
    # Build J2-only Cowell manually
    import heyoka as hy
    sys_j2, _, par_map_j2 = _build_cowell_heyoka_general_system(
        pert_j2, mu_val=MU, use_par=True, time_origin=0.0)
    ic = list(r0) + list(v0)
    par_values_j2 = _build_par_values(pert_j2, par_map_j2)
    ta_j2 = hy.taylor_adaptive(sys_j2, ic, tol=1e-15, compact_mode=True, pars=par_values_j2)
    truth_j2 = _propagate_cowell(ta_j2, t_grid)

    # Cowell J2-J5 truth
    ta_full = _build_cowell(r0, v0)
    truth_full = _propagate_cowell(ta_full, t_grid)

    # Lara W₁ J2-only
    print("  Running Lara W₁ J2-only...")
    lara_w1_pos = _propagate_lara(r0, v0, t_grid, j_coeffs={2: J2},
                                  use_w1_sp=True)
    rss_w1 = np.sqrt(np.sum((truth_j2 - lara_w1_pos)**2, axis=1))

    # GEqOE J2-only (against J2-only truth — equal theoretical order)
    print("  Running GEqOE J2-only...")
    j_coeffs_j2 = {2: J2}
    pert_j2_geqoe = ZonalPerturbation(j_coeffs_j2, mu=MU, re=RE)
    ensure_symbolic_cache(j_coeffs_j2)
    state0_j2 = cart2geqoe(r0, v0, MU, pert_j2_geqoe)
    mean0_j2 = osculating_to_mean_state_equinoctial(state0_j2, j_coeffs_j2,
                                                     re_val=RE, mu_val=MU)
    mean_hist_j2 = rk4_integrate_mean(mean0_j2, t_grid, j_coeffs_j2,
                                       substeps=8, method="rk4")
    osc_rec_j2 = mean_to_osculating_state_equinoctial_batch(
        mean_hist_j2, j_coeffs_j2, re_val=RE, mu_val=MU)
    geqoe_j2_pos, _ = geqoe2cart_zonal_batch(osc_rec_j2, MU, pert_j2_geqoe)
    rss_geqoe_j2 = np.sqrt(np.sum((truth_j2 - geqoe_j2_pos)**2, axis=1))

    # Lara J2-J5 legacy
    print("  Running Lara J2-J5 legacy...")
    lara_legacy_pos = _propagate_lara(r0, v0, t_grid)
    rss_legacy = np.sqrt(np.sum((truth_full - lara_legacy_pos)**2, axis=1))

    # GEqOE J2-J5
    print("  Running GEqOE J2-J5...")
    geqoe_pos = _propagate_geqoe(r0, v0, t_grid)
    rss_geqoe = np.sqrt(np.sum((truth_full - geqoe_pos)**2, axis=1))

    t_days = t_grid / 86400.0

    # Print summary
    print(f"  Lara W₁ J2-only:   RSS(30d) = {rss_w1[-1]*1000:.1f} m")
    print(f"  GEqOE J2-only:     RSS(30d) = {rss_geqoe_j2[-1]*1000:.1f} m")
    print(f"  Lara J2-J5 legacy: RSS(30d) = {rss_legacy[-1]*1000:.1f} m")
    print(f"  GEqOE J2-J5:       RSS(30d) = {rss_geqoe[-1]*1000:.1f} m")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_days, rss_w1 * 1000, "b-", lw=1.5, alpha=0.9,
                label=f"Lara {{1+:2:1}} J2-only ({rss_w1[-1]*1000:.0f} m)")
    ax.semilogy(t_days, rss_geqoe_j2 * 1000, "c-", lw=1.5, alpha=0.9,
                label=f"GEqOE J2-only ({rss_geqoe_j2[-1]*1000:.0f} m)")
    ax.semilogy(t_days, rss_legacy * 1000, color="orange", lw=1.5, alpha=0.9,
                label=f"Lara J2-J5 ({rss_legacy[-1]*1000:.0f} m)")
    ax.semilogy(t_days, rss_geqoe * 1000, "g-", lw=1.5, alpha=0.9,
                label=f"GEqOE J2-J5 ({rss_geqoe[-1]*1000:.0f} m)")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("RSS position error [m]")
    ax.set_title("Topex orbit validation (a=7707 km, e=0.0001, i=66.04°)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_days)

    out_path = FIG_DIR / "topex_validation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {
        "lara_w1_rss_30d_m": float(rss_w1[-1] * 1000),
        "geqoe_j2_rss_30d_m": float(rss_geqoe_j2[-1] * 1000),
        "lara_legacy_rss_30d_m": float(rss_legacy[-1] * 1000),
        "geqoe_rss_30d_m": float(rss_geqoe[-1] * 1000),
    }


# ---------------------------------------------------------------------------
#  JSON serialization helper
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
    print("  Grid-Based Error Heatmap Comparison")
    print("  GEqOE mean+SP  vs  Lara-Brouwer")
    print("=" * 60)

    t_start = time.time()

    # Pre-build symbolic SP cache
    print("\nPre-building symbolic short-period cache...")
    ensure_symbolic_cache(J_COEFFS)

    all_results = {}

    # 1. Topex validation
    topex_result = run_topex_validation()
    all_results["topex"] = topex_result

    # 2. Grid heatmaps
    grids = [
        _make_grid_ecc_inc,
        _make_grid_sma_inc,
        _make_grid_sma_ecc,
        _make_grid_M0_ecc,
    ]

    for grid_func in grids:
        ax1_vals, ax2_vals, geqoe_grid, lara_grid, xlabel, ylabel, name = \
            _run_grid(grid_func)

        # Plot
        _plot_heatmap_pair(ax1_vals, ax2_vals, geqoe_grid, lara_grid,
                           xlabel, ylabel, name)

        # Cache results
        all_results[name] = {
            "ax1_vals": ax1_vals,
            "ax2_vals": ax2_vals,
            "geqoe_rms": geqoe_grid,
            "lara_rms": lara_grid,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }

        # Print quick summary
        g_valid = geqoe_grid[np.isfinite(geqoe_grid)]
        l_valid = lara_grid[np.isfinite(lara_grid)]
        if len(g_valid) and len(l_valid):
            geqoe_wins = np.sum(geqoe_grid < lara_grid)
            total_valid = np.sum(np.isfinite(geqoe_grid) & np.isfinite(lara_grid))
            n_fail = np.sum(~np.isfinite(geqoe_grid) | ~np.isfinite(lara_grid))
            print(f"  GEqOE wins: {geqoe_wins}/{total_valid} points "
                  f"({100*geqoe_wins/total_valid:.0f}%), "
                  f"{n_fail} failures")
            print(f"  GEqOE median RMS: {np.median(g_valid)*1000:.1f} m, "
                  f"Lara median RMS: {np.median(l_valid)*1000:.1f} m")

    # Save JSON cache
    cache_path = DOC_DIR / "grid_comparison_results.json"
    cache_path.write_text(
        json.dumps(_to_serializable(all_results), indent=2),
        encoding="utf-8",
    )
    print(f"\nResults cached to {cache_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
