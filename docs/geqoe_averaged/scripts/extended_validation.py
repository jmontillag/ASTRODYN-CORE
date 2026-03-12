#!/usr/bin/env python
"""Extended validation of GEqOE first-order averaged theory.

Tests across 12 orbital regimes with three comparison models:
  1. Cowell (Cartesian) heyoka — gold-standard truth (same J2-J5 potential)
  2. GEqOE mean + short-period reconstruction — theory under test
  3. Orekit Brouwer-Lyddane (J2-J5) — competing analytical theory

Produces summary LaTeX tables and figures for inclusion in the main note.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/extended_validation.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
FIG_DIR = DOC_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from astrodyn_core.geqoe_taylor import (
    MU, RE,
    ZonalPerturbation,
    build_state_integrator,
    cart2geqoe, geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid
from astrodyn_core.geqoe_taylor.cowell import (
    _build_cowell_heyoka_general_system,
    _build_par_values,
)

from geqoe_mean.constants import J2, J3, J4, J5, J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    evaluate_truncated_mean_rhs_pqm,
    isolated_short_period_expressions_for,
    mean_to_osculating_state,
    osculating_to_mean_state,
)
from geqoe_mean.validation import (
    compute_position_errors as compute_errors,
    ensure_symbolic_cache as _ensure_symbolic_cache,
    rk4_integrate_mean,
)

# --------------------------------------------------------------------------- #
#  Orbit cases
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrbitCase:
    name: str
    label: str
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    M0_deg: float
    n_orbits: int
    samples_per_orbit: int = 64
    rk4_substeps: int = 8


CASES = [
    OrbitCase("LEO-circ",   r"LEO circ.",        6878,  0.001, 51.6,   30, 0,   0,   50),
    OrbitCase("LEO-mod-e",  r"LEO mod-$e$",      7000,  0.05,  40,     25, 60,  90,  50),
    OrbitCase("SSO",        r"SSO",              7078,  0.001, 97.8,   30, 0,   0,   50),
    OrbitCase("Near-equat", r"Near-equat.",       7200,  0.01,  5,      0,  45,  90,  50),
    OrbitCase("Crit-low-e", r"Crit.\ low-$e$",   7500,  0.01,  63.435, 30, 90,  0,  100),
    OrbitCase("Crit-mod-e", r"Crit.\ mod-$e$",  12000,  0.15,  63.435, 30, 90,  0,  100),
    OrbitCase("Molniya",    r"Molniya",         26554,  0.74,  63.4,   40, 270, 90,  30),
    OrbitCase("GTO",        r"GTO",             24500,  0.73,  7,      0,  180, 0,   20),
    OrbitCase("MEO-GPS",    r"MEO/GPS",         26560,  0.01,  55,     30, 0,   0,   50),
    OrbitCase("Polar",      r"Polar",            7200,  0.005, 90,     30, 0,   0,   50),
    OrbitCase("HEO-45",     r"HEO mod-$e$",     15000,  0.4,   45,     30, 120, 90,  30),
    OrbitCase("Retrograde", r"Retrograde",       7500,  0.01,  120,    30, 0,   0,   50),
]


# --------------------------------------------------------------------------- #
#  Cowell heyoka grid propagation
# --------------------------------------------------------------------------- #

def build_cowell_integrator(pert, r0, v0, tol=1e-15):
    """Build a heyoka Cowell (Cartesian) integrator for the given perturbation."""
    import heyoka as hy
    sys, _, par_map = _build_cowell_heyoka_general_system(
        pert, mu_val=MU, use_par=True, time_origin=0.0)
    ic = list(r0) + list(v0)
    par_values = _build_par_values(pert, par_map)
    ta = hy.taylor_adaptive(sys, ic, tol=tol, compact_mode=True, pars=par_values)
    return ta


def propagate_cowell_grid(ta, t_grid):
    """Propagate Cowell integrator to a grid and return position array [N, 3]."""
    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        ta.propagate_until(t)
        positions[i] = ta.state[:3]
    return positions


# --------------------------------------------------------------------------- #
#  GEqOE mean + short-period reconstruction
# --------------------------------------------------------------------------- #

def run_geqoe_meansp(case, r0, v0, pert, t_grid):
    """Run GEqOE mean + short-period reconstruction. Return positions [N, 3]."""
    state0_osc = cart2geqoe(r0, v0, MU, pert)
    mean0 = osculating_to_mean_state(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
    mean_hist = rk4_integrate_mean(mean0, t_grid, J_COEFFS, substeps=case.rk4_substeps)
    osc_rec = np.array([
        mean_to_osculating_state(s, J_COEFFS, re_val=RE, mu_val=MU) for s in mean_hist])
    positions = np.array([geqoe2cart(s, MU, pert)[0] for s in osc_rec])
    return positions


# --------------------------------------------------------------------------- #
#  Orekit Brouwer-Lyddane
# --------------------------------------------------------------------------- #

_OREKIT_AVAILABLE = None

def _init_orekit():
    global _OREKIT_AVAILABLE
    if _OREKIT_AVAILABLE is not None:
        return _OREKIT_AVAILABLE
    try:
        import orekit
        orekit.initVM()
        from orekit.pyhelpers import setup_orekit_curdir
        import os
        orig = os.getcwd()
        os.chdir(str(REPO_ROOT))
        setup_orekit_curdir()
        os.chdir(orig)
        _OREKIT_AVAILABLE = True
    except Exception as exc:
        print(f"  [WARN] Orekit not available: {exc}")
        _OREKIT_AVAILABLE = False
    return _OREKIT_AVAILABLE


def run_brouwer(case, t_grid):
    """Run Orekit Brouwer-Lyddane propagator. Return positions [N, 3] in km."""
    if not _init_orekit():
        return None

    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.propagation.analytical import BrouwerLyddanePropagator
    from org.orekit.utils import Constants

    frame = FramesFactory.getEME2000()
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)

    # Orekit uses SI: meters, seconds, kg
    a_m = case.a_km * 1e3
    mu_m = MU * 1e9        # km³/s² → m³/s²
    re_m = RE * 1e3         # km → m

    orbit = KeplerianOrbit(
        float(a_m),
        float(case.e),
        float(np.deg2rad(case.inc_deg)),
        float(np.deg2rad(case.argp_deg)),
        float(np.deg2rad(case.raan_deg)),
        float(np.deg2rad(case.M0_deg)),
        PositionAngleType.MEAN,
        frame,
        epoch,
        float(mu_m),
    )

    # Unnormalized zonal coefficients: C_{n,0} = -J_n
    c20 = float(-J2)
    c30 = float(-J3)
    c40 = float(-J4)
    c50 = float(-J5)
    m2 = 0.0  # no drag

    try:
        prop = BrouwerLyddanePropagator(
            orbit, float(re_m), float(mu_m),
            c20, c30, c40, c50, m2)
    except Exception as exc:
        print(f"  [WARN] Brouwer failed for {case.name}: {exc}")
        return None

    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        try:
            state = prop.propagate(epoch.shiftedBy(float(t)))
            pv = state.getPVCoordinates(frame)
            pos = pv.getPosition()
            positions[i] = np.array([pos.getX(), pos.getY(), pos.getZ()]) / 1e3
        except Exception:
            positions[i] = np.nan
    return positions


# --------------------------------------------------------------------------- #
#  Main validation loop
# --------------------------------------------------------------------------- #

def run_single_case(case):
    """Run all propagation methods for one orbit case."""
    print(f"\n{'='*60}")
    print(f"  {case.name}: a={case.a_km} km, e={case.e}, i={case.inc_deg} deg")
    print(f"{'='*60}")

    r0, v0 = kepler_to_rv(
        case.a_km, case.e, case.inc_deg,
        case.raan_deg, case.argp_deg, case.M0_deg)

    pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)

    # Compute orbital period and time grid
    state0_geqoe = cart2geqoe(r0, v0, MU, pert)
    nu0 = float(state0_geqoe[0])
    T_orbit = 2.0 * np.pi / nu0
    t_grid = np.linspace(0, case.n_orbits * T_orbit,
                         case.n_orbits * case.samples_per_orbit + 1)

    result = {
        "case_name": case.name,
        "a_km": case.a_km,
        "e": case.e,
        "inc_deg": case.inc_deg,
        "T_orbit_s": T_orbit,
        "t_final_days": t_grid[-1] / 86400.0,
        "n_orbits": case.n_orbits,
    }

    # --- 1. Cowell truth ---
    t0 = time.time()
    try:
        ta_cow = build_cowell_integrator(pert, r0, v0)
        truth_cart = propagate_cowell_grid(ta_cow, t_grid)
        result["cowell_ok"] = True
        print(f"  Cowell:    {time.time()-t0:.1f}s")
    except Exception as exc:
        print(f"  Cowell FAILED: {exc}")
        result["cowell_ok"] = False
        return result

    # --- 2. GEqOE Taylor cross-check ---
    t0 = time.time()
    try:
        ta_geqoe, _ = build_state_integrator(pert, state0_geqoe, tol=1e-15,
                                              compact_mode=True)
        osc_geqoe = propagate_grid(ta_geqoe, t_grid)
        geqoe_cart = np.array([geqoe2cart(s, MU, pert)[0] for s in osc_geqoe])
        cross = compute_errors(truth_cart, geqoe_cart, "GEqOE-Taylor")
        result["geqoe_taylor_vs_cowell"] = cross
        print(f"  GEqOE-Taylor: {time.time()-t0:.1f}s  "
              f"(cross-check: pos RMS = {cross['pos_rms_km']:.2e} km)")
    except Exception as exc:
        print(f"  GEqOE-Taylor FAILED: {exc}")
        traceback.print_exc()
        result["geqoe_taylor_vs_cowell"] = None

    # --- 3. GEqOE mean + short-period ---
    t0 = time.time()
    try:
        _ensure_symbolic_cache(J_COEFFS)
        meansp_cart = run_geqoe_meansp(case, r0, v0, pert, t_grid)
        err_meansp = compute_errors(truth_cart, meansp_cart, "GEqOE-mean+SP")
        result["geqoe_meansp"] = err_meansp
        result["meansp_cart"] = meansp_cart
        print(f"  GEqOE-mean+SP: {time.time()-t0:.1f}s  "
              f"pos RMS = {err_meansp['pos_rms_km']:.4f} km  "
              f"rad RMS = {err_meansp['rad_rms_km']:.4f} km")
    except Exception as exc:
        print(f"  GEqOE-mean+SP FAILED: {exc}")
        traceback.print_exc()
        result["geqoe_meansp"] = None

    # --- 4. Brouwer-Lyddane ---
    t0 = time.time()
    try:
        brouwer_cart = run_brouwer(case, t_grid)
        if brouwer_cart is not None:
            err_brouwer = compute_errors(truth_cart, brouwer_cart, "Brouwer-Lyddane")
            result["brouwer"] = err_brouwer
            print(f"  Brouwer:   {time.time()-t0:.1f}s  "
                  f"pos RMS = {err_brouwer['pos_rms_km']:.4f} km  "
                  f"rad RMS = {err_brouwer['rad_rms_km']:.4f} km")
        else:
            result["brouwer"] = None
            print(f"  Brouwer:   skipped")
    except Exception as exc:
        print(f"  Brouwer FAILED: {exc}")
        traceback.print_exc()
        result["brouwer"] = None

    # Store position error time series for plots
    result["t_grid"] = t_grid
    result["truth_cart"] = truth_cart
    if result.get("geqoe_meansp"):
        diff_ms = meansp_cart - truth_cart
        result["pos_err_meansp"] = np.linalg.norm(diff_ms, axis=1)
        r_hat = truth_cart / np.linalg.norm(truth_cart, axis=1, keepdims=True)
        result["rad_err_meansp"] = np.sum(diff_ms * r_hat, axis=1)
    if result.get("brouwer") and brouwer_cart is not None:
        diff_br = brouwer_cart - truth_cart
        result["pos_err_brouwer"] = np.linalg.norm(diff_br, axis=1)
        r_hat = truth_cart / np.linalg.norm(truth_cart, axis=1, keepdims=True)
        result["rad_err_brouwer"] = np.sum(diff_br * r_hat, axis=1)

    return result


# --------------------------------------------------------------------------- #
#  Output: LaTeX table
# --------------------------------------------------------------------------- #

def write_latex_table(results, out_path):
    """Write summary LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Extended validation: position RMS error [km] against Cowell truth")
    lines.append(r"($J_2$--$J_5$ zonal, same constants). Propagation windows range from")
    lines.append(r"3 to 20 days depending on orbital period.}")
    lines.append(r"\label{tab:extended_validation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & $a$ & $e$ & $i$ & Days"
                 r" & \multicolumn{2}{c}{GEqOE mean+SP}"
                 r" & \multicolumn{2}{c}{Brouwer--Lyddane} \\")
    lines.append(r" & [km] & & [deg] & "
                 r" & pos & rad & pos & rad \\")
    lines.append(r"\midrule")

    for r in results:
        if not r.get("cowell_ok"):
            continue
        ms = r.get("geqoe_meansp")
        br = r.get("brouwer")
        ms_pos = f"{ms['pos_rms_km']:.3f}" if ms else "---"
        ms_rad = f"{ms['rad_rms_km']:.3f}" if ms else "---"
        br_pos = f"{br['pos_rms_km']:.3f}" if br else "---"
        br_rad = f"{br['rad_rms_km']:.3f}" if br else "---"

        case = next(c for c in CASES if c.name == r["case_name"])
        lines.append(
            f"{case.label} & {r['a_km']:.0f} & {r['e']:.3f} & "
            f"{r['inc_deg']:.1f} & {r['t_final_days']:.1f} & "
            f"{ms_pos} & {ms_rad} & {br_pos} & {br_rad} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  LaTeX table written to {out_path}")


# --------------------------------------------------------------------------- #
#  Output: figures
# --------------------------------------------------------------------------- #

def create_figures(results):
    """Create validation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Figure 1: Position error time series for all cases ---
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=False)
    axes_flat = axes.ravel()

    for idx, r in enumerate(results):
        if idx >= 12:
            break
        ax = axes_flat[idx]
        T = r.get("T_orbit_s", 1.0)
        case = next(c for c in CASES if c.name == r["case_name"])

        if "pos_err_meansp" in r:
            tau = r["t_grid"] / T
            ax.plot(tau, r["pos_err_meansp"], lw=0.8, label="GEqOE")
        if "pos_err_brouwer" in r:
            tau = r["t_grid"] / T
            ax.plot(tau, r["pos_err_brouwer"], lw=0.8, label="Brouwer", ls="--")

        ax.set_title(f"{case.name} (e={case.e})", fontsize=9)
        ax.set_ylabel("pos err [km]", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for ax in axes_flat:
        ax.set_xlabel("orbits", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "extended_validation_pos_errors.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  Figure saved: {out}")

    # --- Figure 2: Radial error time series for selected cases ---
    selected = ["LEO-circ", "Crit-low-e", "Crit-mod-e", "Molniya", "GTO", "Polar"]
    sel_results = [r for r in results if r["case_name"] in selected]
    if sel_results:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes_flat = axes.ravel()
        for idx, r in enumerate(sel_results[:6]):
            ax = axes_flat[idx]
            T = r.get("T_orbit_s", 1.0)
            case = next(c for c in CASES if c.name == r["case_name"])
            if "rad_err_meansp" in r:
                tau = r["t_grid"] / T
                ax.plot(tau, r["rad_err_meansp"], lw=0.8, label="GEqOE")
            if "rad_err_brouwer" in r:
                tau = r["t_grid"] / T
                ax.plot(tau, r["rad_err_brouwer"], lw=0.8, label="Brouwer", ls="--")
            ax.set_title(f"{case.name}", fontsize=9)
            ax.set_ylabel("radial err [km]", fontsize=8)
            ax.set_xlabel("orbits", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(fontsize=7)
        fig.tight_layout()
        out = FIG_DIR / "extended_validation_radial_errors.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  Figure saved: {out}")

    # --- Figure 3: Bar chart comparison ---
    names, geqoe_vals, brouwer_vals = [], [], []
    for r in results:
        ms = r.get("geqoe_meansp")
        br = r.get("brouwer")
        if ms and br:
            case = next(c for c in CASES if c.name == r["case_name"])
            names.append(case.name)
            geqoe_vals.append(ms["pos_rms_km"])
            brouwer_vals.append(br["pos_rms_km"])

    if names:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, geqoe_vals, w, label="GEqOE mean+SP")
        ax.bar(x + w/2, brouwer_vals, w, label="Brouwer-Lyddane")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Position RMS error [km]")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        out = FIG_DIR / "extended_validation_comparison_bar.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  Figure saved: {out}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("  Extended GEqOE Averaged Theory Validation")
    print("=" * 60)

    # Pre-build symbolic cache
    print("\nBuilding symbolic short-period cache...")
    _ensure_symbolic_cache(J_COEFFS)

    # Initialize Orekit
    print("Initializing Orekit...")
    _init_orekit()

    all_results = []
    for case in CASES:
        try:
            result = run_single_case(case)
            all_results.append(result)
        except Exception as exc:
            print(f"\n  CASE {case.name} FAILED COMPLETELY: {exc}")
            traceback.print_exc()
            all_results.append({"case_name": case.name, "cowell_ok": False})

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    fmt = "{:<14s} {:>8s} {:>8s} {:>10s} {:>10s}"
    print(fmt.format("Case", "e", "i [deg]", "GEqOE RMS", "Brouwer RMS"))
    print("-" * 54)
    for r in all_results:
        ms = r.get("geqoe_meansp")
        br = r.get("brouwer")
        ms_s = f"{ms['pos_rms_km']:.4f}" if ms else "FAIL"
        br_s = f"{br['pos_rms_km']:.4f}" if br else "FAIL"
        print(fmt.format(
            r["case_name"],
            f"{r.get('e','?')}",
            f"{r.get('inc_deg','?')}",
            ms_s, br_s))

    # Write outputs
    write_latex_table(all_results, DOC_DIR / "extended_validation_table.tex")

    print("\nGenerating figures...")
    create_figures(all_results)

    # Save raw metrics as JSON (for later use)
    json_out = DOC_DIR / "extended_validation_results.json"
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items()
              if not isinstance(v, np.ndarray)}
        serializable.append(sr)
    json_out.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n  JSON results: {json_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
