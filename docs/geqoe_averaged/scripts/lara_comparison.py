#!/usr/bin/env python
"""Lara-Brouwer vs GEqOE head-to-head comparison.

Compares the first-order Brouwer/Lara analytical theory (implemented from
scratch in lara_theory/) against:
  1. Cowell truth (heyoka Cartesian, tol=1e-15)
  2. GEqOE mean + short-period reconstruction
  3. Orekit Brouwer-Lyddane

Run:
  cd docs/geqoe_averaged && conda run -n astrodyn-core-env python scripts/lara_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

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
    compute_position_errors as compute_errors,
    ensure_symbolic_cache as _ensure_symbolic_cache,
    rk4_integrate_mean,
)
from geqoe_mean.short_period import (
    mean_to_osculating_state_batch,
    osculating_to_mean_state,
)
from geqoe_mean.batch_conversions import geqoe2cart_zonal_batch

from lara_theory.propagator import LaraBrouwerPropagator

# Try importing Orekit-dependent functions from extended_validation
try:
    from scripts.extended_validation import (
        run_brouwer,
        _build_orekit_orbit,
        _init_orekit,
    )
    _HAS_OREKIT_SCRIPT = True
except ImportError:
    _HAS_OREKIT_SCRIPT = False


# --------------------------------------------------------------------------- #
#  Orbit cases (same as extended_validation.py)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrbitCase:
    name: str
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
    OrbitCase("LEO-circ",   6878,  0.001, 51.6,   30, 0,   0,   50),
    OrbitCase("LEO-mod-e",  7000,  0.05,  40,     25, 60,  90,  50),
    OrbitCase("SSO",        7078,  0.001, 97.8,   30, 0,   0,   50),
    OrbitCase("Near-equat", 7200,  0.01,  5,      0,  45,  90,  50),
    OrbitCase("Crit-low-e", 7500,  0.01,  63.435, 30, 90,  0,  100),
    OrbitCase("Crit-mod-e", 12000, 0.15,  63.435, 30, 90,  0,  100),
    OrbitCase("Molniya",    26554, 0.74,  63.4,   40, 270, 90,  30),
    OrbitCase("GTO",        24500, 0.73,  7,      0,  180, 0,   20),
    OrbitCase("MEO-GPS",    26560, 0.01,  55,     30, 0,   0,   50),
    OrbitCase("Polar",      7200,  0.005, 90,     30, 0,   0,   50),
    OrbitCase("HEO-45",     15000, 0.4,   45,     30, 120, 90,  30),
    OrbitCase("Retrograde", 7500,  0.01,  120,    30, 0,   0,   50),
]


# --------------------------------------------------------------------------- #
#  Cowell truth
# --------------------------------------------------------------------------- #

def build_cowell_integrator(pert, r0, v0, tol=1e-15):
    import heyoka as hy
    sys, _, par_map = _build_cowell_heyoka_general_system(
        pert, mu_val=MU, use_par=True, time_origin=0.0)
    ic = list(r0) + list(v0)
    par_values = _build_par_values(pert, par_map)
    ta = hy.taylor_adaptive(sys, ic, tol=tol, compact_mode=True, pars=par_values)
    return ta


def propagate_cowell_grid(ta, t_grid):
    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        ta.propagate_until(t)
        positions[i] = ta.state[:3]
    return positions


# --------------------------------------------------------------------------- #
#  GEqOE mean + SP
# --------------------------------------------------------------------------- #

def run_geqoe_meansp(case, r0, v0, pert, t_grid):
    state0_osc = cart2geqoe(r0, v0, MU, pert)
    mean0 = osculating_to_mean_state(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
    mean_hist = rk4_integrate_mean(mean0, t_grid, J_COEFFS, substeps=case.rk4_substeps)
    osc_rec = mean_to_osculating_state_batch(mean_hist, J_COEFFS, re_val=RE, mu_val=MU)
    positions, _ = geqoe2cart_zonal_batch(osc_rec, MU, pert)
    return positions


# --------------------------------------------------------------------------- #
#  Lara-Brouwer propagator
# --------------------------------------------------------------------------- #

def run_lara(r0, v0, t_grid):
    j_coeffs = {2: J2, 3: J3, 4: J4, 5: J5}
    prop = LaraBrouwerPropagator(MU, RE, j_coeffs)
    prop.initialize(r0, v0, 0.0)
    positions, _ = prop.propagate(t_grid)
    return positions


# --------------------------------------------------------------------------- #
#  Main comparison loop
# --------------------------------------------------------------------------- #

def run_single_case(case):
    print(f"\n{'='*60}")
    print(f"  {case.name}: a={case.a_km} km, e={case.e}, i={case.inc_deg} deg")
    print(f"{'='*60}")

    r0, v0 = kepler_to_rv(
        case.a_km, case.e, case.inc_deg,
        case.raan_deg, case.argp_deg, case.M0_deg)

    pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)
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
        result["cowell_time"] = time.time() - t0
        print(f"  Cowell:      {result['cowell_time']:.1f}s")
    except Exception as exc:
        print(f"  Cowell FAILED: {exc}")
        return result

    # --- 2. GEqOE mean + SP ---
    try:
        _ensure_symbolic_cache(J_COEFFS)
        t0 = time.time()
        geqoe_cart = run_geqoe_meansp(case, r0, v0, pert, t_grid)
        result["geqoe_time"] = time.time() - t0
        err_geqoe = compute_errors(truth_cart, geqoe_cart, "GEqOE")
        result["geqoe"] = err_geqoe
        print(f"  GEqOE:       {result['geqoe_time']:.1f}s  "
              f"pos RMS = {err_geqoe['pos_rms_km']:.4f} km")
    except Exception as exc:
        print(f"  GEqOE FAILED: {exc}")
        traceback.print_exc()
        result["geqoe"] = None

    # --- 3. Lara-Brouwer ---
    t0 = time.time()
    try:
        lara_cart = run_lara(r0, v0, t_grid)
        result["lara_time"] = time.time() - t0
        err_lara = compute_errors(truth_cart, lara_cart, "Lara-Brouwer")
        result["lara"] = err_lara
        print(f"  Lara:        {result['lara_time']:.1f}s  "
              f"pos RMS = {err_lara['pos_rms_km']:.4f} km")
    except Exception as exc:
        print(f"  Lara FAILED: {exc}")
        traceback.print_exc()
        result["lara"] = None

    # --- 4. Orekit Brouwer (cross-check) ---
    if _HAS_OREKIT_SCRIPT:
        t0 = time.time()
        try:
            orbit, epoch, frame = _build_orekit_orbit(case)
            if orbit is not None:
                brouwer_cart = run_brouwer(case, t_grid, orbit, epoch, frame)
                result["brouwer_time"] = time.time() - t0
                if brouwer_cart is not None:
                    err_brouwer = compute_errors(truth_cart, brouwer_cart, "Brouwer")
                    result["brouwer"] = err_brouwer
                    print(f"  Brouwer:     {result['brouwer_time']:.1f}s  "
                          f"pos RMS = {err_brouwer['pos_rms_km']:.4f} km")

                    # Lara vs Brouwer cross-check
                    if result.get("lara"):
                        lara_vs_brouwer = np.sqrt(np.mean(
                            np.sum((lara_cart - brouwer_cart)**2, axis=1)))
                        result["lara_vs_brouwer_rms_km"] = lara_vs_brouwer
                        print(f"  Lara-vs-Brouwer: {lara_vs_brouwer:.4f} km "
                              f"= {lara_vs_brouwer*1000:.1f} m")
                else:
                    result["brouwer"] = None
        except Exception as exc:
            print(f"  Brouwer FAILED: {exc}")
            result["brouwer"] = None

    return result


def main():
    print("=" * 60)
    print("  Lara-Brouwer vs GEqOE Comparison")
    print("=" * 60)

    results = []
    for case in CASES:
        result = run_single_case(case)
        results.append(result)

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Case':<14} | {'GEqOE RMS':>10} | {'Lara RMS':>10} | "
          f"{'Brouwer RMS':>12} | {'Lara-vs-Brw':>12} | {'Days':>6}")
    print(f"{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")

    for r in results:
        geqoe_s = f"{r['geqoe']['pos_rms_km']:.4f}" if r.get("geqoe") else "---"
        lara_s = f"{r['lara']['pos_rms_km']:.4f}" if r.get("lara") else "---"
        brouwer_s = f"{r['brouwer']['pos_rms_km']:.4f}" if r.get("brouwer") else "---"
        lvb_s = (f"{r['lara_vs_brouwer_rms_km']*1000:.1f} m"
                 if r.get("lara_vs_brouwer_rms_km") is not None else "---")
        days_s = f"{r.get('t_final_days', 0):.1f}"
        print(f"{r['case_name']:<14} | {geqoe_s:>10} | {lara_s:>10} | "
              f"{brouwer_s:>12} | {lvb_s:>12} | {days_s:>6}")

    # Save results
    out_path = DOC_DIR / "lara_comparison_results.json"
    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, dict):
                sr[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
            else:
                sr[k] = v
        serializable.append(sr)

    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
