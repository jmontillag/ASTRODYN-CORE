#!/usr/bin/env python
"""Full J2-J4 (or J2-J5 if available) validation.

Runs all the validation checks from the user's requirements:
- Inverse map round-trip
- Full osculating GEqOE reconstruction
- Cartesian position error
- Multiple anomalies
- Low-e and high-e orbits
- Zonal scaling
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DOC_DIR = Path(__file__).resolve().parents[1]
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

# Script-level imports (sys.path needed for zonal_short_period_validation)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from zonal_short_period_validation import run_case, ValidationCase

from geqoe_mean.constants import J2, J3, J4, J5, J_COEFFS, MU, RE
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    _generated_short_period_expressions,
    isolated_short_period_expressions_for,
    osculating_to_mean_state,
    mean_to_osculating_state,
)
from astrodyn_core.geqoe_taylor import (
    ZonalPerturbation,
    cart2geqoe, geqoe2cart,
)
import numpy as np

LOG = open('/tmp/full_validation.log', 'w')
def log(msg):
    LOG.write(msg + '\n')
    LOG.flush()

# Check which degrees have COMPLETE short-period data (all 5 variables)
available = {}
for n in (2, 3, 4, 5):
    try:
        all_ok = True
        for var in ('g', 'Q', 'Psi', 'Omega', 'M'):
            data = _generated_short_period_expressions(var, n)
            if data is None:
                all_ok = False
                break
        if all_ok:
            available[n] = {2: J2, 3: J3, 4: J4, 5: J5}[n]
    except Exception:
        pass

max_degree = max(available.keys())
j_coeffs = {n: v for n, v in available.items()}
log(f"Available degrees: {sorted(available.keys())} (max J{max_degree})")

# --- 1. Inverse map round-trip test ---
log("\n" + "=" * 70)
log("1. Inverse map round-trip (osc -> mean -> osc)")
log("=" * 70)

for label, a_km, e, inc_deg in [("low-e", 9000, 0.05, 40), ("high-e", 18000, 0.65, 63)]:
    r0, v0 = kepler_to_rv(a_km, e, inc_deg, 25.0, 60.0, 45.0)
    pert = ZonalPerturbation(j_coeffs, mu=MU, re=RE)
    state_osc = cart2geqoe(r0, v0, MU, pert)
    state_mean = osculating_to_mean_state(state_osc, j_coeffs, re_val=RE, mu_val=MU)
    state_rec = mean_to_osculating_state(state_mean, j_coeffs, re_val=RE, mu_val=MU)
    geqoe_err = np.max(np.abs(state_rec - state_osc))

    r_osc, v_osc = geqoe2cart(state_osc, MU, pert)
    r_rec, v_rec = geqoe2cart(state_rec, MU, pert)
    pos_err_m = np.linalg.norm(r_rec - r_osc) * 1000.0

    log(f"  {label}: GEqOE max err = {geqoe_err:.3e}, Cartesian = {pos_err_m:.1f} m")

# --- 2. Multi-anomaly propagation ---
log("\n" + "=" * 70)
log(f"2. Osculating-history reconstruction (J2-J{max_degree})")
log("=" * 70)

cases = [
    ValidationCase("low-e", 9000, 0.05, 40, 25, 60, (20, 80, 140, 220, 300), 10, 64),
    ValidationCase("high-e", 18000, 0.65, 63, 40, 250, (35, 90, 170, 260, 340), 8, 64),
]

results = []
for case in cases:
    for M0 in case.anomalies_deg:
        result = run_case(case, M0, scale=1.0, j_coeffs=j_coeffs)
        results.append(result)
        m = result["metrics"]
        log(f"  {case.name:7s} M0={M0:5.0f}  K_rms={m['K_rms']:.3e}  "
            f"pos_rms={m['pos_rms_km']:.3e} km  pos_max={m['pos_max_km']:.3e} km  "
            f"Psi={m['Psi_rms']:.3e}  Omega={m['Omega_rms']:.3e}")

# --- 3. Zonal scaling ---
log("\n" + "=" * 70)
log("3. Zonal scaling test")
log("=" * 70)

scales = [1.0, 0.5, 0.25, 0.125]
errs = []
for s in scales:
    r = run_case(cases[1], 35.0, scale=s, j_coeffs=j_coeffs)
    errs.append(r["metrics"]["pos_rms_km"])
    log(f"  scale={s:.3f}  pos_rms={r['metrics']['pos_rms_km']:.4e} km")

slope = np.polyfit(np.log(scales), np.log(errs), 1)[0]
log(f"  Log-log slope: {slope:.3f}")

# --- Summary ---
log("\n" + "=" * 70)
log("Summary")
log("=" * 70)

low_e_rms = [r["metrics"]["pos_rms_km"] for r in results if r["case"].name == "low-e"]
high_e_rms = [r["metrics"]["pos_rms_km"] for r in results if r["case"].name == "high-e"]
log(f"  Low-e pos_rms range:  {min(low_e_rms):.3f} - {max(low_e_rms):.3f} km")
log(f"  High-e pos_rms range: {min(high_e_rms):.3f} - {max(high_e_rms):.3f} km")
log(f"  Scaling slope: {slope:.3f}")
log(f"  Degrees used: J2-J{max_degree}")
log("DONE")
