#!/usr/bin/env python
"""Extended validation of GEqOE first-order averaged theory.

Tests across 12 orbital regimes with five comparison models:
  1. Cowell (Cartesian) heyoka — gold-standard truth (same J2-J5 potential)
  2. GEqOE mean + short-period reconstruction — theory under test
  3. GEqOE mean-only — isolates mean-drift accuracy
  4. Orekit Brouwer-Lyddane (J2-J5) — competing analytical theory
  5. Orekit DSST (zonal J2-J5) — competing semi-analytical theory
     (osculating, mean-only, and high-eccentricity-power variants)

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
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen

from geqoe_mean.constants import J2, J3, J4, J5, J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    evaluate_truncated_mean_rhs_pqm,
    isolated_short_period_expressions_for,
    mean_to_osculating_state,
    mean_to_osculating_state_batch,
    osculating_to_mean_state,
    osculating_to_mean_state_equinoctial,
    mean_to_osculating_state_equinoctial_batch,
)
from geqoe_mean.batch_conversions import geqoe2cart_zonal_batch
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
    """Run GEqOE mean + short-period reconstruction.

    Returns
    -------
    positions : (N, 3) array of osculating Cartesian positions [km]
    mean_hist : (N, 6) array of mean GEqOE states
    """
    state0_osc = cart2geqoe(r0, v0, MU, pert)
    mean0 = osculating_to_mean_state(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
    mean_hist = rk4_integrate_mean(mean0, t_grid, J_COEFFS, substeps=case.rk4_substeps)
    osc_rec = mean_to_osculating_state_batch(mean_hist, J_COEFFS, re_val=RE, mu_val=MU)
    positions, _ = geqoe2cart_zonal_batch(osc_rec, MU, pert)
    return positions, mean_hist


# --------------------------------------------------------------------------- #
#  Orekit initialization and shared helpers
# --------------------------------------------------------------------------- #

_OREKIT_AVAILABLE = None

def _init_orekit():
    global _OREKIT_AVAILABLE
    if _OREKIT_AVAILABLE is not None:
        return _OREKIT_AVAILABLE
    try:
        import orekit
        orekit.initVM()
        from org.orekit.data import DataContext, ZipJarCrawler
        import java
        zip_path = REPO_ROOT / "orekit-data.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"orekit-data.zip not found at {zip_path}")
        crawler = ZipJarCrawler(java.io.File(str(zip_path)))
        DataContext.getDefault().getDataProvidersManager().addProvider(crawler)
        _OREKIT_AVAILABLE = True
    except Exception as exc:
        print(f"  [WARN] Orekit not available: {exc}")
        _OREKIT_AVAILABLE = False
    return _OREKIT_AVAILABLE


def _build_orekit_orbit(case):
    """Build an Orekit KeplerianOrbit + epoch + frame from an OrbitCase.

    Returns (orbit, epoch, frame) or (None, None, None) if Orekit unavailable.
    """
    if not _init_orekit():
        return None, None, None

    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    frame = FramesFactory.getEME2000()
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)

    a_m = case.a_km * 1e3
    mu_m = MU * 1e9  # km^3/s^2 -> m^3/s^2

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
    return orbit, epoch, frame


# --------------------------------------------------------------------------- #
#  Orekit Brouwer-Lyddane
# --------------------------------------------------------------------------- #

def run_brouwer(case, t_grid, orbit=None, epoch=None, frame=None):
    """Run Orekit Brouwer-Lyddane propagator. Return positions [N, 3] in km."""
    if orbit is None:
        orbit, epoch, frame = _build_orekit_orbit(case)
    if orbit is None:
        return None

    from org.orekit.propagation.analytical import BrouwerLyddanePropagator

    re_m = RE * 1e3
    mu_m = MU * 1e9
    c20 = float(-J2)
    c30 = float(-J3)
    c40 = float(-J4)
    c50 = float(-J5)

    try:
        prop = BrouwerLyddanePropagator(
            orbit, float(re_m), float(mu_m),
            c20, c30, c40, c50, 0.0)
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
#  Custom zonal harmonics provider (matched constants)
# --------------------------------------------------------------------------- #

_CUSTOM_PROVIDER = None

def _get_matched_zonal_provider():
    """Return an UnnormalizedSphericalHarmonicsProvider using our exact J2-J5.

    This ensures DSST uses the same gravity constants as GEqOE, Cowell, and
    Brouwer, eliminating the ~0.1% EGM2008 mismatch.
    """
    global _CUSTOM_PROVIDER
    if _CUSTOM_PROVIDER is not None:
        return _CUSTOM_PROVIDER

    from org.orekit.forces.gravity.potential import (
        PythonUnnormalizedSphericalHarmonicsProvider,
        PythonUnnormalizedSphericalHarmonics,
        TideSystem,
    )
    from org.orekit.time import AbsoluteDate

    j_map = {2: J2, 3: J3, 4: J4, 5: J5}
    mu_m3s2 = MU * 1e9   # km^3/s^2 -> m^3/s^2
    re_m = RE * 1e3       # km -> m

    class _FixedZonalHarmonics(PythonUnnormalizedSphericalHarmonics):
        def __init__(self, date):
            super().__init__()
            self._date = date

        def getDate(self):
            return self._date

        def getUnnormalizedCnm(self, n, m):
            if m != 0:
                return 0.0
            return -j_map.get(n, 0.0)      # C_{n,0} = -J_n

        def getUnnormalizedSnm(self, n, m):
            return 0.0

    class _FixedZonalProvider(PythonUnnormalizedSphericalHarmonicsProvider):
        def getAe(self):             return re_m
        def getMu(self):             return mu_m3s2
        def getMaxDegree(self):      return 5
        def getMaxOrder(self):       return 0
        def getReferenceDate(self):  return AbsoluteDate.J2000_EPOCH
        def getTideSystem(self):     return TideSystem.UNKNOWN
        def onDate(self, date):      return _FixedZonalHarmonics(date)
        def getUnnormalizedC20(self, date):  return -j_map[2]

    _CUSTOM_PROVIDER = _FixedZonalProvider()
    return _CUSTOM_PROVIDER


# --------------------------------------------------------------------------- #
#  DSST propagation
# --------------------------------------------------------------------------- #

def _build_dsst_propagator(orbit, state_type="OSCULATING"):
    """Build a DSST (zonal-only J2-J5) propagator with matched constants.

    Uses our exact J2-J5 values (not Orekit EGM2008) so that DSST, GEqOE,
    Cowell, and Brouwer all share the same gravity model.  Includes the
    DSSTJ2SquaredClosedForm second-order correction (Zeis model).
    """
    from org.orekit.propagation.semianalytical.dsst.forces import (
        DSSTZonal, DSSTJ2SquaredClosedForm, ZeisModel,
    )
    from org.orekit.propagation.conversion import (
        DSSTPropagatorBuilder, DormandPrince853IntegratorBuilder,
    )
    from org.orekit.propagation import PropagationType

    provider = _get_matched_zonal_provider()

    dsst_zonal = DSSTZonal(provider)
    j2_sq = DSSTJ2SquaredClosedForm(ZeisModel(), provider)

    integrator_builder = DormandPrince853IntegratorBuilder(
        float(1e-3), float(300.0), float(1e-6))

    prop_type = PropagationType.MEAN
    state_type_enum = (PropagationType.OSCULATING if state_type == "OSCULATING"
                       else PropagationType.MEAN)

    builder = DSSTPropagatorBuilder(
        orbit, integrator_builder, float(10.0),
        prop_type, state_type_enum,
    )
    builder.setMass(1.0)
    builder.addForceModel(dsst_zonal)
    builder.addForceModel(j2_sq)

    return builder.buildPropagator(builder.getSelectedNormalizedParameters())


def _propagate_orekit_grid(propagator, epoch, frame, t_grid):
    """Propagate an Orekit propagator to a time grid, return positions [N, 3] km."""
    positions = np.empty((len(t_grid), 3))
    for i, t in enumerate(t_grid):
        state = propagator.propagate(epoch.shiftedBy(float(t)))
        pv = state.getPVCoordinates(frame)
        pos = pv.getPosition()
        positions[i] = np.array([pos.getX(), pos.getY(), pos.getZ()]) / 1e3
    return positions


def run_dsst_zonal_osc(orbit, epoch, frame, t_grid):
    """DSST zonal, osculating output — primary comparison."""
    prop = _build_dsst_propagator(orbit, state_type="OSCULATING")
    return _propagate_orekit_grid(prop, epoch, frame, t_grid)


def run_dsst_zonal_mean(orbit, epoch, frame, t_grid):
    """DSST zonal, mean-only output — isolates mean-drift accuracy."""
    prop = _build_dsst_propagator(orbit, state_type="MEAN")
    return _propagate_orekit_grid(prop, epoch, frame, t_grid)


def run_dsst_zonal_high_ecc(orbit, epoch, frame, t_grid, max_ecc_pow=6):
    """DSST zonal with elevated maxEccPowShortPeriodics for high-e orbits.

    Uses the same matched constants as _build_dsst_propagator, with explicit
    eccentricity power control to test convergence of DSST's truncated
    Fourier series.
    """
    from org.orekit.propagation.semianalytical.dsst.forces import (
        DSSTZonal, DSSTJ2SquaredClosedForm, ZeisModel,
    )
    from org.orekit.propagation.conversion import (
        DSSTPropagatorBuilder, DormandPrince853IntegratorBuilder,
    )
    from org.orekit.propagation import PropagationType

    provider = _get_matched_zonal_provider()

    dsst_zonal = DSSTZonal(
        provider,
        int(5),            # maxDegreeShortPeriodics
        int(max_ecc_pow),  # maxEccPowShortPeriodics
        int(11),           # maxFrequencyShortPeriodics = 2*5+1
    )
    j2_sq = DSSTJ2SquaredClosedForm(ZeisModel(), provider)

    integrator_builder = DormandPrince853IntegratorBuilder(
        float(1e-3), float(300.0), float(1e-6))

    builder = DSSTPropagatorBuilder(
        orbit,
        integrator_builder,
        float(10.0),  # positionScale
        PropagationType.MEAN,
        PropagationType.OSCULATING,
    )
    builder.setMass(1.0)
    builder.addForceModel(dsst_zonal)
    builder.addForceModel(j2_sq)

    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
    return _propagate_orekit_grid(propagator, epoch, frame, t_grid)


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

    # Initialize cart arrays to None for safe access in plotting section
    meansp_cart = None
    mean_hist = None
    geqoe_mean_cart = None
    brouwer_cart = None
    dsst_osc_cart = None
    dsst_mean_cart = None

    # --- 1. Cowell truth ---
    t0 = time.time()
    try:
        ta_cow = build_cowell_integrator(pert, r0, v0)
        truth_cart = propagate_cowell_grid(ta_cow, t_grid)
        result["cowell_ok"] = True
        result["cowell_time"] = time.time() - t0
        print(f"  Cowell:        {result['cowell_time']:.1f}s")
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
        geqoe_cart, _ = geqoe2cart_zonal_batch(osc_geqoe, MU, pert)
        cross = compute_errors(truth_cart, geqoe_cart, "GEqOE-Taylor")
        result["geqoe_taylor_vs_cowell"] = cross
        result["geqoe_taylor_time"] = time.time() - t0
        print(f"  GEqOE-Taylor:  {result['geqoe_taylor_time']:.1f}s  "
              f"(cross-check: pos RMS = {cross['pos_rms_km']:.2e} km)")
    except Exception as exc:
        print(f"  GEqOE-Taylor FAILED: {exc}")
        traceback.print_exc()
        result["geqoe_taylor_vs_cowell"] = None

    # --- 3. GEqOE mean + short-period ---
    try:
        _ensure_symbolic_cache(J_COEFFS)

        # Time the mean propagation separately for accurate cost breakdown
        t0_mean = time.time()
        state0_osc = cart2geqoe(r0, v0, MU, pert)
        mean0 = osculating_to_mean_state(state0_osc, J_COEFFS, re_val=RE, mu_val=MU)
        mean_hist = rk4_integrate_mean(mean0, t_grid, J_COEFFS, substeps=case.rk4_substeps)
        mean_prop_time = time.time() - t0_mean

        # Short-period reconstruction + geqoe2cart (vectorized)
        t0_sp = time.time()
        osc_rec = mean_to_osculating_state_batch(mean_hist, J_COEFFS, re_val=RE, mu_val=MU)
        meansp_cart, _ = geqoe2cart_zonal_batch(osc_rec, MU, pert)
        sp_time = time.time() - t0_sp

        result["meansp_time"] = mean_prop_time + sp_time
        err_meansp = compute_errors(truth_cart, meansp_cart, "GEqOE-mean+SP")
        result["geqoe_meansp"] = err_meansp
        result["meansp_cart"] = meansp_cart
        print(f"  GEqOE-mean+SP: {result['meansp_time']:.1f}s  "
              f"pos RMS = {err_meansp['pos_rms_km']:.4f} km  "
              f"rad RMS = {err_meansp['rad_rms_km']:.4f} km")

        # --- 3b. GEqOE mean-only (no short-period map) ---
        # mean_hist has [nu, p1, p2, M, q1, q2] where element [3] = M (mean
        # fast phase).  geqoe2cart_zonal_batch expects K (eccentric longitude).
        # Convert M -> L -> K via the generalized Kepler equation.
        t0_mo = time.time()
        mean_for_cart = mean_hist.copy()
        Psi_arr = np.arctan2(mean_hist[:, 1], mean_hist[:, 2])
        L_arr = Psi_arr + mean_hist[:, 3]          # L = Psi + M
        K_arr = solve_kepler_gen(L_arr, mean_hist[:, 1], mean_hist[:, 2])
        mean_for_cart[:, 3] = K_arr
        geqoe_mean_cart, _ = geqoe2cart_zonal_batch(mean_for_cart, MU, pert)
        mo_time = time.time() - t0_mo
        result["geqoe_mean_only_time"] = mean_prop_time + mo_time
        err_geqoe_mean = compute_errors(truth_cart, geqoe_mean_cart, "GEqOE-mean-only")
        result["geqoe_mean_only"] = err_geqoe_mean
        print(f"  GEqOE-mean:    "
              f"pos RMS = {err_geqoe_mean['pos_rms_km']:.4f} km")

    except Exception as exc:
        print(f"  GEqOE-mean+SP FAILED: {exc}")
        traceback.print_exc()
        result["geqoe_meansp"] = None
        result["geqoe_mean_only"] = None

    # --- 3c. GEqOE equinoctial mean + SP ---
    try:
        _ensure_symbolic_cache(J_COEFFS)
        state0_osc_eq = cart2geqoe(r0, v0, MU, pert)
        mean0_eq = osculating_to_mean_state_equinoctial(state0_osc_eq, J_COEFFS, re_val=RE, mu_val=MU)
        mean_hist_eq = rk4_integrate_mean(mean0_eq, t_grid, J_COEFFS, substeps=case.rk4_substeps)
        osc_rec_eq = mean_to_osculating_state_equinoctial_batch(mean_hist_eq, J_COEFFS, re_val=RE, mu_val=MU)
        eqnoc_cart, _ = geqoe2cart_zonal_batch(osc_rec_eq, MU, pert)
        err_eqnoc = compute_errors(truth_cart, eqnoc_cart, "GEqOE-eqnoc+SP")
        result["geqoe_eqnoc"] = err_eqnoc
        print(f"  GEqOE-eqnoc:   "
              f"pos RMS = {err_eqnoc['pos_rms_km']:.4f} km  "
              f"rad RMS = {err_eqnoc['rad_rms_km']:.4f} km")
    except Exception as exc:
        print(f"  GEqOE-eqnoc FAILED: {exc}")
        traceback.print_exc()
        result["geqoe_eqnoc"] = None

    # Build Orekit orbit once for Brouwer + DSST
    orekit_orbit, epoch_ok, frame_ok = _build_orekit_orbit(case)

    # --- 4. Brouwer-Lyddane ---
    t0 = time.time()
    try:
        brouwer_cart = run_brouwer(case, t_grid, orekit_orbit, epoch_ok, frame_ok)
        result["brouwer_time"] = time.time() - t0
        if brouwer_cart is not None:
            err_brouwer = compute_errors(truth_cart, brouwer_cart, "Brouwer-Lyddane")
            result["brouwer"] = err_brouwer
            print(f"  Brouwer:       {result['brouwer_time']:.1f}s  "
                  f"pos RMS = {err_brouwer['pos_rms_km']:.4f} km  "
                  f"rad RMS = {err_brouwer['rad_rms_km']:.4f} km")
        else:
            result["brouwer"] = None
            print(f"  Brouwer:       skipped")
    except Exception as exc:
        print(f"  Brouwer FAILED: {exc}")
        traceback.print_exc()
        result["brouwer"] = None

    # --- 5. DSST zonal (osculating) — primary comparison ---
    t0 = time.time()
    try:
        if orekit_orbit is not None:
            dsst_osc_cart = run_dsst_zonal_osc(orekit_orbit, epoch_ok, frame_ok, t_grid)
            result["dsst_osc_time"] = time.time() - t0
            err_dsst_osc = compute_errors(truth_cart, dsst_osc_cart, "DSST-osc")
            result["dsst_osc"] = err_dsst_osc
            print(f"  DSST-osc:      {result['dsst_osc_time']:.1f}s  "
                  f"pos RMS = {err_dsst_osc['pos_rms_km']:.4f} km  "
                  f"rad RMS = {err_dsst_osc['rad_rms_km']:.4f} km")
        else:
            result["dsst_osc"] = None
            print(f"  DSST-osc:      skipped (no Orekit)")
    except Exception as exc:
        print(f"  DSST-osc FAILED: {exc}")
        traceback.print_exc()
        result["dsst_osc"] = None

    # --- 6. DSST zonal (mean-only) — isolates mean-drift accuracy ---
    t0 = time.time()
    try:
        if orekit_orbit is not None:
            dsst_mean_cart = run_dsst_zonal_mean(orekit_orbit, epoch_ok, frame_ok, t_grid)
            result["dsst_mean_time"] = time.time() - t0
            err_dsst_mean = compute_errors(truth_cart, dsst_mean_cart, "DSST-mean")
            result["dsst_mean"] = err_dsst_mean
            print(f"  DSST-mean:     {result['dsst_mean_time']:.1f}s  "
                  f"pos RMS = {err_dsst_mean['pos_rms_km']:.4f} km")
        else:
            result["dsst_mean"] = None
    except Exception as exc:
        print(f"  DSST-mean FAILED: {exc}")
        traceback.print_exc()
        result["dsst_mean"] = None

    # --- 7. DSST high-eccentricity-power (only for e >= 0.35) ---
    if case.e >= 0.35 and orekit_orbit is not None:
        for max_ecc in [6, 8]:
            t0 = time.time()
            try:
                dsst_he_cart = run_dsst_zonal_high_ecc(
                    orekit_orbit, epoch_ok, frame_ok, t_grid, max_ecc_pow=max_ecc)
                result[f"dsst_osc_ecc{max_ecc}_time"] = time.time() - t0
                err_he = compute_errors(truth_cart, dsst_he_cart, f"DSST-osc-ecc{max_ecc}")
                result[f"dsst_osc_ecc{max_ecc}"] = err_he
                print(f"  DSST-ecc{max_ecc}:    {result[f'dsst_osc_ecc{max_ecc}_time']:.1f}s  "
                      f"pos RMS = {err_he['pos_rms_km']:.4f} km")
            except Exception as exc:
                print(f"  DSST-ecc{max_ecc} FAILED: {exc}")
                traceback.print_exc()
                result[f"dsst_osc_ecc{max_ecc}"] = None

    # Store position error time series for plots
    result["t_grid"] = t_grid
    result["truth_cart"] = truth_cart

    if result.get("geqoe_meansp") and meansp_cart is not None:
        diff_ms = meansp_cart - truth_cart
        result["pos_err_meansp"] = np.linalg.norm(diff_ms, axis=1)
        r_hat = truth_cart / np.linalg.norm(truth_cart, axis=1, keepdims=True)
        result["rad_err_meansp"] = np.sum(diff_ms * r_hat, axis=1)
    if result.get("brouwer") and brouwer_cart is not None:
        diff_br = brouwer_cart - truth_cart
        result["pos_err_brouwer"] = np.linalg.norm(diff_br, axis=1)
        r_hat = truth_cart / np.linalg.norm(truth_cart, axis=1, keepdims=True)
        result["rad_err_brouwer"] = np.sum(diff_br * r_hat, axis=1)
    if result.get("dsst_osc") and dsst_osc_cart is not None:
        diff_do = dsst_osc_cart - truth_cart
        result["pos_err_dsst_osc"] = np.linalg.norm(diff_do, axis=1)
        r_hat = truth_cart / np.linalg.norm(truth_cart, axis=1, keepdims=True)
        result["rad_err_dsst_osc"] = np.sum(diff_do * r_hat, axis=1)
    if result.get("dsst_mean") and dsst_mean_cart is not None:
        diff_dm = dsst_mean_cart - truth_cart
        result["pos_err_dsst_mean"] = np.linalg.norm(diff_dm, axis=1)
    if result.get("geqoe_mean_only") and geqoe_mean_cart is not None:
        diff_gm = geqoe_mean_cart - truth_cart
        result["pos_err_geqoe_mean_only"] = np.linalg.norm(diff_gm, axis=1)

    return result


# --------------------------------------------------------------------------- #
#  Output: LaTeX tables
# --------------------------------------------------------------------------- #

def write_osculating_table(results, out_path):
    """Table A: osculating comparison — GEqOE polar vs eqnoc vs DSST vs Brouwer."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Osculating position RMS error [km] against Cowell truth")
    lines.append(r"($J_2$--$J_5$ zonal, identical constants for all methods).")
    lines.append(r"``polar'': polar SP map; ``eqnoc'': equinoctial (singularity-free) SP map.}")
    lines.append(r"\label{tab:dsst_osculating}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrr" + "rr" * 4 + r"@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & $a$ & $e$ & $i$ & Days"
                 r" & \multicolumn{2}{c}{GEqOE polar}"
                 r" & \multicolumn{2}{c}{GEqOE eqnoc}"
                 r" & \multicolumn{2}{c}{DSST osc.}"
                 r" & \multicolumn{2}{c}{Brouwer--Lyddane} \\")
    lines.append(r" & [km] & & [deg] & "
                 r" & pos & rad & pos & rad & pos & rad & pos & rad \\")
    lines.append(r"\midrule")

    for r in results:
        if not r.get("cowell_ok"):
            continue
        ms = r.get("geqoe_meansp")
        eq = r.get("geqoe_eqnoc")
        do = r.get("dsst_osc")
        br = r.get("brouwer")
        case = next(c for c in CASES if c.name == r["case_name"])

        def _fmt(d, key):
            if d is None:
                return "---"
            v = d[key]
            if v < 0.01:
                return f"{v:.4f}"
            return f"{v:.3f}"

        lines.append(
            f"{case.label} & {r['a_km']:.0f} & {r['e']:.3f} & "
            f"{r['inc_deg']:.1f} & {r['t_final_days']:.1f} & "
            f"{_fmt(ms, 'pos_rms_km')} & {_fmt(ms, 'rad_rms_km')} & "
            f"{_fmt(eq, 'pos_rms_km')} & {_fmt(eq, 'rad_rms_km')} & "
            f"{_fmt(do, 'pos_rms_km')} & {_fmt(do, 'rad_rms_km')} & "
            f"{_fmt(br, 'pos_rms_km')} & {_fmt(br, 'rad_rms_km')} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table A (osculating) written to {out_path}")


def write_mean_drift_table(results, out_path):
    """Table B: mean-drift decomposition — GEqOE mean-only vs DSST mean-only."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean-drift position RMS [km]: mean-element propagation without")
    lines.append(r"short-period reconstruction, isolating secular/long-period accuracy.}")
    lines.append(r"\label{tab:dsst_mean_drift}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & $a$ [km] & $e$ & Days"
                 r" & GEqOE mean & DSST mean & Ratio \\")
    lines.append(r"\midrule")

    for r in results:
        if not r.get("cowell_ok"):
            continue
        gm = r.get("geqoe_mean_only")
        dm = r.get("dsst_mean")
        case = next(c for c in CASES if c.name == r["case_name"])
        gm_s = f"{gm['pos_rms_km']:.3f}" if gm else "---"
        dm_s = f"{dm['pos_rms_km']:.3f}" if dm else "---"
        if gm and dm and dm["pos_rms_km"] > 0:
            ratio = gm["pos_rms_km"] / dm["pos_rms_km"]
            ratio_s = f"{ratio:.1f}"
        else:
            ratio_s = "---"
        lines.append(
            f"{case.label} & {r['a_km']:.0f} & {r['e']:.3f} & "
            f"{r['t_final_days']:.1f} & {gm_s} & {dm_s} & {ratio_s} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table B (mean-drift) written to {out_path}")


def write_ecc_power_table(results, out_path):
    """Table C: eccentricity power convergence for high-e cases."""
    high_e = [r for r in results
              if r.get("cowell_ok") and r.get("e", 0) >= 0.35]
    if not high_e:
        return

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Eccentricity power convergence: position RMS [km] for")
    lines.append(r"high-eccentricity cases. DSST ecc$=N$ denotes")
    lines.append(r"\texttt{maxEccPowShortPeriodics}$=N$ (default 4).}")
    lines.append(r"\label{tab:dsst_ecc_power}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & $e$ & GEqOE & DSST ecc=4 & DSST ecc=6 & DSST ecc=8 \\")
    lines.append(r"\midrule")

    for r in high_e:
        case = next(c for c in CASES if c.name == r["case_name"])
        ms = r.get("geqoe_meansp")
        do = r.get("dsst_osc")  # default ecc=4
        e6 = r.get("dsst_osc_ecc6")
        e8 = r.get("dsst_osc_ecc8")

        def _f(d):
            return f"{d['pos_rms_km']:.3f}" if d else "---"

        lines.append(
            f"{case.label} & {r['e']:.2f} & {_f(ms)} & {_f(do)} & {_f(e6)} & {_f(e8)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table C (ecc-power) written to {out_path}")


def write_timing_table(results, out_path):
    """Wall-clock timing table for all methods."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Wall-clock propagation time [s] per case.}")
    lines.append(r"\label{tab:dsst_timing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & Cowell & GEqOE m+SP & GEqOE mean & DSST osc & Brouwer \\")
    lines.append(r"\midrule")

    for r in results:
        if not r.get("cowell_ok"):
            continue
        case = next(c for c in CASES if c.name == r["case_name"])

        def _t(key):
            v = r.get(key)
            return f"{v:.2f}" if v is not None else "---"

        lines.append(
            f"{case.label} & {_t('cowell_time')} & {_t('meansp_time')} & "
            f"{_t('geqoe_mean_only_time')} & {_t('dsst_osc_time')} & "
            f"{_t('brouwer_time')} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Timing table written to {out_path}")


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
        if "pos_err_dsst_osc" in r:
            tau = r["t_grid"] / T
            ax.plot(tau, r["pos_err_dsst_osc"], lw=0.8, label="DSST", ls=":")
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
            if "rad_err_dsst_osc" in r:
                tau = r["t_grid"] / T
                ax.plot(tau, r["rad_err_dsst_osc"], lw=0.8, label="DSST", ls=":")
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

    # --- Figure 3: Bar chart — three theories ---
    names, geqoe_vals, dsst_vals, brouwer_vals = [], [], [], []
    for r in results:
        ms = r.get("geqoe_meansp")
        do = r.get("dsst_osc")
        br = r.get("brouwer")
        if ms and br:
            case = next(c for c in CASES if c.name == r["case_name"])
            names.append(case.name)
            geqoe_vals.append(ms["pos_rms_km"])
            dsst_vals.append(do["pos_rms_km"] if do else np.nan)
            brouwer_vals.append(br["pos_rms_km"])

    if names:
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(names))
        w = 0.25
        ax.bar(x - w, geqoe_vals, w, label="GEqOE mean+SP", color="C0")
        ax.bar(x, dsst_vals, w, label="DSST osculating", color="C1")
        ax.bar(x + w, brouwer_vals, w, label="Brouwer-Lyddane", color="C2")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Position RMS error [km]")
        # Only use log scale if there are positive values to display
        all_vals = [v for v in geqoe_vals + dsst_vals + brouwer_vals
                    if np.isfinite(v) and v > 0]
        if all_vals:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        out = FIG_DIR / "extended_validation_comparison_bar.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  Figure saved: {out}")

    # --- Figure 4: Cost-accuracy scatter plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"GEqOE m+SP": ("o", "C0"), "DSST osc": ("s", "C1"),
               "Brouwer": ("^", "C2"), "Cowell": ("D", "C3")}
    for r in results:
        if not r.get("cowell_ok"):
            continue
        case = next(c for c in CASES if c.name == r["case_name"])
        pairs = []
        if r.get("geqoe_meansp") and r.get("meansp_time"):
            pairs.append(("GEqOE m+SP", r["meansp_time"], r["geqoe_meansp"]["pos_rms_km"]))
        if r.get("dsst_osc") and r.get("dsst_osc_time"):
            pairs.append(("DSST osc", r["dsst_osc_time"], r["dsst_osc"]["pos_rms_km"]))
        if r.get("brouwer") and r.get("brouwer_time"):
            pairs.append(("Brouwer", r["brouwer_time"], r["brouwer"]["pos_rms_km"]))
        for label, t_wall, pos_rms in pairs:
            m, c = markers[label]
            ax.scatter(t_wall, pos_rms, marker=m, color=c, s=40, alpha=0.7)

    # Legend (one entry per theory)
    for label, (m, c) in markers.items():
        if label == "Cowell":
            continue
        ax.scatter([], [], marker=m, color=c, s=40, label=label)
    ax.set_xlabel("Wall-clock time [s]")
    ax.set_ylabel("Position RMS error [km]")
    # Guard against empty data (all DSST/Brouwer failed)
    try:
        ax.set_xscale("log")
        ax.set_yscale("log")
    except ValueError:
        pass  # no positive data to log-scale
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Cost-accuracy Pareto frontier")
    fig.tight_layout()
    out = FIG_DIR / "cost_accuracy_scatter.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  Figure saved: {out}")

    # --- Figure 5: Mean-drift comparison ---
    drift_cases = ["LEO-circ", "Crit-low-e", "Molniya", "MEO-GPS"]
    drift_results = [r for r in results if r["case_name"] in drift_cases]
    if drift_results:
        n_panels = len(drift_results)
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]
        for idx, r in enumerate(drift_results):
            ax = axes[idx]
            T = r.get("T_orbit_s", 1.0)
            case = next(c for c in CASES if c.name == r["case_name"])
            if "pos_err_geqoe_mean_only" in r:
                tau = r["t_grid"] / T
                ax.plot(tau, r["pos_err_geqoe_mean_only"], lw=0.8, label="GEqOE mean")
            if "pos_err_dsst_mean" in r:
                tau = r["t_grid"] / T
                ax.plot(tau, r["pos_err_dsst_mean"], lw=0.8, label="DSST mean", ls=":")
            ax.set_title(f"{case.name}", fontsize=10)
            ax.set_xlabel("orbits", fontsize=9)
            ax.set_ylabel("pos err [km]", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / "mean_drift_comparison.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  Figure saved: {out}")

    # --- Figure 6: Eccentricity power convergence ---
    high_e_results = [r for r in results
                      if r.get("e", 0) >= 0.35 and r.get("cowell_ok")]
    if high_e_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = []
        bar_data = {}  # {label: [values]}
        for r in high_e_results:
            case = next(c for c in CASES if c.name == r["case_name"])
            x_labels.append(case.name)
            ms = r.get("geqoe_meansp")
            do = r.get("dsst_osc")
            e6 = r.get("dsst_osc_ecc6")
            e8 = r.get("dsst_osc_ecc8")
            bar_data.setdefault("GEqOE", []).append(ms["pos_rms_km"] if ms else np.nan)
            bar_data.setdefault("DSST ecc=4", []).append(do["pos_rms_km"] if do else np.nan)
            bar_data.setdefault("DSST ecc=6", []).append(e6["pos_rms_km"] if e6 else np.nan)
            bar_data.setdefault("DSST ecc=8", []).append(e8["pos_rms_km"] if e8 else np.nan)

        x = np.arange(len(x_labels))
        n_bars = len(bar_data)
        w = 0.8 / n_bars
        for i, (label, vals) in enumerate(bar_data.items()):
            offset = (i - n_bars / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("Position RMS error [km]")
        all_bar = [v for vs in bar_data.values() for v in vs
                   if np.isfinite(v) and v > 0]
        if all_bar:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title("Eccentricity power convergence (high-e cases)")
        fig.tight_layout()
        out = FIG_DIR / "ecc_power_convergence.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"  Figure saved: {out}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("  Extended GEqOE Averaged Theory Validation")
    print("  (with DSST semi-analytical comparison)")
    print("=" * 60)

    # Pre-build symbolic cache
    print("\nBuilding symbolic short-period cache...")
    _ensure_symbolic_cache(J_COEFFS)

    # Pre-build heyoka cfuncs (one-time cost, excluded from per-case timings)
    try:
        from geqoe_mean.heyoka_compiled import get_mean_rhs_cfunc, get_sp_cfunc
        print("Building heyoka cfuncs (one-time cost)...")
        _, info_mean = get_mean_rhs_cfunc()
        print(f"  Mean RHS cfunc: {info_mean['build_time']:.1f}s build + "
              f"{info_mean['compile_time']:.1f}s compile")
        _, info_sp = get_sp_cfunc()
        print(f"  SP cfunc: {info_sp['build_time']:.1f}s build + "
              f"{info_sp['compile_time']:.1f}s compile")
    except ImportError:
        print("  heyoka cfuncs not available, using Python fallback")

    # Full pipeline warmup: run a short dummy case to trigger all first-call
    # JIT costs (scipy DOP853, numpy, geqoe2cart heyoka compilation, etc.)
    print("Pipeline warmup (short dummy propagation)...")
    _r0, _v0 = kepler_to_rv(7000, 0.01, 45, 0, 0, 0)
    _pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)
    _s0 = cart2geqoe(_r0, _v0, MU, _pert)
    _m0 = osculating_to_mean_state(_s0, J_COEFFS, re_val=RE, mu_val=MU)
    _tg = np.linspace(0, 5000, 10)
    _mh = rk4_integrate_mean(_m0, _tg, J_COEFFS, substeps=4)
    _ob = mean_to_osculating_state_batch(_mh, J_COEFFS, re_val=RE, mu_val=MU)
    geqoe2cart_zonal_batch(_ob, MU, _pert)
    print("  done.")

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
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    fmt = "{:<14s} {:>6s} {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Case", "e", "i", "GEqOE RMS", "Eqnoc RMS", "DSST RMS", "Brouwer"))
    print("-" * 75)
    for r in all_results:
        ms = r.get("geqoe_meansp")
        eq = r.get("geqoe_eqnoc")
        do = r.get("dsst_osc")
        br = r.get("brouwer")
        ms_s = f"{ms['pos_rms_km']:.4f}" if ms else "FAIL"
        eq_s = f"{eq['pos_rms_km']:.4f}" if eq else "FAIL"
        do_s = f"{do['pos_rms_km']:.4f}" if do else "FAIL"
        br_s = f"{br['pos_rms_km']:.4f}" if br else "FAIL"
        print(fmt.format(
            r["case_name"],
            f"{r.get('e', '?')}",
            f"{r.get('inc_deg', '?')}",
            ms_s, eq_s, do_s, br_s))

    # Write LaTeX tables
    table_dir = DOC_DIR / "main_docs"
    write_osculating_table(all_results, table_dir / "dsst_osculating_table.tex")
    write_mean_drift_table(all_results, table_dir / "dsst_mean_drift_table.tex")
    write_ecc_power_table(all_results, table_dir / "dsst_ecc_power_table.tex")
    write_timing_table(all_results, table_dir / "dsst_timing_table.tex")

    print("\nGenerating figures...")
    create_figures(all_results)

    # Save raw metrics as JSON
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
