#!/usr/bin/env python3
"""Fair secular-rate comparison: GEqOE vs DSST vs Cowell truth.

Compares the slow-element secular rates (Ψ̇, Ω̇) from three sources:
  1. Cowell truth: osculating → classical Keplerian elements → orbit-average
     over each revolution (independent of any SP theory)
  2. GEqOE mean: direct mean-element propagation
  3. DSST mean: Orekit DSST with state_type=MEAN

The orbit-average truth avoids bias toward either theory's SP definition.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
REPO_ROOT = DOC_DIR.parents[1]
FIG_DIR = DOC_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from astrodyn_core.geqoe_taylor import (
    MU, RE, ZonalPerturbation, cart2geqoe, geqoe2cart,
)
from astrodyn_core.geqoe_taylor.cowell import (
    _build_cowell_heyoka_general_system, _build_par_values,
)
from geqoe_mean.constants import J2, J3, J4, J5, J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import osculating_to_mean_state
from geqoe_mean.validation import rk4_integrate_mean


# ---------------------------------------------------------------------------
#  Orbit cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    label: str
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float = 30.0
    argp_deg: float = 60.0
    M0_deg: float = 45.0
    n_orbits: int = 30
    rk4_substeps: int = 8

CASES = [
    Case("LEO-circ",    "LEO circ.",      6878, 0.001, 51.6,  n_orbits=30),
    Case("LEO-mod-e",   "LEO mod-$e$",    7000, 0.05,  40.0,  n_orbits=30),
    Case("Near-equat",  "Near-equat.",     7200, 0.01,   5.0,  n_orbits=30),
    Case("Crit-low-e",  r"Crit.\ low-$e$",7500, 0.01,  63.435,n_orbits=50),
    Case("Molniya",     "Molniya",       26554, 0.74,  63.4,  n_orbits=15),
    Case("MEO-GPS",     "MEO/GPS",       26560, 0.01,  55.0,  n_orbits=30),
]


# ---------------------------------------------------------------------------
#  Orekit setup
# ---------------------------------------------------------------------------

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
        crawler = ZipJarCrawler(java.io.File(str(zip_path)))
        DataContext.getDefault().getDataProvidersManager().addProvider(crawler)
        _OREKIT_AVAILABLE = True
    except Exception as exc:
        print(f"  [WARN] Orekit not available: {exc}")
        _OREKIT_AVAILABLE = False
    return _OREKIT_AVAILABLE


def _build_orekit_orbit(case):
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    import math
    frame = FramesFactory.getEME2000()
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)
    orbit = KeplerianOrbit(
        float(case.a_km * 1e3), float(case.e),
        float(math.radians(case.inc_deg)),
        float(math.radians(case.argp_deg)),
        float(math.radians(case.raan_deg)),
        float(math.radians(case.M0_deg)),
        PositionAngleType.MEAN, frame, epoch, float(MU * 1e9),
    )
    return orbit, epoch, frame


_CUSTOM_PROVIDER = None

def _get_matched_zonal_provider():
    global _CUSTOM_PROVIDER
    if _CUSTOM_PROVIDER is not None:
        return _CUSTOM_PROVIDER
    from org.orekit.forces.gravity.potential import (
        PythonUnnormalizedSphericalHarmonicsProvider,
        PythonUnnormalizedSphericalHarmonics, TideSystem,
    )
    from org.orekit.time import AbsoluteDate
    j_map = {2: J2, 3: J3, 4: J4, 5: J5}
    mu_m3s2, re_m = MU * 1e9, RE * 1e3

    class _H(PythonUnnormalizedSphericalHarmonics):
        def __init__(self, d): super().__init__(); self._d = d
        def getDate(self): return self._d
        def getUnnormalizedCnm(self, n, m):
            return -j_map.get(n, 0.0) if m == 0 else 0.0
        def getUnnormalizedSnm(self, n, m): return 0.0

    class _P(PythonUnnormalizedSphericalHarmonicsProvider):
        def getAe(self):            return re_m
        def getMu(self):            return mu_m3s2
        def getMaxDegree(self):     return 5
        def getMaxOrder(self):      return 0
        def getReferenceDate(self): return AbsoluteDate.J2000_EPOCH
        def getTideSystem(self):    return TideSystem.UNKNOWN
        def onDate(self, d):        return _H(d)
        def getUnnormalizedC20(self, d): return -j_map[2]

    _CUSTOM_PROVIDER = _P()
    return _CUSTOM_PROVIDER


def propagate_dsst_mean_equinoctial(orbit, epoch, frame, t_grid):
    """DSST mean → (ey, ex, hy, hx) at each epoch."""
    from org.orekit.propagation.semianalytical.dsst.forces import (
        DSSTZonal, DSSTJ2SquaredClosedForm, ZeisModel,
    )
    from org.orekit.propagation.conversion import (
        DSSTPropagatorBuilder, DormandPrince853IntegratorBuilder,
    )
    from org.orekit.propagation import PropagationType
    from org.orekit.orbits import EquinoctialOrbit

    provider = _get_matched_zonal_provider()
    builder = DSSTPropagatorBuilder(
        orbit,
        DormandPrince853IntegratorBuilder(float(1e-3), float(300.0), float(1e-6)),
        float(10.0), PropagationType.MEAN, PropagationType.MEAN,
    )
    builder.setMass(1.0)
    builder.addForceModel(DSSTZonal(provider))
    builder.addForceModel(DSSTJ2SquaredClosedForm(ZeisModel(), provider))
    prop = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    out = np.empty((len(t_grid), 4))
    for i, t in enumerate(t_grid):
        orb = EquinoctialOrbit(
            prop.propagate(epoch.shiftedBy(float(t))).getOrbit())
        out[i] = [float(orb.getEquinoctialEy()),   # ~ p1
                  float(orb.getEquinoctialEx()),    # ~ p2
                  float(orb.getHy()),                # ~ q1
                  float(orb.getHx())]                # ~ q2
    return out


# ---------------------------------------------------------------------------
#  Theory-independent orbit-average truth
# ---------------------------------------------------------------------------

def cart_to_keplerian_angles(r, v, mu):
    """Cartesian → (Ω, ω, i, e) via angular momentum and eccentricity vectors."""
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    n_vec = np.cross([0, 0, 1], h)  # nodal vector
    n_mag = np.linalg.norm(n_vec)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e_mag = np.linalg.norm(e_vec)

    # Inclination
    inc = np.arccos(np.clip(h[2] / h_mag, -1, 1))

    # RAAN
    if n_mag > 1e-15:
        Omega = np.arctan2(n_vec[1], n_vec[0])
    else:
        Omega = 0.0

    # Argument of perigee
    if n_mag > 1e-15 and e_mag > 1e-15:
        cos_omega = np.dot(n_vec, e_vec) / (n_mag * e_mag)
        omega = np.arccos(np.clip(cos_omega, -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    return Omega, omega, inc, e_mag


def orbit_average_slow_elements(ta, T_orbit, n_orbits, pts_per_orbit=128):
    """Orbit-average Ω, ω+Ω, i, e from Cowell truth — theory-independent.

    Averages the osculating Keplerian angles over each complete revolution
    to extract the secular trend, without using any SP theory.

    Returns (t_mid, avg_Psi, avg_Omega) at orbit midpoints.
    """
    t_mid = []
    avg_Psi = []
    avg_Omega = []

    for k in range(n_orbits):
        t_start = k * T_orbit
        t_end = (k + 1) * T_orbit
        t_rev = np.linspace(t_start, t_end, pts_per_orbit + 1)

        Omega_rev = np.empty(len(t_rev))
        Psi_rev = np.empty(len(t_rev))

        for j, t in enumerate(t_rev):
            ta.propagate_until(t)
            r_j = np.array(ta.state[:3])
            v_j = np.array(ta.state[3:6])
            Om_j, om_j, _, _ = cart_to_keplerian_angles(r_j, v_j, MU)
            Omega_rev[j] = Om_j
            Psi_rev[j] = om_j + Om_j

        # Unwrap before averaging (handle 2π jumps)
        Omega_rev = np.unwrap(Omega_rev)
        Psi_rev = np.unwrap(Psi_rev)

        # Trapezoidal average over one revolution
        avg_Om = np.trapezoid(Omega_rev, t_rev) / T_orbit
        avg_Ps = np.trapezoid(Psi_rev, t_rev) / T_orbit

        t_mid.append((t_start + t_end) / 2)
        avg_Omega.append(avg_Om)
        avg_Psi.append(avg_Ps)

    return np.array(t_mid), np.array(avg_Psi), np.array(avg_Omega)


# ---------------------------------------------------------------------------
#  Rate fitting
# ---------------------------------------------------------------------------

def fit_rate(t, y):
    """Linear fit → (slope, intercept, rms_residual)."""
    c = np.polyfit(t, y, 1)
    return c[0], c[1], np.sqrt(np.mean((y - np.polyval(c, t))**2))


def slow_to_angles(p1, p2, q1, q2):
    """Equinoctial pairs → unwrapped Ψ, Ω."""
    return (np.unwrap(np.arctan2(p1, p2)),
            np.unwrap(np.arctan2(q1, q2)))


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def run_case(case):
    print(f"\n{'='*70}")
    print(f"  {case.name}: a={case.a_km} km, e={case.e}, i={case.inc_deg} deg")
    print(f"{'='*70}")

    r0, v0 = kepler_to_rv(case.a_km, case.e, case.inc_deg,
                           case.raan_deg, case.argp_deg, case.M0_deg)
    pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)
    state0 = cart2geqoe(r0, v0, MU, pert)
    nu0 = float(state0[0])
    T = 2 * np.pi / nu0

    result = {"case": case, "T": T}

    # --- 1. Cowell truth: orbit-averaged angles (theory-independent) ---
    import heyoka as hy
    t0 = time.time()
    sys_c, _, pm = _build_cowell_heyoka_general_system(
        pert, mu_val=MU, use_par=True, time_origin=0.0)
    ta = hy.taylor_adaptive(
        sys_c, list(r0) + list(v0), tol=1e-15,
        compact_mode=True, pars=_build_par_values(pert, pm))

    t_mid, truth_Psi, truth_Omega = orbit_average_slow_elements(
        ta, T, case.n_orbits, pts_per_orbit=128)
    ta.propagate_until(0.0)

    # Unwrap the orbit-averaged series (may still have jumps across orbits)
    truth_Psi = np.unwrap(truth_Psi)
    truth_Omega = np.unwrap(truth_Omega)

    r_Psi_truth, _, _ = fit_rate(t_mid, truth_Psi)
    r_Om_truth, _, _ = fit_rate(t_mid, truth_Omega)
    print(f"  Cowell orbit-avg: {time.time()-t0:.1f}s "
          f"({case.n_orbits} revolutions, 128 pts/rev)")

    # --- 2. GEqOE mean propagation ---
    t0 = time.time()
    mean0 = osculating_to_mean_state(state0, J_COEFFS, RE, MU)
    # Propagate at orbit midpoints
    mean_hist = rk4_integrate_mean(mean0, t_mid, J_COEFFS,
                                    substeps=case.rk4_substeps)
    geqoe_Psi, geqoe_Omega = slow_to_angles(
        mean_hist[:, 1], mean_hist[:, 2], mean_hist[:, 4], mean_hist[:, 5])
    r_Psi_geqoe, _, _ = fit_rate(t_mid, geqoe_Psi)
    r_Om_geqoe, _, _ = fit_rate(t_mid, geqoe_Omega)
    print(f"  GEqOE mean:       {time.time()-t0:.1f}s")

    # --- 3. DSST mean propagation ---
    r_Psi_dsst = r_Om_dsst = float('nan')
    if _init_orekit():
        t0 = time.time()
        try:
            orbit, epoch, frame = _build_orekit_orbit(case)
            dsst_slow = propagate_dsst_mean_equinoctial(
                orbit, epoch, frame, t_mid)
            dsst_Psi, dsst_Omega = slow_to_angles(*dsst_slow.T)
            r_Psi_dsst, _, _ = fit_rate(t_mid, dsst_Psi)
            r_Om_dsst, _, _ = fit_rate(t_mid, dsst_Omega)
            print(f"  DSST mean:        {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"  DSST mean FAILED: {exc}")

    # --- 4. Compare rates ---
    err_Psi_g = abs(r_Psi_geqoe - r_Psi_truth)
    err_Om_g = abs(r_Om_geqoe - r_Om_truth)
    err_Psi_d = abs(r_Psi_dsst - r_Psi_truth)
    err_Om_d = abs(r_Om_dsst - r_Om_truth)

    ratio_Psi = err_Psi_g / err_Psi_d if err_Psi_d > 0 else float('inf')
    ratio_Om = err_Om_g / err_Om_d if err_Om_d > 0 else float('inf')

    s = 86400.0  # rad/s → rad/day
    print(f"\n  {'':>10s}  {'Truth':>14s}  {'GEqOE':>14s}  {'GEqOE err':>10s}"
          f"  {'DSST':>14s}  {'DSST err':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*85}")
    print(f"  {'Psi_dot':>10s}  {r_Psi_truth*s:+14.8f}  {r_Psi_geqoe*s:+14.8f}"
          f"  {err_Psi_g*s:10.2e}  {r_Psi_dsst*s:+14.8f}"
          f"  {err_Psi_d*s:10.2e}  {ratio_Psi:8.1f}")
    print(f"  {'Omega_dot':>10s}  {r_Om_truth*s:+14.8f}  {r_Om_geqoe*s:+14.8f}"
          f"  {err_Om_g*s:10.2e}  {r_Om_dsst*s:+14.8f}"
          f"  {err_Om_d*s:10.2e}  {ratio_Om:8.1f}")

    result.update({
        "t_mid": t_mid,
        "truth_Psi": truth_Psi, "truth_Omega": truth_Omega,
        "geqoe_Psi": geqoe_Psi, "geqoe_Omega": geqoe_Omega,
        "r_Psi_truth": r_Psi_truth, "r_Om_truth": r_Om_truth,
        "r_Psi_geqoe": r_Psi_geqoe, "r_Om_geqoe": r_Om_geqoe,
        "r_Psi_dsst": r_Psi_dsst, "r_Om_dsst": r_Om_dsst,
        "err_Psi_g": err_Psi_g, "err_Om_g": err_Om_g,
        "err_Psi_d": err_Psi_d, "err_Om_d": err_Om_d,
        "ratio_Psi": ratio_Psi, "ratio_Om": ratio_Om,
    })
    return result


def create_figure(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)

    for row, res in enumerate(all_results):
        case = res["case"]
        t_orb = res["t_mid"] / res["T"]

        # Detrend: subtract truth linear fit for clearer comparison
        s = 86400.0
        r_Psi_t = res["r_Psi_truth"]
        r_Om_t = res["r_Om_truth"]

        # Left: Ψ residual from truth trend
        ax = axes[row, 0]
        truth_detrend = res["truth_Psi"] - (r_Psi_t * res["t_mid"] + res["truth_Psi"][0])
        geqoe_detrend = res["geqoe_Psi"] - (r_Psi_t * res["t_mid"] + res["geqoe_Psi"][0])
        ax.plot(t_orb, truth_detrend * 1e6, 'k.-', ms=2, lw=0.5,
                alpha=0.5, label="Truth (orbit-avg)")
        ax.plot(t_orb, geqoe_detrend * 1e6, 'C0.-', ms=3, lw=1,
                label="GEqOE mean")
        if not np.isnan(res["r_Psi_dsst"]):
            # Reconstruct DSST Psi at orbit midpoints
            dsst_Psi_0 = res["truth_Psi"][0]  # align at t=0
            dsst_detrend_vals = (res["r_Psi_dsst"] - r_Psi_t) * res["t_mid"]
            ax.plot(t_orb, dsst_detrend_vals * 1e6, 'C1.--', ms=3, lw=1,
                    label="DSST mean")
        ax.set_ylabel(r"$\Psi - \Psi_\mathrm{truth\_trend}$ [$\mu$rad]")
        ax.set_title(f"{case.name}: $\\dot\\Psi$ drift from truth", fontsize=10)
        if row == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: Ω residual from truth trend
        ax = axes[row, 1]
        truth_detrend = res["truth_Omega"] - (r_Om_t * res["t_mid"] + res["truth_Omega"][0])
        geqoe_detrend = res["geqoe_Omega"] - (r_Om_t * res["t_mid"] + res["geqoe_Omega"][0])
        ax.plot(t_orb, truth_detrend * 1e6, 'k.-', ms=2, lw=0.5,
                alpha=0.5, label="Truth (orbit-avg)")
        ax.plot(t_orb, geqoe_detrend * 1e6, 'C0.-', ms=3, lw=1,
                label="GEqOE mean")
        if not np.isnan(res["r_Om_dsst"]):
            dsst_detrend_vals = (res["r_Om_dsst"] - r_Om_t) * res["t_mid"]
            ax.plot(t_orb, dsst_detrend_vals * 1e6, 'C1.--', ms=3, lw=1,
                    label="DSST mean")
        ax.set_ylabel(r"$\Omega - \Omega_\mathrm{truth\_trend}$ [$\mu$rad]")
        ax.set_title(f"{case.name}: $\\dot\\Omega$ drift from truth", fontsize=10)
        ax.grid(True, alpha=0.3)

        if row == n - 1:
            axes[row, 0].set_xlabel("orbit index")
            axes[row, 1].set_xlabel("orbit index")

    fig.tight_layout()
    out = FIG_DIR / "secular_rate_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\n  Figure saved: {out}")


def write_rate_table(all_results, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Secular rate errors for $\dot\Psi$ and $\dot\Omega$ [rad/day]",
        r"relative to theory-independent orbit-averaged Cowell truth.",
        r"Ratio $= |\text{GEqOE err}|/|\text{DSST err}|$: values ${>}\,1$",
        r"indicate DSST has better secular rates (expected from its $J_2^2$",
        r"correction).}",
        r"\label{tab:secular_rates}",
        r"\small",
        r"\begin{tabular}{@{}lrrlrrrr@{}}",
        r"\toprule",
        r"Case & $e$ & $i$ & Rate & Truth & GEqOE err & DSST err & Ratio \\",
        r"\midrule",
    ]
    s = 86400.0
    for res in all_results:
        c = res["case"]
        for j, (key, r_t, eg, ed, ratio) in enumerate([
            (r"$\dot\Psi$", res["r_Psi_truth"], res["err_Psi_g"],
             res["err_Psi_d"], res["ratio_Psi"]),
            (r"$\dot\Omega$", res["r_Om_truth"], res["err_Om_g"],
             res["err_Om_d"], res["ratio_Om"]),
        ]):
            name = f"{c.label} & {c.e:.3f} & {c.inc_deg:.1f}" if j == 0 else "& &"
            r_str = "---" if np.isnan(ratio) else f"{ratio:.1f}"
            ed_str = "---" if np.isnan(ed) else f"{ed*s:.2e}"
            lines.append(
                f"{name} & {key} & {r_t*s:+.6f} & {eg*s:.2e} & {ed_str} & {r_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Table written: {out_path}")


def main():
    print("=" * 70)
    print("  Secular Rate Comparison (orbit-averaged Cowell truth)")
    print("=" * 70)

    all_results = []
    for case in CASES:
        try:
            all_results.append(run_case(case))
        except Exception as exc:
            print(f"  CASE {case.name} FAILED: {exc}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  SUMMARY: Ratio = |GEqOE err| / |DSST err|")
    print("    >1 → DSST has better secular rates (J₂² advantage)")
    print("    <1 → GEqOE has better secular rates")
    print(f"{'='*70}")
    fmt = "{:<14s} {:>6s} {:>6s} {:>12s} {:>12s}"
    print(fmt.format("Case", "e", "i", "Ψ̇ ratio", "Ω̇ ratio"))
    print("-" * 55)
    for res in all_results:
        c = res["case"]
        print(fmt.format(
            c.name, f"{c.e}", f"{c.inc_deg}",
            f"{res['ratio_Psi']:.1f}",
            f"{res['ratio_Om']:.1f}"))

    create_figure(all_results)
    write_rate_table(all_results,
                     DOC_DIR / "main_docs" / "secular_rate_table.tex")
    print("\nDone.")


if __name__ == "__main__":
    main()
