#!/usr/bin/env python
"""Diagnostic: lambda-scaling slope decomposition.

Investigates why the zonal-scaling test gives slope ~ 1 (not 2).
Four tests decompose the error to identify whether the O(eps) scaling
comes from the theory itself or from the geqoe->cart mapping.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/scaling_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

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
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen

from geqoe_mean.constants import J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    evaluate_truncated_short_period,
    mean_to_osculating_state,
    osculating_to_mean_state,
)
from geqoe_mean.validation import (
    ensure_symbolic_cache as _ensure_symbolic_cache,
)

from zonal_short_period_validation import (
    CASES,
    ValidationCase,
    run_case,
)

OUT_FIG = FIG_DIR / "scaling_diagnostic.png"


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------


def fit_slope(scales: np.ndarray, errors: np.ndarray) -> float:
    """Fit log-log slope via least squares."""
    mask = (scales > 0) & (errors > 0)
    log_s = np.log(scales[mask])
    log_e = np.log(errors[mask])
    slope, _ = np.polyfit(log_s, log_e, 1)
    return float(slope)


def rms(x: np.ndarray) -> float:
    """Root-mean-square of an array."""
    return float(np.sqrt(np.mean(x * x)))


# ---------------------------------------------------------------------------
#  Test A: GEqOE-level vs position-level scaling
# ---------------------------------------------------------------------------


def test_a(case: ValidationCase, anomaly_deg: float) -> dict:
    """Measure scaling slopes for individual GEqOE components and position."""
    scales = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.125, 0.0625])

    err_p1 = []
    err_p2 = []
    err_K = []
    err_q1 = []
    err_q2 = []
    err_pos = []

    for lam in scales:
        print(f"  Test A [{case.name}]: lambda={lam:.4f}", flush=True)
        result = run_case(case, anomaly_deg, scale=lam)
        osc = result["osc_hist"]
        rec = result["rec_hist"]

        err_p1.append(rms(rec[:, 1] - osc[:, 1]))
        err_p2.append(rms(rec[:, 2] - osc[:, 2]))
        err_K.append(rms(np.unwrap(rec[:, 3]) - np.unwrap(osc[:, 3])))
        err_q1.append(rms(rec[:, 4] - osc[:, 4]))
        err_q2.append(rms(rec[:, 5] - osc[:, 5]))
        err_pos.append(result["metrics"]["pos_rms_km"])

    err_p1 = np.array(err_p1)
    err_p2 = np.array(err_p2)
    err_K = np.array(err_K)
    err_q1 = np.array(err_q1)
    err_q2 = np.array(err_q2)
    err_pos = np.array(err_pos)

    slopes = {
        "p1": fit_slope(scales, err_p1),
        "p2": fit_slope(scales, err_p2),
        "K": fit_slope(scales, err_K),
        "q1": fit_slope(scales, err_q1),
        "q2": fit_slope(scales, err_q2),
        "pos": fit_slope(scales, err_pos),
    }

    return {
        "case_name": case.name,
        "scales": scales,
        "errors": {
            "p1": err_p1, "p2": err_p2, "K": err_K,
            "q1": err_q1, "q2": err_q2, "pos": err_pos,
        },
        "slopes": slopes,
    }


# ---------------------------------------------------------------------------
#  Test B: Time-resolved scaling
# ---------------------------------------------------------------------------


def test_b(case: ValidationCase, anomaly_deg: float) -> dict:
    """Measure position error scaling at specific time slices."""
    scales = np.array([1.0, 0.5, 0.25, 0.125])

    # We need at least 8 orbits of data; run_case uses case.n_orbits
    # Collect the full time history for each scale, then slice
    results_by_scale = {}
    for lam in scales:
        print(f"  Test B [{case.name}]: lambda={lam:.4f}", flush=True)
        results_by_scale[lam] = run_case(case, anomaly_deg, scale=lam)

    # Determine time slices: t=0, t=T (1 orbit), t=8T or last available
    T_orbit = results_by_scale[scales[0]]["metrics"]["T_orbit"]
    n_per_orbit = case.samples_per_orbit
    total_samples = case.n_orbits * n_per_orbit + 1

    # Indices: t=0 -> index 0, t=T -> index n_per_orbit, t=8T -> index 8*n_per_orbit
    idx_0 = 0
    idx_1T = min(n_per_orbit, total_samples - 1)
    idx_8T = min(8 * n_per_orbit, total_samples - 1)

    slice_labels = ["t=0", "t=T", f"t={min(8, case.n_orbits)}T"]
    slice_indices = [idx_0, idx_1T, idx_8T]

    # For each time slice, gather position error across scales
    time_slopes = {}
    time_errors = {}
    for label, idx in zip(slice_labels, slice_indices):
        errs = []
        for lam in scales:
            pos_err = results_by_scale[lam]["pos_err"]
            errs.append(float(pos_err[idx]))
        errs = np.array(errs)
        time_errors[label] = errs
        # Only fit slope if errors are all positive
        if np.all(errs > 0):
            time_slopes[label] = fit_slope(scales, errs)
        else:
            time_slopes[label] = float("nan")

    return {
        "case_name": case.name,
        "scales": scales,
        "slice_labels": slice_labels,
        "time_errors": time_errors,
        "time_slopes": time_slopes,
    }


# ---------------------------------------------------------------------------
#  Test C: Direct SP validation (bypass round-trip)
# ---------------------------------------------------------------------------


def test_c(case: ValidationCase, anomaly_deg: float) -> dict:
    """Compare numerical vs theoretical short-period oscillation over 1 orbit."""
    # Run truth propagation for 1 orbit with dense sampling
    n_samples = 1000
    one_orbit_case = ValidationCase(
        name=case.name + "_1orb",
        a_km=case.a_km,
        e=case.e,
        inc_deg=case.inc_deg,
        raan_deg=case.raan_deg,
        argp_deg=case.argp_deg,
        anomalies_deg=(anomaly_deg,),
        n_orbits=1,
        samples_per_orbit=n_samples,
    )
    print(f"  Test C [{case.name}]: propagating 1 orbit with {n_samples} samples", flush=True)

    # Use run_case infrastructure but we only need osc_hist
    coeffs = J_COEFFS
    pert = ZonalPerturbation(coeffs, mu=MU, re=RE)

    r0, v0 = kepler_to_rv(
        case.a_km, case.e, case.inc_deg, case.raan_deg,
        case.argp_deg, anomaly_deg,
    )
    from astrodyn_core.geqoe_taylor import build_state_integrator
    from astrodyn_core.geqoe_taylor.integrator import propagate_grid

    state0 = cart2geqoe(r0, v0, MU, pert)
    nu0 = float(state0[0])
    T_orbit = 2.0 * np.pi / nu0
    t_grid = np.linspace(0.0, T_orbit, n_samples + 1, dtype=float)

    ta, _ = build_state_integrator(pert, state0, tol=1.0e-15, compact_mode=True)
    osc_hist = propagate_grid(ta, t_grid)

    # Compute mean state for each osculating snapshot
    print(f"  Test C [{case.name}]: computing osc->mean inversion for {len(osc_hist)} points", flush=True)
    mean_hist = np.array([
        osculating_to_mean_state(s, coeffs, re_val=RE, mu_val=MU) for s in osc_hist
    ])

    # Numerical SP = osculating - orbit_averaged (for slow variables in polar form)
    # Extract slow polar variables from osculating and mean states
    # Slow variables: g, Q, Psi, Omega, M
    def _extract_polar(states):
        """Extract (g, Q, Psi, Omega, M) from GEqOE state array."""
        p1 = states[:, 1]
        p2 = states[:, 2]
        K = states[:, 3]
        q1 = states[:, 4]
        q2 = states[:, 5]
        g = np.hypot(p1, p2)
        Q = np.hypot(q1, q2)
        Psi = np.arctan2(p1, p2)
        Omega = np.arctan2(q1, q2)
        # M = L - Psi where L = K + p1*cos(K) - p2*sin(K)
        L = K + p1 * np.cos(K) - p2 * np.sin(K)
        M = L - Psi
        return g, Q, Psi, Omega, M

    g_osc, Q_osc, Psi_osc, Omega_osc, M_osc = _extract_polar(osc_hist)

    # Orbit average: numerical mean over the 1-orbit grid
    g_avg = np.mean(g_osc)
    Q_avg = np.mean(Q_osc)
    Psi_avg = np.mean(Psi_osc)
    Omega_avg = np.mean(Omega_osc)
    M_avg = np.mean(M_osc)

    # Numerical short-period: deviation from orbit average
    sp_num = {
        "g": g_osc - g_avg,
        "Q": Q_osc - Q_avg,
        "Psi": Psi_osc - Psi_avg,
        "Omega": Omega_osc - Omega_avg,
        "M": M_osc - M_avg,
    }

    # Theoretical SP: mean_to_osculating - mean for each mean state
    # u1 = mean_to_osculating(mean) - mean
    print(f"  Test C [{case.name}]: computing theoretical SP corrections", flush=True)
    rec_hist = np.array([
        mean_to_osculating_state(s, coeffs, re_val=RE, mu_val=MU) for s in mean_hist
    ])
    g_rec, Q_rec, Psi_rec, Omega_rec, M_rec = _extract_polar(rec_hist)
    g_mean, Q_mean, Psi_mean, Omega_mean, M_mean = _extract_polar(mean_hist)

    sp_theo = {
        "g": g_rec - g_mean,
        "Q": Q_rec - Q_mean,
        "Psi": Psi_rec - Psi_mean,
        "Omega": Omega_rec - Omega_mean,
        "M": M_rec - M_mean,
    }

    # Compare
    sp_comparison = {}
    for var in ("g", "Q", "Psi", "Omega", "M"):
        amp_num = rms(sp_num[var])
        amp_theo = rms(sp_theo[var])
        residual = rms(sp_num[var] - sp_theo[var])
        sp_comparison[var] = {
            "numerical_rms": amp_num,
            "theoretical_rms": amp_theo,
            "residual_rms": residual,
        }

    return {
        "case_name": case.name,
        "t_grid": t_grid,
        "T_orbit": T_orbit,
        "sp_num": sp_num,
        "sp_theo": sp_theo,
        "sp_comparison": sp_comparison,
    }


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


def create_diagnostic_figure(
    test_a_low: dict,
    test_a_high: dict,
    test_b_low: dict,
    test_b_high: dict,
    test_c_low: dict,
    test_c_high: dict,
) -> None:
    """Create a 4-panel diagnostic figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: GEqOE component scaling (low-e)
    ax = axes[0, 0]
    a_data = test_a_low
    scales = a_data["scales"]
    for comp, label in [("p1", r"$p_1$"), ("p2", r"$p_2$"), ("K", r"$K$"),
                        ("q1", r"$q_1$"), ("q2", r"$q_2$"), ("pos", "position")]:
        style = "o--" if comp == "pos" else "s-"
        lw = 2.0 if comp == "pos" else 1.2
        ax.loglog(scales, a_data["errors"][comp], style, lw=lw,
                  label=f"{label} (slope={a_data['slopes'][comp]:.2f})")
    # Reference slopes
    ref_x = np.array([scales[0], scales[-1]])
    ref_y1 = a_data["errors"]["pos"][0] * (ref_x / ref_x[0]) ** 1
    ref_y2 = a_data["errors"]["pos"][0] * (ref_x / ref_x[0]) ** 2
    ax.loglog(ref_x, ref_y1, "k:", alpha=0.4, label="slope 1")
    ax.loglog(ref_x, ref_y2, "k--", alpha=0.4, label="slope 2")
    ax.set_xlabel(r"$\lambda$ (zonal scale)")
    ax.set_ylabel("RMS error")
    ax.set_title(f"GEqOE component scaling ({test_a_low['case_name']})")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: GEqOE component scaling (high-e)
    ax = axes[0, 1]
    a_data = test_a_high
    scales = a_data["scales"]
    for comp, label in [("p1", r"$p_1$"), ("p2", r"$p_2$"), ("K", r"$K$"),
                        ("q1", r"$q_1$"), ("q2", r"$q_2$"), ("pos", "position")]:
        style = "o--" if comp == "pos" else "s-"
        lw = 2.0 if comp == "pos" else 1.2
        ax.loglog(scales, a_data["errors"][comp], style, lw=lw,
                  label=f"{label} (slope={a_data['slopes'][comp]:.2f})")
    ref_x = np.array([scales[0], scales[-1]])
    ref_y1 = a_data["errors"]["pos"][0] * (ref_x / ref_x[0]) ** 1
    ref_y2 = a_data["errors"]["pos"][0] * (ref_x / ref_x[0]) ** 2
    ax.loglog(ref_x, ref_y1, "k:", alpha=0.4, label="slope 1")
    ax.loglog(ref_x, ref_y2, "k--", alpha=0.4, label="slope 2")
    ax.set_xlabel(r"$\lambda$ (zonal scale)")
    ax.set_ylabel("RMS error")
    ax.set_title(f"GEqOE component scaling ({test_a_high['case_name']})")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 3: Time-resolved scaling
    ax = axes[1, 0]
    markers = ["o", "s", "D"]
    colors_low = ["#1f77b4", "#2ca02c", "#d62728"]
    colors_high = ["#aec7e8", "#98df8a", "#ff9896"]
    for i, label in enumerate(test_b_low["slice_labels"]):
        errs = test_b_low["time_errors"][label]
        slope = test_b_low["time_slopes"][label]
        ax.loglog(test_b_low["scales"], errs, f"{markers[i]}-",
                  color=colors_low[i], lw=1.5,
                  label=f"low-e {label} (slope={slope:.2f})")
    for i, label in enumerate(test_b_high["slice_labels"]):
        errs = test_b_high["time_errors"][label]
        slope = test_b_high["time_slopes"][label]
        ax.loglog(test_b_high["scales"], errs, f"{markers[i]}--",
                  color=colors_high[i], lw=1.5,
                  label=f"high-e {label} (slope={slope:.2f})")
    ax.set_xlabel(r"$\lambda$ (zonal scale)")
    ax.set_ylabel("position error [km]")
    ax.set_title("Time-resolved scaling")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 4: SP amplitude comparison
    ax = axes[1, 1]
    variables = ["g", "Q", "Psi", "Omega", "M"]
    x_pos = np.arange(len(variables))
    width = 0.18

    # low-e bars
    num_low = [test_c_low["sp_comparison"][v]["numerical_rms"] for v in variables]
    theo_low = [test_c_low["sp_comparison"][v]["theoretical_rms"] for v in variables]
    resid_low = [test_c_low["sp_comparison"][v]["residual_rms"] for v in variables]

    # high-e bars
    num_high = [test_c_high["sp_comparison"][v]["numerical_rms"] for v in variables]
    theo_high = [test_c_high["sp_comparison"][v]["theoretical_rms"] for v in variables]
    resid_high = [test_c_high["sp_comparison"][v]["residual_rms"] for v in variables]

    ax.bar(x_pos - 1.5 * width, num_low, width, label="numerical (low-e)", color="#1f77b4")
    ax.bar(x_pos - 0.5 * width, theo_low, width, label="theoretical (low-e)", color="#ff7f0e")
    ax.bar(x_pos + 0.5 * width, num_high, width, label="numerical (high-e)", color="#2ca02c")
    ax.bar(x_pos + 1.5 * width, theo_high, width, label="theoretical (high-e)", color="#d62728")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r"$g$", r"$Q$", r"$\Psi$", r"$\Omega$", r"$M$"])
    ax.set_ylabel("SP RMS amplitude")
    ax.set_title("Short-period: numerical vs theoretical (1 orbit)")
    ax.legend(fontsize=7, loc="best")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3, axis="y")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"\nSaved figure: {OUT_FIG}")


# ---------------------------------------------------------------------------
#  Summary output
# ---------------------------------------------------------------------------


def print_summary(
    test_a_low: dict,
    test_a_high: dict,
    test_b_low: dict,
    test_b_high: dict,
    test_c_low: dict,
    test_c_high: dict,
) -> None:
    """Print a summary table."""
    sep = "-" * 78

    print(f"\n{sep}")
    print("TEST A: GEqOE component scaling slopes")
    print(sep)
    header = f"{'case':>8s}  {'p1':>8s}  {'p2':>8s}  {'K':>8s}  {'q1':>8s}  {'q2':>8s}  {'pos':>8s}"
    print(header)
    for data in (test_a_low, test_a_high):
        s = data["slopes"]
        row = (f"{data['case_name']:>8s}  {s['p1']:8.3f}  {s['p2']:8.3f}  "
               f"{s['K']:8.3f}  {s['q1']:8.3f}  {s['q2']:8.3f}  {s['pos']:8.3f}")
        print(row)

    print(f"\n{sep}")
    print("TEST B: Time-resolved position error slopes")
    print(sep)
    header = f"{'case':>8s}  {'slice':>8s}  {'slope':>8s}"
    print(header)
    for data in (test_b_low, test_b_high):
        for label in data["slice_labels"]:
            row = f"{data['case_name']:>8s}  {label:>8s}  {data['time_slopes'][label]:8.3f}"
            print(row)

    print(f"\n{sep}")
    print("TEST C: Short-period amplitude comparison (1 orbit)")
    print(sep)
    header = (f"{'case':>8s}  {'var':>6s}  {'num_rms':>12s}  {'theo_rms':>12s}  "
              f"{'resid_rms':>12s}  {'ratio':>8s}")
    print(header)
    for data in (test_c_low, test_c_high):
        for var in ("g", "Q", "Psi", "Omega", "M"):
            c = data["sp_comparison"][var]
            ratio = c["residual_rms"] / c["numerical_rms"] if c["numerical_rms"] > 0 else float("nan")
            row = (f"{data['case_name']:>8s}  {var:>6s}  {c['numerical_rms']:12.6e}  "
                   f"{c['theoretical_rms']:12.6e}  {c['residual_rms']:12.6e}  {ratio:8.4f}")
            print(row)

    print(sep)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    print("Precomputing exact symbolic short-period cache...", flush=True)
    _ensure_symbolic_cache(J_COEFFS)

    case_low = CASES[0]   # low-e
    case_high = CASES[1]  # high-e
    anom_low = case_low.anomalies_deg[0]
    anom_high = case_high.anomalies_deg[0]

    # Test A: GEqOE component scaling
    print("\n=== Test A: GEqOE component scaling ===", flush=True)
    test_a_low = test_a(case_low, anom_low)
    test_a_high = test_a(case_high, anom_high)

    # Test B: Time-resolved scaling
    print("\n=== Test B: Time-resolved scaling ===", flush=True)
    test_b_low = test_b(case_low, anom_low)
    test_b_high = test_b(case_high, anom_high)

    # Test C: Direct SP validation
    print("\n=== Test C: Direct SP validation ===", flush=True)
    test_c_low = test_c(case_low, anom_low)
    test_c_high = test_c(case_high, anom_high)

    # Test D is covered by running A and B for both cases above

    # Summary
    print_summary(test_a_low, test_a_high, test_b_low, test_b_high,
                  test_c_low, test_c_high)

    # Figure
    create_diagnostic_figure(test_a_low, test_a_high, test_b_low, test_b_high,
                             test_c_low, test_c_high)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
