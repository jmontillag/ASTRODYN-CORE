#!/usr/bin/env python3
"""Feasibility test: direct equinoctial short-period corrections.

Constructs SP corrections for (p1, p2, q1, q2) directly via the complex
eccentricity zeta = p2 + i*p1 = g*exp(i*Psi) and complex inclination
eta = q2 + i*q1 = Q*exp(i*Omega), bypassing the polar (g, Psi, Q, Omega)
intermediate form.

Key insight: The "reduced" complex rates

    zeta_dot_red / (nu*lambda_n) = [g_series + i*g*Psi_series] * w
    eta_dot_red  / (nu*lambda_n) = Q_series + i*Q*Omega_series

are finite Laurent polynomials in (w, F) with well-behaved coefficients
in (q, Q). The 1/g singularity in Psi_dot is cancelled by the g factor.
The same residue machinery then yields SP kernels that are regular at g -> 0.

Reconstruction:  delta_p1 = Im(delta_zeta_red * exp(i*Omega))
                 delta_p2 = Re(delta_zeta_red * exp(i*Omega))
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

# Ensure geqoe_mean package is importable
_PKG_DIR = Path(__file__).resolve().parent.parent / "geqoe_mean"
sys.path.insert(0, str(_PKG_DIR.parent))

from geqoe_mean.short_period import (
    dimensionless_rate_series,
    _series_by_m,
    _mean_rate_from_raw_series,
    _poly_add,
    _poly_scale,
    _poly_mul,
    _prune,
    _complex_f_from_g,
    evaluate_isolated_degree_short_period,
    q as q_sym,
    Q as Q_sym,
    F as F_sym,
    I,
    g as g_sym,
)
from geqoe_mean.direct_residue import integrate_harmonic_residue
from geqoe_mean.symbolic import q_from_g
from astrodyn_core.geqoe_taylor import MU, RE
from astrodyn_core.geqoe_taylor.constants import J2, J3, J4, J5


# ── Complex rate series construction ─────────────────────────────────────

_W_SHIFT = {(1, 0): sp.Integer(1)}  # multiply by w^1 (shift m-index by +1)


def zeta_reduced_series(n: int) -> dict:
    """zeta_dot_red/(nu*lambda_n) = [g_dot/(nu*lambda_n) + i*g*Psi_dot/(nu*lambda_n)] * w

    Laurent poly in (w, F) with coefficients in (q, Q).
    The g * (1/g) cancellation in the Psi term makes this regular at g -> 0.
    """
    g_s = dimensionless_rate_series("g", n)
    psi_s = dimensionless_rate_series("Psi", n)

    # g_sym = 2q/(1+q^2) cancels the (1+q^2)/(2q) = 1/g in Psi_series
    combined = _poly_add(g_s, _poly_scale(psi_s, I * g_sym))
    shifted = _poly_mul(combined, _W_SHIFT)
    return _prune(shifted)


def eta_reduced_series(n: int) -> dict:
    """eta_dot_red/(nu*lambda_n) = Q_dot/(nu*lambda_n) + i*Q*Omega_dot/(nu*lambda_n)

    Laurent poly in (w, F) with coefficients in (q, Q).
    No w-shift (exp(i*Omega) doesn't involve omega).
    """
    Q_s = dimensionless_rate_series("Q", n)
    omega_s = dimensionless_rate_series("Omega", n)

    combined = _poly_add(Q_s, _poly_scale(omega_s, I * Q_sym))
    return _prune(combined)


# ── SP kernel computation ────────────────────────────────────────────────

def compute_equinoctial_sp(n: int) -> dict[str, dict[int, sp.Expr]]:
    """Compute SP expressions for zeta_reduced and eta_reduced for degree n.

    Returns dict with keys "zeta", "eta", each mapping m -> expr(q, Q, F).
    """
    result = {}
    for name, series_fn in [("zeta", zeta_reduced_series),
                            ("eta", eta_reduced_series)]:
        raw = series_fn(n)
        mean_coeffs = _mean_rate_from_raw_series(raw)
        solved = {}
        for m_val, raw_by_k in _series_by_m(raw).items():
            rational, _c_log = integrate_harmonic_residue(
                raw_by_k, mean_coeffs.get(m_val, sp.Integer(0))
            )
            if rational != 0:
                solved[m_val] = rational
        result[name] = solved
        print(f"    {name}: {len(solved)} harmonics (m = {sorted(solved.keys())})")
    return result


# ── Numerical evaluation ─────────────────────────────────────────────────

def _lambdify_sp(sp_exprs: dict[int, sp.Expr]) -> dict[int, object]:
    """Lambdify SP expressions for fast numerical evaluation."""
    return {
        m_val: sp.lambdify((q_sym, Q_sym, F_sym), expr, "numpy")
        for m_val, expr in sp_exprs.items()
    }


def evaluate_equinoctial_sp_numerical(
    sp_funcs: dict[str, dict[int, object]],
    q_val: float, Q_val: float,
    F_val: complex, w_val: complex, Omega_val: float,
    scale: float,
) -> dict[str, float]:
    """Evaluate delta_p1, delta_p2, delta_q1, delta_q2 from equinoctial SP kernels."""
    exp_iOmega = np.exp(1j * Omega_val)

    # zeta_reduced SP correction
    dzeta_red = sum(
        complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
        for m_val, func in sp_funcs["zeta"].items()
    )
    dzeta = scale * dzeta_red * exp_iOmega

    # eta_reduced SP correction
    deta_red = sum(
        complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
        for m_val, func in sp_funcs["eta"].items()
    )
    deta = scale * deta_red * exp_iOmega

    return {
        "dp1": float(dzeta.imag),   # p1 = Im(zeta)
        "dp2": float(dzeta.real),   # p2 = Re(zeta)
        "dq1": float(deta.imag),    # q1 = Im(eta)
        "dq2": float(deta.real),    # q2 = Re(eta)
    }


# ── Comparison with polar route ──────────────────────────────────────────

def polar_to_cartesian_sp(
    g_val: float, Psi_val: float, Q_val: float, Omega_val: float,
    dg: float, dPsi: float, dQ: float, dOmega: float,
) -> dict[str, float]:
    """Convert polar SP corrections to Cartesian deltas (exact for first-order)."""
    dp1 = dg * np.sin(Psi_val) + g_val * np.cos(Psi_val) * dPsi
    dp2 = dg * np.cos(Psi_val) - g_val * np.sin(Psi_val) * dPsi
    dq1 = dQ * np.sin(Omega_val) + Q_val * np.cos(Omega_val) * dOmega
    dq2 = dQ * np.cos(Omega_val) - Q_val * np.sin(Omega_val) * dOmega
    return {"dp1": dp1, "dp2": dp2, "dq1": dq1, "dq2": dq2}


def run_comparison(
    label: str,
    nu_val: float, p1_val: float, p2_val: float,
    K_val: float, q1_val: float, q2_val: float,
    jn_val: float, n: int,
    sp_funcs: dict[str, dict[int, object]],
):
    """Compare equinoctial vs polar SP for a single degree at one state."""
    g_val = np.hypot(p1_val, p2_val)
    Q_val = np.hypot(q1_val, q2_val)
    Psi_val = np.arctan2(p1_val, p2_val)
    Omega_val = np.arctan2(q1_val, q2_val)
    omega_val = Psi_val - Omega_val
    G_val = K_val - Psi_val

    q_val = q_from_g(g_val)
    a_val = (MU / nu_val**2) ** (1 / 3)
    scale = jn_val * (RE / a_val) ** n
    F_val = _complex_f_from_g(g_val, G_val)
    w_val = np.exp(1j * omega_val)

    # Polar route
    polar = evaluate_isolated_degree_short_period(
        n, nu_val, g_val, Q_val, G_val, omega_val, jn_val, RE, MU
    )
    polar_cart = polar_to_cartesian_sp(
        g_val, Psi_val, Q_val, Omega_val,
        polar["dg"], polar["dPsi"], polar["dQ"], polar["dOmega"]
    )

    # Equinoctial route
    eqn = evaluate_equinoctial_sp_numerical(
        sp_funcs, q_val, Q_val, F_val, w_val, Omega_val, scale
    )

    print(f"\n  {label} (g = {g_val:.2e}, Q = {Q_val:.4f}):")
    print(f"  {'':8s} {'Polar->Cart':>14s}  {'Equinoctial':>14s}  {'Diff':>12s}")
    for key in ("dp1", "dp2", "dq1", "dq2"):
        diff = abs(eqn[key] - polar_cart[key])
        print(f"  {key:8s} {polar_cart[key]:14.10f}  {eqn[key]:14.10f}  {diff:12.2e}")

    # Also show the raw polar corrections for context
    print(f"  [polar raw: dg={polar['dg']:.4e}, dPsi={polar['dPsi']:.4e},"
          f" dQ={polar['dQ']:.4e}, dOmega={polar['dOmega']:.4e}]")


# ── Test orbits ──────────────────────────────────────────────────────────

def make_state(a_km, e, i_deg, omega_deg, Omega_deg, K_deg):
    """Create GEqOE state from Keplerian-like parameters."""
    nu_val = np.sqrt(MU / a_km**3)
    i_rad = np.deg2rad(i_deg)
    omega_rad = np.deg2rad(omega_deg)
    Omega_rad = np.deg2rad(Omega_deg)
    K_val = np.deg2rad(K_deg)

    g_val = e
    Q_val = np.tan(i_rad / 2)
    Psi_val = omega_rad + Omega_rad
    p1_val = g_val * np.sin(Psi_val)
    p2_val = g_val * np.cos(Psi_val)
    q1_val = Q_val * np.sin(Omega_rad)
    q2_val = Q_val * np.cos(Omega_rad)
    return nu_val, p1_val, p2_val, K_val, q1_val, q2_val


def main():
    print("=" * 72)
    print("  Equinoctial Short-Period Feasibility Test")
    print("  Direct (p1,p2,q1,q2) SP via complex eccentricity/inclination")
    print("=" * 72)

    # --- Step 1: Compute SP kernels ---
    degrees_jn = {2: J2, 3: J3}
    sp_data = {}
    sp_funcs = {}
    for n in degrees_jn:
        print(f"\n  Computing degree {n} equinoctial SP expressions...")
        t0 = time.time()
        sp_data[n] = compute_equinoctial_sp(n)
        dt = time.time() - t0
        print(f"    Done in {dt:.1f}s")

        # Lambdify for fast evaluation
        sp_funcs[n] = {
            name: _lambdify_sp(exprs) for name, exprs in sp_data[n].items()
        }

    # --- Step 2: Inspect coefficient regularity at g -> 0 ---
    print("\n" + "-" * 72)
    print("  Coefficient regularity check: evaluate at q -> 0 (g -> 0)")
    print("-" * 72)
    for n, exprs in sp_data.items():
        print(f"\n  Degree {n}, zeta SP (m-harmonics):")
        for m_val, expr in sorted(exprs["zeta"].items()):
            # Check if the expression has a pole at q=0
            limit_val = sp.limit(expr.subs(F_sym, sp.Rational(1, 2)), q_sym, 0)
            has_pole = limit_val in (sp.oo, -sp.oo, sp.zoo)
            if has_pole:
                status = "SINGULAR"
            else:
                num_val = complex(limit_val.subs(Q_sym, sp.Rational(1, 3)))
                status = f"finite (= {num_val:.6f})"
            print(f"    m={m_val:+d}: q->0 limit {status}")

    # --- Step 3: Numerical comparison ---
    test_cases = [
        ("Moderate (e=0.05, i=40)",   7000.0, 0.05,   40.0,  60.0, 30.0, 45.0),
        ("Near-circular (e=0.001)",    7000.0, 0.001,  40.0,  60.0, 30.0, 45.0),
        ("Very circular (e=1e-5)",     7000.0, 1e-5,   40.0,  60.0, 30.0, 45.0),
        ("Near-equatorial (i=1)",      7000.0, 0.05,    1.0,  60.0, 30.0, 45.0),
        ("Near-equatorial (i=0.1)",    7000.0, 0.05,    0.1,  60.0, 30.0, 45.0),
        ("Both small (e=1e-4, i=0.5)", 7000.0, 1e-4,    0.5,  60.0, 30.0, 45.0),
    ]

    for case_label, a, e, i, omega, Omega, K in test_cases:
        print(f"\n{'=' * 72}")
        print(f"  {case_label}")
        print("=" * 72)
        state = make_state(a, e, i, omega, Omega, K)
        for n, jn in degrees_jn.items():
            run_comparison(f"J{n}", *state, jn, n, sp_funcs[n])

    # --- Step 4: Stability sweep at g -> 0 ---
    print(f"\n{'=' * 72}")
    print("  Stability sweep: |delta_p1| as g -> 0 (J2 only)")
    print("=" * 72)
    print(f"  {'g':>12s}  {'|dp1| polar':>14s}  {'|dp1| eqnoc':>14s}  {'|dPsi| raw':>14s}")

    base = make_state(7000.0, 0.05, 40.0, 60.0, 30.0, 45.0)
    nu0, _, _, K0, q10, q20 = base
    Omega0 = np.arctan2(q10, q20)

    for log_g in np.arange(-1, -8, -1):
        g_test = 10.0 ** log_g
        Psi0 = np.deg2rad(90.0)  # arbitrary
        p1_t = g_test * np.sin(Psi0)
        p2_t = g_test * np.cos(Psi0)

        q_t = q_from_g(g_test)
        Q_t = np.hypot(q10, q20)
        omega_t = Psi0 - Omega0
        G_t = K0 - Psi0
        a_t = (MU / nu0**2) ** (1 / 3)
        scale_t = J2 * (RE / a_t) ** 2
        F_t = _complex_f_from_g(g_test, G_t)
        w_t = np.exp(1j * omega_t)

        # Polar
        polar_t = evaluate_isolated_degree_short_period(
            2, nu0, g_test, Q_t, G_t, omega_t, J2, RE, MU
        )
        polar_cart_t = polar_to_cartesian_sp(
            g_test, Psi0, Q_t, Omega0,
            polar_t["dg"], polar_t["dPsi"], polar_t["dQ"], polar_t["dOmega"]
        )

        # Equinoctial
        eqn_t = evaluate_equinoctial_sp_numerical(
            sp_funcs[2], q_t, Q_t, F_t, w_t, Omega0, scale_t
        )

        print(f"  {g_test:12.1e}  {abs(polar_cart_t['dp1']):14.6e}  "
              f"{abs(eqn_t['dp1']):14.6e}  {abs(polar_t['dPsi']):14.6e}")

    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print("=" * 72)
    print("  - If Diff ~ 1e-15 for moderate g: algebraic equivalence confirmed")
    print("  - If |dPsi| grows as g->0 but |dp1| equinoctial stays bounded:")
    print("    the equinoctial route eliminates the polar singularity")
    print("  - If SP kernel coefficients are finite at q->0:")
    print("    the symbolic expressions are genuinely regular")


if __name__ == "__main__":
    main()
