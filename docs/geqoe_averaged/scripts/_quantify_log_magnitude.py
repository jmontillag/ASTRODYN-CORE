#!/usr/bin/env python3
"""Phase 3a: Quantify the log term magnitude on the physical orbit.

Key mathematical insight: On |F|=1, the three log terms combine to:
  u1_log = scale * E1 * [log(1+qF) - log(1+q/F)]
         = scale * E1 * 2i * arctan(q*sinK / (1+q*cosK))
         = 2*(1-q^2)^3/(1+q^2) * E1 * arctan(q*sinK / (1+q*cosK))

This is a REAL, smooth, periodic function of K.

This script evaluates the log correction magnitude relative to the rational SP
for both validation cases (low-e and high-e) and for Psi and Omega.
"""
import sys, cmath, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp
from collections import defaultdict
from geqoe_mean.short_period import dimensionless_rate_series, q, Q, F, mean_f_power
from geqoe_mean.direct_residue import (
    _build_combined_numerator,
    integrate_harmonic_residue,
    _fast_cancel,
)


def series_by_m(poly):
    out = defaultdict(dict)
    for (m, k), coeff in poly.items():
        out[m][k] = coeff
    return dict(out)


def mean_rate(raw):
    out = defaultdict(lambda: sp.Integer(0))
    for (m_val, k_val), coeff in raw.items():
        out[m_val] += coeff * mean_f_power(k_val)
    result = {}
    for m, v in out.items():
        v_clean = sp.cancel(sp.together(v))
        if v_clean != 0:
            result[m] = v_clean
    return result


def compute_E1(var, n, m_val):
    """Compute E1 (log F coefficient) for one harmonic."""
    raw = dimensionless_rate_series(var, n)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    if m_val not in by_m:
        return None

    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    if N_poly == 0 or shift < 1:
        return sp.Integer(0)

    denom_expr = sp.expand(F**shift * (F + q)**2 * (1 + q*F)**2)
    N_sp = sp.Poly(N_poly, F)
    denom_sp = sp.Poly(denom_expr, F)
    if N_sp.degree() >= denom_sp.degree():
        _, N_rem_poly = sp.div(N_sp, denom_sp)
        N_rem = N_rem_poly.as_expr()
    else:
        N_rem = N_poly
    N_rem = sp.expand(N_rem)

    chi = N_rem / ((F + q)**2 * (1 + q*F)**2)
    chi_deriv = chi
    for _ in range(shift - 1):
        chi_deriv = sp.diff(chi_deriv, F)
    E1 = _fast_cancel(chi_deriv.subs(F, 0) / sp.factorial(shift - 1))
    return E1


def evaluate_log_and_rational_sp(var, n, m_val, q_val, Q_val, K_grid):
    """Evaluate log and rational SP corrections on a grid of K values."""
    raw = dimensionless_rate_series(var, n)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))

    # Rational SP (what the code computes)
    u1_expr = integrate_harmonic_residue(raw_by_k, mean_coeff)

    # E1 for the log correction
    E1_expr = compute_E1(var, n, m_val)

    # Overall scale
    scale_expr = -sp.I * (1 - q**2)**3 / (1 + q**2)

    # Substitute q, Q values
    vals = {q: q_val, Q: Q_val}
    E1_num = complex(E1_expr.subs(vals))
    scale_num = complex(scale_expr.subs(vals))

    # Evaluate on K grid
    u1_rational = np.zeros(len(K_grid), dtype=complex)
    u1_log = np.zeros(len(K_grid))

    for i, K in enumerate(K_grid):
        Fv = cmath.exp(1j * K)
        # Rational SP
        u1_rational[i] = complex(u1_expr.subs(vals).subs(F, Fv))

        # Log correction: scale * E1 * [log(1+qF) - log(1+q/F)]
        # On |F|=1: this equals scale * E1 * 2i * arctan(q*sinK / (1+q*cosK))
        phi = np.arctan2(q_val * np.sin(K), 1.0 + q_val * np.cos(K))
        u1_log[i] = (scale_num * E1_num * 2j * phi).real

    return u1_rational, u1_log


def q_from_e(e):
    """Compute q from eccentricity."""
    return e / (1.0 + np.sqrt(1.0 - e**2))


def Q_from_inc(inc_deg):
    """Compute Q = tan(i/2) from inclination in degrees."""
    return np.tan(np.radians(inc_deg) / 2.0)


def main():
    print("Phase 3a: Log term magnitude on the physical orbit")
    print("=" * 70)

    # Validation cases
    cases = [
        ("low-e",  0.05, 40.0),  # e=0.05, i=40 deg
        ("high-e", 0.65, 63.0),  # e=0.65, i=63 deg
        ("mid-e",  0.30, 40.0),  # additional test point
    ]

    K_grid = np.linspace(0, 2 * np.pi, 361)

    for case_name, e, inc_deg in cases:
        q_val = q_from_e(e)
        Q_val = Q_from_inc(inc_deg)
        print(f"\n--- {case_name}: e={e}, i={inc_deg}°, q={q_val:.6f}, Q={Q_val:.6f} ---")

        for var in ['Psi', 'Omega']:
            u1_rat, u1_log = evaluate_log_and_rational_sp(var, 2, 0, q_val, Q_val, K_grid)

            # The SP correction should be real on the orbit (it's the real part)
            u1_rat_real = u1_rat.real

            rms_rat = np.sqrt(np.mean(u1_rat_real**2))
            rms_log = np.sqrt(np.mean(u1_log**2))
            max_rat = np.max(np.abs(u1_rat_real))
            max_log = np.max(np.abs(u1_log))

            ratio_rms = rms_log / rms_rat if rms_rat > 0 else float('inf')
            ratio_max = max_log / max_rat if max_rat > 0 else float('inf')

            print(f"  {var} m=0 (n=2):")
            print(f"    rational SP:  rms={rms_rat:.6e}, max={max_rat:.6e}")
            print(f"    log term:     rms={rms_log:.6e}, max={max_log:.6e}")
            print(f"    log/rational: rms_ratio={ratio_rms:.4f}, max_ratio={ratio_max:.4f}")

            # Also show the log term as fraction of J2-scaled epsilon
            # J2 ≈ 1.08e-3, so epsilon = J2 * (RE/a)^2 ~ 1e-3 to 1e-4
            # The u1 values are dimensionless (radians)
            print(f"    log term O(q^2): q^2={q_val**2:.6f}")

    # Analytical insight
    print("\n" + "=" * 70)
    print("ANALYTICAL INSIGHT:")
    print("On |F|=1, the log terms combine to:")
    print("  u1_log = C(q,Q) * arctan(q*sin(K) / (1+q*cos(K)))")
    print("where C = 2*(1-q^2)^3/(1+q^2) * E1")
    print("")
    print("The arctan function expands as:")
    print("  arctan(...) = q*sin(K) - q^2*sin(2K)/2 + q^3*sin(3K)/3 - ...")
    print("Leading term is O(q), so u1_log = C*q*sin(K) + O(q^2)")
    print("")
    print("For the SP correction to position:")
    print("  delta_r ~ a * delta_Psi * sin(i) * [...] + a * delta_Omega * [...]")
    print("If delta_Psi_log ~ E1*q, and E1 ~ O(1) in q,")
    print("then delta_Psi_log ~ O(q) = O(e/2)")
    print("And the position error ~ a * O(e) * O(epsilon)")
    print("For small e, this is suppressed by the O(e) factor.")
    print("For large e (e=0.65, q=0.39), the suppression is weaker.")
    print("=" * 70)


if __name__ == "__main__":
    main()
