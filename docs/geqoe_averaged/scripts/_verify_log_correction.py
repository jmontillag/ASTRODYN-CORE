#!/usr/bin/env python3
"""Phase 1: Verify that adding back the dropped log-term derivatives
closes the antiderivative gap to machine precision.

The code's antiderivative u1 drops three log contributions:
  E1*log(F)        from the pole at F=0
  alpha*log(F+q)   from the pole at F=-q
  beta*log(1+qF)/q from the pole at F=-1/q  (beta = simple-pole residue)

Their derivatives are:  E1/F + alpha/(F+q) + beta*q/(1+qF)

If du1_code/dF + scale*(E1/F + alpha/(F+q) + beta*q/(1+qF)) == integrand
to machine precision, the log terms fully explain the mismatch.
"""
import sys, cmath
from pathlib import Path

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


def compute_log_residues(var, n, m_val):
    """Compute the three log-term coefficients (E1, alpha, beta) for a harmonic."""
    raw = dimensionless_rate_series(var, n)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    if m_val not in by_m:
        return None

    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)

    if N_poly == 0:
        return None

    # Polynomial long division
    denom_expr = sp.expand(F**shift * (F + q)**2 * (1 + q*F)**2)
    N_sp = sp.Poly(N_poly, F)
    denom_sp = sp.Poly(denom_expr, F)
    if N_sp.degree() >= denom_sp.degree():
        _, N_rem_poly = sp.div(N_sp, denom_sp)
        N_rem = N_rem_poly.as_expr()
    else:
        N_rem = N_poly
    N_rem = sp.expand(N_rem)

    # --- E1: residue at F=0 (coefficient of 1/F in the Laurent expansion) ---
    if shift >= 1:
        chi = N_rem / ((F + q)**2 * (1 + q*F)**2)
        chi_deriv = chi
        for _ in range(shift - 1):
            chi_deriv = sp.diff(chi_deriv, F)
        E1 = _fast_cancel(chi_deriv.subs(F, 0) / sp.factorial(shift - 1))
    else:
        E1 = sp.Integer(0)

    # --- alpha: simple-pole residue at F=-q ---
    psi_func = N_rem / (F**shift * (1 + q*F)**2)
    dpsi = sp.diff(psi_func, F)
    alpha = _fast_cancel(dpsi.subs(F, -q))

    # --- beta: simple-pole residue at F=-1/q ---
    phi_func = N_rem / (q**2 * F**shift * (F + q)**2)
    dphi = sp.diff(phi_func, F)
    beta = _fast_cancel(dphi.subs(F, -sp.Integer(1)/q))

    return {
        'E1': E1, 'alpha': alpha, 'beta': beta,
        'shift': shift, 'mean_coeff': mean_coeff,
        'raw_by_k': raw_by_k, 'N_poly': N_poly,
    }


def verify_correction(var, n, m_val, test_points):
    """Verify that log derivatives close the gap for one harmonic."""
    info = compute_log_residues(var, n, m_val)
    if info is None:
        print(f"  {var} m={m_val}: no data")
        return

    E1, alpha, beta = info['E1'], info['alpha'], info['beta']
    raw_by_k = info['raw_by_k']
    mean_coeff = info['mean_coeff']

    # Compute u1 (code's rational antiderivative)
    u1 = integrate_harmonic_residue(raw_by_k, mean_coeff)

    # Build exact integrand
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    scale = -sp.I * (1 - q**2)**3 / (1 + q**2)
    integrand = scale * N_poly / (F**shift * (F + q)**2 * (1 + q*F)**2)

    # Code's derivative
    du1_dF = sp.diff(u1, F)

    # Missing derivative from log terms
    missing = scale * (E1/F + alpha/(F + q) + beta*q/(1 + q*F))

    print(f"\n  {var} n={n} m={m_val}:")
    print(f"    E1    = {E1}")
    print(f"    alpha = {alpha}")
    print(f"    beta  = {beta}")

    # Check E1 + alpha = 0 (multi-valued log cancellation)
    sum_ea = _fast_cancel(E1 + alpha)
    print(f"    E1+alpha = {sum_ea}  (should be 0 for single-valuedness)")

    # Numerical verification at multiple points
    max_err_without = 0.0
    max_err_with = 0.0
    for qv, Qv, Fv in test_points:
        vals = {q: qv, Q: Qv, F: Fv}
        int_n = complex(integrand.subs(vals))
        du1_n = complex(du1_dF.subs(vals))
        miss_n = complex(missing.subs(vals))

        err_without = abs(du1_n - int_n) / max(abs(int_n), 1e-30)
        err_with = abs(du1_n + miss_n - int_n) / max(abs(int_n), 1e-30)
        max_err_without = max(max_err_without, err_without)
        max_err_with = max(max_err_with, err_with)

    print(f"    max rel error WITHOUT log terms: {max_err_without:.2e}")
    print(f"    max rel error WITH    log terms: {max_err_with:.2e}")

    status = "PASS" if max_err_with < 1e-12 else "FAIL"
    print(f"    => {status}")
    return max_err_with < 1e-12


def main():
    # Test points: (q, Q, F) with F on the unit circle |F|=1
    test_points = [
        (0.3, 0.4, cmath.exp(0.7j)),
        (0.3, 0.4, cmath.exp(2.1j)),
        (0.3, 0.4, cmath.exp(-1.5j)),
        (0.1, 0.2, cmath.exp(1.0j)),
        (0.5, 0.6, cmath.exp(0.3j)),
        (0.05, 0.35, cmath.exp(2.5j)),
        (0.39, 0.52, cmath.exp(-0.8j)),  # high-e case (q≈0.39)
    ]

    print("=" * 70)
    print("Phase 1: Verify log-term derivatives close the antiderivative gap")
    print("=" * 70)

    all_pass = True

    # Harmonics with nonzero mean (where log terms are dropped)
    for var, m_val in [('Psi', 0), ('Omega', 0)]:
        result = verify_correction(var, 2, m_val, test_points)
        if result is not True:
            all_pass = False

    # Sanity check: harmonics with zero mean (should already be exact)
    print("\n--- Sanity checks (zero-mean harmonics, should already be exact) ---")
    for var, m_val in [('g', 0), ('g', 2), ('Psi', 2), ('M', 0)]:
        info = compute_log_residues(var, 2, m_val)
        if info is None:
            print(f"  {var} m={m_val}: no data")
            continue
        E1, alpha, beta = info['E1'], info['alpha'], info['beta']
        print(f"  {var} m={m_val}: E1={E1}, alpha={alpha}, beta={beta}")

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: Log terms fully explain the derivative error.")
    else:
        print("RESULT: Log terms do NOT fully close the gap — investigate further.")
    print("=" * 70)


if __name__ == "__main__":
    main()
