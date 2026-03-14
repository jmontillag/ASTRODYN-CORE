#!/usr/bin/env python3
"""Phase 2: Extend log-term verification to n=2,...,5 and check beta/D ratio.

For each zonal degree and each (variable, m) with nonzero mean coefficient:
1. Compute D (double-pole coeff at F=-1/q) and beta (simple-pole residue)
2. Check if D=0, beta=0
3. Compute beta/D ratio — expected to be (1+q^2)/(q(1-q^2)) universally
4. Verify derivative identity at test points
"""
import sys, cmath, time
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
    _poly_eval,
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


def analyze_harmonic(var, n, m_val):
    """Full analysis of one harmonic: D, B, E1, alpha, beta and ratios."""
    raw = dimensionless_rate_series(var, n)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    if m_val not in by_m:
        return None

    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))
    has_mean = (mean_coeff != 0)
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)

    if N_poly == 0:
        return {'has_mean': has_mean, 'D': sp.Integer(0), 'B': sp.Integer(0),
                'E1': sp.Integer(0), 'alpha': sp.Integer(0), 'beta': sp.Integer(0),
                'zero': True}

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

    # D: double-pole coefficient at F=-1/q
    m1q = -sp.Integer(1)/q
    phi_at = _fast_cancel(N_rem.subs(F, m1q) / (m1q**shift * (m1q + q)**2))
    D = phi_at

    # B: double-pole coefficient at F=-q
    psi_at = _fast_cancel(N_rem.subs(F, -q) / ((-q)**shift * (1 - q**2)**2))
    B = psi_at

    # E1: log(F) coefficient
    if shift >= 1:
        chi = N_rem / ((F + q)**2 * (1 + q*F)**2)
        chi_deriv = chi
        for _ in range(shift - 1):
            chi_deriv = sp.diff(chi_deriv, F)
        E1 = _fast_cancel(chi_deriv.subs(F, 0) / sp.factorial(shift - 1))
    else:
        E1 = sp.Integer(0)

    # alpha: simple-pole residue at F=-q
    psi_func = N_rem / (F**shift * (1 + q*F)**2)
    dpsi = sp.diff(psi_func, F)
    alpha = _fast_cancel(dpsi.subs(F, -q))

    # beta: simple-pole residue at F=-1/q
    phi_func = N_rem / (q**2 * F**shift * (F + q)**2)
    dphi = sp.diff(phi_func, F)
    beta = _fast_cancel(dphi.subs(F, m1q))

    return {
        'has_mean': has_mean,
        'D': D, 'B': B, 'E1': E1, 'alpha': alpha, 'beta': beta,
        'shift': shift, 'raw_by_k': raw_by_k, 'mean_coeff': mean_coeff,
        'zero': False,
    }


def check_ratio(beta, D, label):
    """Check if beta/D = (1+q^2)/(q(1-q^2))."""
    if D == 0 and beta == 0:
        return "both_zero"
    if D == 0:
        return f"D=0 but beta={beta}"

    ratio = _fast_cancel(beta / D)
    expected = (1 + q**2) / (q * (1 - q**2))
    diff = _fast_cancel(ratio - expected)
    if diff == 0:
        return "exact_match"
    else:
        # Try numerical check
        vals = {q: sp.Rational(3, 10), Q: sp.Rational(4, 10)}
        r_num = float(ratio.subs(vals))
        e_num = float(expected.subs(vals))
        if abs(r_num - e_num) < 1e-12:
            return f"numerical_match (ratio={r_num:.6f})"
        return f"MISMATCH: ratio={r_num:.6f}, expected={e_num:.6f}"


VARIABLES = ["g", "Q", "Psi", "Omega", "M"]


def main():
    n_max = 5
    print(f"Phase 2: Log-term analysis for n=2..{n_max}")
    print("=" * 80)

    test_points = [
        (0.3, 0.4, cmath.exp(0.7j)),
        (0.1, 0.2, cmath.exp(2.1j)),
        (0.39, 0.52, cmath.exp(-0.8j)),
    ]

    for n in range(2, n_max + 1):
        t0 = time.time()
        print(f"\n--- n = {n} ---")

        for var in VARIABLES:
            raw = dimensionless_rate_series(var, n)
            mc = mean_rate(raw)
            by_m = series_by_m(raw)

            for m_val in sorted(by_m.keys()):
                has_mean = (m_val in mc)
                if not has_mean:
                    # Quick check: D should be 0 for zero-mean harmonics
                    info = analyze_harmonic(var, n, m_val)
                    if info is None:
                        continue
                    D_zero = (info['D'] == 0)
                    beta_zero = (info['beta'] == 0)
                    if not D_zero or not beta_zero:
                        print(f"  UNEXPECTED: {var} m={m_val}: D_zero={D_zero}, "
                              f"beta_zero={beta_zero} (zero-mean harmonic!)")
                    continue

                # Nonzero mean: full analysis
                print(f"  {var} m={m_val} (has_mean):", end=" ", flush=True)
                info = analyze_harmonic(var, n, m_val)
                if info is None or info['zero']:
                    print("zero numerator")
                    continue

                D_zero = (info['D'] == 0)
                beta_zero = (info['beta'] == 0)
                E1_alpha = _fast_cancel(info['E1'] + info['alpha'])
                D_eq_B = _fast_cancel(info['D'] - info['B'])

                # Derivative verification
                scale = -sp.I * (1 - q**2)**3 / (1 + q**2)
                u1, _c_log = integrate_harmonic_residue(info['raw_by_k'], info['mean_coeff'])
                N_poly, shift = _build_combined_numerator(info['raw_by_k'], info['mean_coeff'])
                integrand = scale * N_poly / (F**shift * (F + q)**2 * (1 + q*F)**2)
                du1_dF = sp.diff(u1, F)
                missing = scale * (info['E1']/F + info['alpha']/(F+q) + info['beta']*q/(1+q*F))

                max_err = 0.0
                for qv, Qv, Fv in test_points:
                    vals = {q: qv, Q: Qv, F: Fv}
                    int_n = complex(integrand.subs(vals))
                    corrected = complex((du1_dF + missing).subs(vals))
                    err = abs(corrected - int_n) / max(abs(int_n), 1e-30)
                    max_err = max(max_err, err)

                ratio_result = check_ratio(info['beta'], info['D'],
                                           f"{var} n={n} m={m_val}")

                status = "PASS" if max_err < 1e-12 else "FAIL"
                print(f"D≠0={not D_zero}, β≠0={not beta_zero}, "
                      f"E1+α={E1_alpha}, D=B: {D_eq_B==0}, "
                      f"β/D: {ratio_result}, deriv: {status} ({max_err:.1e})")

        elapsed = time.time() - t0
        print(f"  [n={n} took {elapsed:.1f}s]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
