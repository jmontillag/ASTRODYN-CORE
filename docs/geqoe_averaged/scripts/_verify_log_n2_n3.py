#!/usr/bin/env python3
"""Phase 2 (fast): Check beta/D ratio and E1=beta identity at n=2 and n=3 only.

Avoids the expensive n=4,5 symbolic computations.
Also checks the beta/D = (1+q^2)/(q(1-q^2)) conjecture.
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


def full_residue_analysis(var, n, m_val):
    """Full analysis: D, B, E1, alpha, beta."""
    raw = dimensionless_rate_series(var, n)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    if m_val not in by_m:
        return None

    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    if N_poly == 0:
        return {'zero': True}

    denom_expr = sp.expand(F**shift * (F + q)**2 * (1 + q*F)**2)
    N_sp = sp.Poly(N_poly, F)
    denom_sp = sp.Poly(denom_expr, F)
    if N_sp.degree() >= denom_sp.degree():
        _, N_rem_poly = sp.div(N_sp, denom_sp)
        N_rem = N_rem_poly.as_expr()
    else:
        N_rem = N_poly
    N_rem = sp.expand(N_rem)

    m1q = -sp.Integer(1)/q

    # D at F=-1/q
    D = _fast_cancel(N_rem.subs(F, m1q) / (m1q**shift * (m1q + q)**2))
    # B at F=-q
    B = _fast_cancel(N_rem.subs(F, -q) / ((-q)**shift * (1 - q**2)**2))
    # E1
    if shift >= 1:
        chi = N_rem / ((F + q)**2 * (1 + q*F)**2)
        chi_d = chi
        for _ in range(shift - 1):
            chi_d = sp.diff(chi_d, F)
        E1 = _fast_cancel(chi_d.subs(F, 0) / sp.factorial(shift - 1))
    else:
        E1 = sp.Integer(0)
    # alpha
    psi_f = N_rem / (F**shift * (1 + q*F)**2)
    alpha = _fast_cancel(sp.diff(psi_f, F).subs(F, -q))
    # beta
    phi_f = N_rem / (q**2 * F**shift * (F + q)**2)
    beta = _fast_cancel(sp.diff(phi_f, F).subs(F, m1q))

    return {'D': D, 'B': B, 'E1': E1, 'alpha': alpha, 'beta': beta,
            'shift': shift, 'zero': False}


VARIABLES = ["g", "Q", "Psi", "Omega", "M"]


def main():
    print("Phase 2 (fast): n=2 and n=3 residue analysis")
    print("=" * 70)

    for n in [2, 3]:
        t0 = time.time()
        print(f"\n--- n = {n} ---")

        for var in VARIABLES:
            raw = dimensionless_rate_series(var, n)
            mc = mean_rate(raw)
            by_m = series_by_m(raw)

            for m_val in sorted(by_m.keys()):
                has_mean = (m_val in mc)
                if not has_mean:
                    continue  # skip zero-mean (handled by verify_bhat_residue.py)

                print(f"  {var} m={m_val} (secular rate):", end=" ", flush=True)
                info = full_residue_analysis(var, n, m_val)
                if info is None or info.get('zero'):
                    print("zero")
                    continue

                D, B, E1, alpha, beta = info['D'], info['B'], info['E1'], info['alpha'], info['beta']

                # Check identities
                e1_plus_alpha = _fast_cancel(E1 + alpha)
                e1_eq_beta = _fast_cancel(E1 - beta)
                d_eq_b = _fast_cancel(D - B)

                # Check beta/D ratio
                if D != 0:
                    ratio = _fast_cancel(beta / D)
                    expected = (1 + q**2) / (q * (1 - q**2))
                    ratio_check = _fast_cancel(ratio - expected)
                    ratio_str = "MATCH" if ratio_check == 0 else f"DIFF={ratio_check}"
                else:
                    ratio_str = "D=0"

                # Numerical values at q=0.3, Q=0.4
                vals = {q: sp.Rational(3, 10), Q: sp.Rational(4, 10)}
                D_num = complex(D.subs(vals)) if D != 0 else 0.0
                beta_num = complex(beta.subs(vals)) if beta != 0 else 0.0

                print(f"E1+α={e1_plus_alpha}, E1=β: {e1_eq_beta==0}, D=B: {d_eq_b==0}, "
                      f"β/D: {ratio_str}")
                print(f"         D={D_num:.6f}, β={beta_num:.6f}")

        elapsed = time.time() - t0
        print(f"  [n={n} took {elapsed:.1f}s]")

    print("\n" + "=" * 70)
    print("Expected: E1+alpha=0, E1=beta, D=B, beta/D = (1+q^2)/(q(1-q^2))")
    print("for ALL secular-rate harmonics at both n=2 and n=3.")
    print("=" * 70)


if __name__ == "__main__":
    main()
