#!/usr/bin/env python3
"""Check whether the code's antiderivative matches the integrand exactly."""
import sys, cmath
sys.path.insert(0, 'docs/geqoe_averaged')
import sympy as sp
from geqoe_mean.short_period import dimensionless_rate_series, q, Q, F, mean_f_power
from geqoe_mean.direct_residue import _build_combined_numerator, integrate_harmonic_residue
from collections import defaultdict


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


for var, m_val in [('g', 0), ('g', 2), ('Psi', 2), ('Psi', 0), ('Omega', 0), ('M', 0)]:
    raw = dimensionless_rate_series(var, 2)
    mc = mean_rate(raw)
    by_m = series_by_m(raw)
    if m_val not in by_m:
        print(f'{var} m={m_val}: no data')
        continue
    raw_by_k = by_m[m_val]
    mean_coeff = mc.get(m_val, sp.Integer(0))
    u1, _c_log = integrate_harmonic_residue(raw_by_k, mean_coeff)
    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    scale = -sp.I * (1 - q**2)**3 / (1 + q**2)
    integrand = scale * N_poly / (F**shift * (F + q)**2 * (1 + q*F)**2)
    du1_dF = sp.diff(u1, F)

    # Numerical check at q=0.3, Q=0.4, F=e^{0.7i}
    vals = {q: 0.3, Q: 0.4, F: cmath.exp(0.7j)}
    du1_n = complex(du1_dF.subs(vals))
    int_n = complex(integrand.subs(vals))
    rel_err = abs(du1_n - int_n) / max(abs(du1_n), abs(int_n), 1e-30)
    has_mean = (mean_coeff != 0)
    print(f'{var:5s} m={m_val:+d}: mean={str(has_mean):5s}, rel_err = {rel_err:.2e}')
