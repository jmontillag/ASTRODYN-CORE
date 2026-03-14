#!/usr/bin/env python3
"""Full residue analysis at all three pole locations for each harmonic."""
import sys, cmath
sys.path.insert(0, 'docs/geqoe_averaged')
import sympy as sp
from geqoe_mean.short_period import dimensionless_rate_series, q, Q, F, mean_f_power
from geqoe_mean.direct_residue import _build_combined_numerator, _fast_cancel, _poly_eval
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


# Psi, n=2, m=0 — where mean is nonzero
var = 'Psi'
n = 2
m_val = 0

raw = dimensionless_rate_series(var, n)
mc = mean_rate(raw)
by_m = series_by_m(raw)
raw_by_k = by_m[m_val]
mean_coeff = mc.get(m_val, sp.Integer(0))

N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
scale = -sp.I * (1 - q**2)**3 / (1 + q**2)
denom_expr = sp.expand(F**shift * (F + q)**2 * (1 + q*F)**2)

N_sp = sp.Poly(N_poly, F)
denom_sp = sp.Poly(denom_expr, F)
if N_sp.degree() >= denom_sp.degree():
    _, N_rem_poly = sp.div(N_sp, denom_sp)
    N_rem = N_rem_poly.as_expr()
else:
    N_rem = N_poly
N_rem = sp.expand(N_rem)

print(f"=== {var}, n={n}, m={m_val} ===")
print(f"shift = {shift}")
print(f"deg(N_rem) = {sp.Poly(N_rem, F).degree()}")
print(f"deg(denom) = {denom_sp.degree()}")
print()

# Double-pole coefficient at F = -1/q (= D in code)
phi_m1q = _fast_cancel(N_rem.subs(F, -sp.Integer(1)/q) /
                        ((-sp.Integer(1)/q)**shift * (-sp.Integer(1)/q + q)**2))
print(f"D (double-pole at -1/q) = {_fast_cancel(phi_m1q)}")

# Simple-pole residue at F = -1/q (log coefficient)
# Res = d/dF[N_rem/(q^2 F^s (F+q)^2)] at F=-1/q
phi = N_rem / (q**2 * F**shift * (F + q)**2)
dphi = sp.diff(phi, F)
res_m1q = _fast_cancel(dphi.subs(F, -sp.Integer(1)/q))
print(f"Simple-pole res at -1/q = {res_m1q}")

# Double-pole coefficient at F = -q (= B in code)
phi_mq = _fast_cancel(N_rem.subs(F, -q) / ((-q)**shift * (1 - q**2)**2))
print(f"B (double-pole at -q) = {_fast_cancel(phi_mq)}")

# Simple-pole residue at F = -q
# Res = d/dF[N_rem/(q^2 F^s (1+qF)^2)] at F=-q
# Actually: (F+q)^2 * Integrand = N_rem/[F^s(1+qF)^2]
psi_func = N_rem / (F**shift * (1 + q*F)**2)
dpsi = sp.diff(psi_func, F)
res_mq = _fast_cancel(dpsi.subs(F, -q))
print(f"Simple-pole res at -q = {res_mq}")

# E_1 residue at F = 0 (log F coefficient)
if shift >= 1:
    # For a pole of order s at F=0, the residue (coefficient of 1/F) is
    # the Taylor coefficient of order s-1 of chi = N_rem/[(F+q)^2(1+qF)^2]
    chi = N_rem / ((F + q)**2 * (1 + q*F)**2)
    # Taylor coefficient c_{s-1} of chi at F=0
    chi_deriv = chi
    for _ in range(shift - 1):
        chi_deriv = sp.diff(chi_deriv, F)
    E1 = _fast_cancel(chi_deriv.subs(F, 0) / sp.factorial(shift - 1))
    print(f"E1 (log F coefficient) = {E1}")
else:
    E1 = sp.Integer(0)
    print(f"E1 = 0 (no pole at F=0, shift={shift})")

# Check: sum of all simple-pole residues
# E1 + res_mq + res_m1q/q should = 0 for the residue theorem
# (if the function decays at infinity)
total = _fast_cancel(scale * (E1 + res_mq + res_m1q))
print(f"\nscale*(E1 + res_mq + res_m1q) = {total}")

# Also check numerically
vals = {q: sp.Rational(3, 10), Q: sp.Rational(4, 10)}
print(f"\nNumerical at q=0.3, Q=0.4:")
print(f"  D     = {float(phi_m1q.subs(vals)):.6f}")
print(f"  B     = {float(phi_mq.subs(vals)):.6f}")
print(f"  res_mq  = {float(res_mq.subs(vals)):.6f}")
print(f"  res_m1q = {float(res_m1q.subs(vals)):.6f}")
if E1 != 0:
    print(f"  E1    = {float(E1.subs(vals)):.6f}")
print(f"  sum   = {complex(total.subs(vals))}")
