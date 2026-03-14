"""Exact first-order mixed-zonal GEqOE short-period generator.

The short-period map is formulated through the uniformly advancing generalized
mean longitude / mean anomaly:

    L = K + p1 cos(K) - p2 sin(K),   M = L - Psi.

At frozen mean state the first-order zonal rates become finite Laurent series
in

    F = exp(i f),   w = exp(i omega),   omega = Psi - Omega,

which makes the degree-wise homological equations solvable exactly in a finite
coefficient space. The practical evaluator exposed below works with the mean
state

    [nu, p1, p2, M, q1, q2]

and reconstructs the full osculating GEqOE state by first rebuilding the
osculating slow variables and osculating M/L, then solving the generalized
Kepler equation for K.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import argparse
import ast
import sys

import numpy as np
import sympy as sp


PKG_DIR = Path(__file__).resolve().parent
DOC_DIR = PKG_DIR.parent
OUT_TEX = DOC_DIR / "zonal_short_period_general.tex"
GENERATED_DATA = PKG_DIR / "generated_coefficients.py"

import math as _math

from astrodyn_core.geqoe_taylor import MU, RE
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen

from sympy.integrals.rationaltools import ratint

from .symbolic import q_from_g

# Stage B: CSE-optimized fast path for mean RHS
try:
    from .hardcoded_rates import mean_rates_cse as _mean_rates_cse
    _USE_HARDCODED = True
except ImportError:
    _USE_HARDCODED = False

try:
    from .generated_coefficients import MEAN_DATA as _MEAN_DATA_STR, SHORT_DATA as _SHORT_DATA_STR
except ImportError:
    _MEAN_DATA_STR = {}
    _SHORT_DATA_STR = {}

try:
    from .generated_coefficients import LOG_DATA as _LOG_DATA_STR
except ImportError:
    _LOG_DATA_STR = {}

try:
    from .generated_coefficients import EQNOC_SHORT_DATA as _EQNOC_SHORT_DATA_STR
except ImportError:
    _EQNOC_SHORT_DATA_STR = {}

try:
    from .generated_coefficients import EQNOC_LOG_DATA as _EQNOC_LOG_DATA_STR
except ImportError:
    _EQNOC_LOG_DATA_STR = {}

# Stage C.2: heyoka cfunc fast path for short-period corrections
try:
    from .heyoka_compiled import get_sp_cfunc as _get_sp_cfunc
    _HAS_SP_CFUNC = True
except ImportError:
    _HAS_SP_CFUNC = False


q, Q, F, w, z = sp.symbols("q Q F w z", positive=True, real=True)
I = sp.I

beta = (1 - q**2) / (1 + q**2)
g = 2 * q / (1 + q**2)
alpha = 1 / (1 + beta)
gamma = 1 + Q**2
delta = 1 - Q**2


LaurentKey = tuple[int, int]
LaurentPoly = dict[LaurentKey, sp.Expr]
HarmonicExpr = dict[int, sp.Expr]


def _clean(expr: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.together(expr))


_SYMPIFY_LOCALS = {"q": q, "Q": Q, "F": F, "I": sp.I}


def _prune(poly: LaurentPoly) -> LaurentPoly:
    out: LaurentPoly = {}
    for key, value in poly.items():
        if value == 0:
            continue
        cleaned = _clean(value)
        if cleaned != 0:
            out[key] = cleaned
    return out


def _poly_add(*polys: LaurentPoly) -> LaurentPoly:
    out: defaultdict[LaurentKey, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for poly in polys:
        for key, value in poly.items():
            out[key] += value
    return {key: value for key, value in out.items() if value != 0}


def _poly_scale(poly: LaurentPoly, scale: sp.Expr) -> LaurentPoly:
    return {key: value * scale for key, value in poly.items() if value != 0}


def _poly_mul(lhs: LaurentPoly, rhs: LaurentPoly) -> LaurentPoly:
    out: defaultdict[LaurentKey, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (m1, k1), v1 in lhs.items():
        for (m2, k2), v2 in rhs.items():
            out[(m1 + m2, k1 + k2)] += v1 * v2
    return {key: value for key, value in out.items() if value != 0}


def _poly_pow(poly: LaurentPoly, power: int) -> LaurentPoly:
    if power < 0:
        raise ValueError("Laurent polynomial power must be non-negative.")
    out: LaurentPoly = {(0, 0): sp.Integer(1)}
    for _ in range(power):
        out = _poly_mul(out, poly)
    return out


@lru_cache(maxsize=None)
def _legendre_series(n: int, derivative: bool = False) -> LaurentPoly:
    x = sp.symbols("x", real=True)
    poly_expr = sp.diff(sp.legendre(n, x), x) if derivative else sp.legendre(n, x)
    poly = sp.Poly(sp.expand(poly_expr), x)
    out: LaurentPoly = {}
    for (power,), coeff in poly.terms():
        out = _poly_add(out, _poly_scale(_poly_pow(_zhat_series(), power), coeff))
    return out


@lru_cache(maxsize=None)
def _d_inv_series() -> LaurentPoly:
    coeff_1 = (1 + q**2) * q / (1 - q**2) ** 2
    coeff_0 = (1 + q**2) ** 2 / (1 - q**2) ** 2
    return {
        (0, -1): _clean(coeff_1),
        (0, 0): _clean(coeff_0),
        (0, 1): _clean(coeff_1),
    }


@lru_cache(maxsize=None)
def _d_inv_power(power: int) -> LaurentPoly:
    return _poly_pow(_d_inv_series(), power)


@lru_cache(maxsize=None)
def _zhat_series() -> LaurentPoly:
    coeff = Q / (I * (1 + Q**2))
    return {
        (1, 1): _clean(coeff),
        (-1, -1): _clean(-coeff),
    }


@lru_cache(maxsize=None)
def _cos_u_series() -> LaurentPoly:
    return {
        (1, 1): sp.Rational(1, 2),
        (-1, -1): sp.Rational(1, 2),
    }


@lru_cache(maxsize=None)
def _sin_u_series() -> LaurentPoly:
    return {
        (1, 1): _clean(1 / (2 * I)),
        (-1, -1): _clean(-1 / (2 * I)),
    }


@lru_cache(maxsize=None)
def _f_plus_series() -> LaurentPoly:
    return {(0, -1): sp.Integer(1), (0, 1): sp.Integer(1)}


@lru_cache(maxsize=None)
def _f_minus_series() -> LaurentPoly:
    return {(0, -1): sp.Integer(-1), (0, 1): sp.Integer(1)}


def _series_by_m(poly: LaurentPoly) -> dict[int, dict[int, sp.Expr]]:
    out: dict[int, dict[int, sp.Expr]] = defaultdict(dict)
    for (m, k), coeff in poly.items():
        out[m][k] = coeff
    return dict(out)


@lru_cache(maxsize=None)
def _lambdify_laurent(poly_items: tuple[tuple[LaurentKey, sp.Expr], ...]) -> dict[LaurentKey, object]:
    out: dict[LaurentKey, object] = {}
    for key, expr in poly_items:
        out[key] = sp.lambdify((q, Q), expr, "numpy")
    return out


def _lambdify_poly(poly: LaurentPoly) -> dict[LaurentKey, object]:
    return _lambdify_laurent(tuple(sorted(poly.items())))


def _dimensionless_rate_series(variable: str, n: int) -> LaurentPoly:
    p = _legendre_series(n, derivative=False)
    dp = _legendre_series(n, derivative=True)
    d_pow_n = _d_inv_power(n)
    d_pow_np1 = _d_inv_power(n + 1)

    if variable == "g":
        return _poly_scale(
            _poly_mul(d_pow_n, _poly_mul(p, _f_minus_series())),
            (n - 1) / (2 * I * beta),
        )

    if variable == "Q":
        return _poly_scale(
            _poly_mul(d_pow_np1, _poly_mul(dp, _cos_u_series())),
            -delta / (2 * beta),
        )

    if variable == "Omega":
        return _poly_scale(
            _poly_mul(d_pow_np1, _poly_mul(dp, _sin_u_series())),
            -delta / (2 * beta * Q),
        )

    if variable == "M":
        return _poly_scale(
            _poly_mul(d_pow_n, _poly_mul(p, _f_plus_series())),
            (n - 1) / (2 * g),
        )

    if variable == "Psi":
        return _poly_add(
            _poly_scale(_poly_mul(d_pow_np1, p), -(2 * n - 1) / beta),
            _poly_scale(_poly_mul(d_pow_n, _poly_mul(p, _f_plus_series())), -(n - 1) / (2 * g * beta)),
            _poly_scale(_poly_mul(d_pow_np1, _poly_mul(_zhat_series(), dp)), -delta / (2 * beta)),
        )

    raise ValueError(f"Unsupported variable: {variable}")


@lru_cache(maxsize=None)
def dimensionless_rate_series(variable: str, n: int) -> LaurentPoly:
    return _prune(_dimensionless_rate_series(variable, n))


# ---------------------------------------------------------------------------
# Equinoctial (complex) reduced rate series
# ---------------------------------------------------------------------------

_W_SHIFT: LaurentPoly = {(1, 0): sp.Integer(1)}


@lru_cache(maxsize=None)
def zeta_reduced_series(n: int) -> LaurentPoly:
    """zeta_dot_red/(nu*lambda_n) = [g_dot + i*g*Psi_dot] * w.

    The g*(1/g) cancellation in the Psi term makes this regular at g -> 0.
    """
    g_s = dimensionless_rate_series("g", n)
    psi_s = dimensionless_rate_series("Psi", n)
    combined = _poly_add(g_s, _poly_scale(psi_s, I * g))
    return _prune(_poly_mul(combined, _W_SHIFT))


@lru_cache(maxsize=None)
def eta_reduced_series(n: int) -> LaurentPoly:
    """eta_dot_red/(nu*lambda_n) = Q_dot + i*Q*Omega_dot."""
    Q_s = dimensionless_rate_series("Q", n)
    omega_s = dimensionless_rate_series("Omega", n)
    return _prune(_poly_add(Q_s, _poly_scale(omega_s, I * Q)))


@lru_cache(maxsize=None)
def mean_f_power(k: int) -> sp.Expr:
    F_z = (z - q) / (1 - q * z)
    D_z = ((z - q) * (1 - q * z)) / ((1 + q**2) * z)
    integrand = _clean(D_z * F_z**k / z)
    poles = [sp.Integer(0)]
    if k < 0:
        poles.append(q)
    return _clean(sum(sp.residue(integrand, z, pole) for pole in poles))


def _mean_rate_from_raw_series(raw: LaurentPoly) -> dict[int, sp.Expr]:
    out: defaultdict[int, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (m_val, k_val), coeff in raw.items():
        out[m_val] += coeff * mean_f_power(k_val)
    return {m_val: _clean(value) for m_val, value in out.items() if value != 0}


def _compute_isolated_mean_laurent_coefficients(n: int) -> dict[str, dict[int, sp.Expr]]:
    return {
        "g": _mean_rate_from_raw_series(dimensionless_rate_series("g", n)),
        "Q": _mean_rate_from_raw_series(dimensionless_rate_series("Q", n)),
        "Psi": _mean_rate_from_raw_series(dimensionless_rate_series("Psi", n)),
        "Omega": _mean_rate_from_raw_series(dimensionless_rate_series("Omega", n)),
        "M": _mean_rate_from_raw_series(dimensionless_rate_series("M", n)),
    }


def _compute_isolated_short_period_expressions(variable: str, n: int) -> HarmonicExpr:
    # Use the direct residue method which avoids log-term crashes and
    # expression swell from the term-by-term ratint approach.
    try:
        from .direct_residue import integrate_harmonic_residue
        raw = dimensionless_rate_series(variable, n)
        mean_coeffs = _mean_rate_from_raw_series(raw)
        solved: HarmonicExpr = {}
        for m_val, raw_by_k in _series_by_m(raw).items():
            rational, _c_log = integrate_harmonic_residue(
                raw_by_k, mean_coeffs.get(m_val, sp.Integer(0))
            )
            solved[m_val] = rational
        # Skip _clean: integrate_harmonic_residue already simplifies each term.
        # Calling _clean (sp.together + sp.cancel) would combine all structural
        # terms over a common denominator, which is catastrophically slow for
        # higher zonal degrees.
        return {m_val: expr for m_val, expr in solved.items() if expr != 0}
    except ImportError:
        # Fallback to the original ratint approach (may fail for Psi/Omega)
        raw = dimensionless_rate_series(variable, n)
        mean_coeffs = _mean_rate_from_raw_series(raw)
        solved: HarmonicExpr = {}
        for m_val, raw_by_k in _series_by_m(raw).items():
            solved[m_val] = _integrate_periodic_expression(raw_by_k, mean_coeffs.get(m_val, sp.Integer(0)))
        return {m_val: _clean(expr) for m_val, expr in solved.items() if expr != 0}


@lru_cache(maxsize=None)
def _generated_mean_coefficients(n: int) -> dict[str, dict[int, sp.Expr]] | None:
    raw = _MEAN_DATA_STR.get(n)
    if raw is None:
        return None
    return {
        variable: {
            int(m_key): sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
            for m_key, expr_str in coeffs.items()
        }
        for variable, coeffs in raw.items()
    }


@lru_cache(maxsize=None)
def _generated_short_period_expressions(variable: str, n: int) -> HarmonicExpr | None:
    raw_n = _SHORT_DATA_STR.get(n)
    if raw_n is None:
        return None
    raw_variable = raw_n.get(variable)
    if raw_variable is None or not raw_variable:
        return None
    out: HarmonicExpr = {}
    for key_str, expr_str in raw_variable.items():
        # Skip _clean: the generated data is already in canonical form.
        # Calling _clean (sp.cancel + sp.together) on large parsed expressions
        # is catastrophically slow and unnecessary.
        out[int(key_str)] = sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
    return out


@lru_cache(maxsize=None)
def isolated_mean_laurent_coefficients(n: int) -> dict[str, dict[int, sp.Expr]]:
    generated = _generated_mean_coefficients(n)
    if generated is not None:
        return generated
    return _compute_isolated_mean_laurent_coefficients(n)


@lru_cache(maxsize=None)
def _dM_dF() -> sp.Expr:
    return _clean(-I * (1 - q**2) ** 3 / (1 + q**2) * F / ((F + q) ** 2 * (1 + q * F) ** 2))


@lru_cache(maxsize=None)
def _primitive_f_power(k_val: int) -> sp.Expr:
    return _clean(ratint(sp.together(F**k_val * _dM_dF()), F))


def _mean_of_f_expression(expr_f: sp.Expr) -> sp.Expr:
    F_z = (z - q) / (1 - q * z)
    D_z = ((z - q) * (1 - q * z)) / ((1 + q**2) * z)
    integrand = _clean(D_z * expr_f.subs(F, F_z) / z)
    residues = [_clean(sp.residue(integrand, z, sp.Integer(0)))]
    if integrand.has(q):
        residues.append(_clean(sp.residue(integrand, z, q)))
    return _clean(sum(residues))


@lru_cache(maxsize=None)
def _mean_primitive_f_power(k_val: int) -> sp.Expr:
    return _mean_of_f_expression(_primitive_f_power(k_val))


def _integrate_periodic_expression(raw_by_k: dict[int, sp.Expr], mean_coeff: sp.Expr) -> sp.Expr:
    primitive = sum(coeff * _primitive_f_power(k_val) for k_val, coeff in raw_by_k.items())
    avg = sum(coeff * _mean_primitive_f_power(k_val) for k_val, coeff in raw_by_k.items())
    if mean_coeff != 0:
        primitive -= mean_coeff * _primitive_f_power(0)
        avg -= mean_coeff * _mean_primitive_f_power(0)
    return _clean(primitive - avg)


@lru_cache(maxsize=None)
def isolated_short_period_expressions_for(variable: str, n: int) -> HarmonicExpr:
    generated = _generated_short_period_expressions(variable, n)
    if generated is not None:
        return generated
    return _compute_isolated_short_period_expressions(variable, n)


@lru_cache(maxsize=None)
def isolated_short_period_coefficients_for(variable: str, n: int) -> LaurentPoly:
    exprs = isolated_short_period_expressions_for(variable, n)
    out: LaurentPoly = {}
    for m_val, expr in exprs.items():
        rational = sp.cancel(sp.together(sp.expand(expr)))
        num, den = sp.fraction(rational)
        den_factors = den.as_powers_dict()
        f_power = int(den_factors.pop(F, 0))
        den_coeff = sp.Integer(1)
        for factor, power in den_factors.items():
            den_coeff *= factor**power
        poly_num = sp.Poly(sp.expand(num), F)
        for (power,), coeff in poly_num.terms():
            out[(m_val, power - f_power)] = _clean(coeff / den_coeff)
    return _prune(out)


@lru_cache(maxsize=None)
def isolated_short_period_coefficients(n: int) -> dict[str, LaurentPoly]:
    return {
        variable: isolated_short_period_coefficients_for(variable, n)
        for variable in ("g", "Q", "Psi", "Omega", "M")
    }


def isolated_short_period_support(n: int) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for variable in ("g", "Q", "Psi", "Omega", "M"):
        exprs = isolated_short_period_expressions_for(variable, n)
        raw = dimensionless_rate_series(variable, n)
        m_keys = [abs(m_val) for m_val in exprs]
        k_keys = [abs(k_val) for (_, k_val) in raw]
        out[variable] = {
            "m_max": max(m_keys) if m_keys else 0,
            "k_raw_max": max(k_keys) if k_keys else 0,
            "terms": len(exprs),
        }
    return out


def _evaluate_mean_laurent(coeffs: dict[int, object], q_val: float, Q_val: float, w_val: complex) -> float:
    total = 0.0j
    for m_val, func in coeffs.items():
        total += complex(func(q_val, Q_val)) * (w_val**m_val)
    return float(total.real)


@lru_cache(maxsize=None)
def _lambdified_mean_coefficients(n: int) -> dict[str, dict[int, object]]:
    out: dict[str, dict[int, object]] = {}
    for variable, coeffs in isolated_mean_laurent_coefficients(n).items():
        out[variable] = {
            m_val: sp.lambdify((q, Q), expr, "numpy")
            for m_val, expr in coeffs.items()
        }
    return out


@lru_cache(maxsize=None)
def _lambdified_short_period_expressions(n: int) -> dict[str, dict[int, object]]:
    return {
        variable: {
            m_val: sp.lambdify((q, Q, F), expr, "numpy")
            for m_val, expr in isolated_short_period_expressions_for(variable, n).items()
        }
        for variable in ("g", "Q", "Psi", "Omega", "M")
    }


@lru_cache(maxsize=None)
def _lambdified_log_coefficients(n: int) -> dict[str, dict[int, object]]:
    """Parse and lambdify log coefficients from LOG_DATA for degree n."""
    raw_n = _LOG_DATA_STR.get(n)
    if raw_n is None:
        return {}
    out: dict[str, dict[int, object]] = {}
    for variable in ("g", "Q", "Psi", "Omega", "M"):
        raw_var = raw_n.get(variable)
        if not raw_var:
            continue
        funcs = {}
        for m_str, expr_str in raw_var.items():
            c_sp = sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
            if c_sp != 0:
                funcs[int(m_str)] = sp.lambdify((q, Q), c_sp, "numpy")
        if funcs:
            out[variable] = funcs
    return out


@lru_cache(maxsize=None)
def _lambdified_equinoctial_expressions(n: int) -> dict[str, dict[int, object]] | None:
    """Parse and lambdify equinoctial SP expressions from EQNOC_SHORT_DATA."""
    raw_n = _EQNOC_SHORT_DATA_STR.get(n)
    if raw_n is None:
        return None
    out: dict[str, dict[int, object]] = {}
    for channel in ("zeta", "eta"):
        raw_ch = raw_n.get(channel, {})
        if not raw_ch:
            continue
        funcs = {}
        for m_str, expr_str in raw_ch.items():
            sp_expr = sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
            if sp_expr != 0:
                funcs[int(m_str)] = sp.lambdify((q, Q, F), sp_expr, "numpy")
        if funcs:
            out[channel] = funcs
    return out if out else None


@lru_cache(maxsize=None)
def _lambdified_equinoctial_log_coefficients(n: int) -> dict[str, dict[int, object]]:
    """Parse and lambdify equinoctial log coefficients from EQNOC_LOG_DATA."""
    raw_n = _EQNOC_LOG_DATA_STR.get(n)
    if raw_n is None:
        return {}
    out: dict[str, dict[int, object]] = {}
    for channel in ("zeta", "eta"):
        raw_ch = raw_n.get(channel, {})
        if not raw_ch:
            continue
        funcs = {}
        for m_str, expr_str in raw_ch.items():
            c_sp = sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
            if c_sp != 0:
                funcs[int(m_str)] = sp.lambdify((q, Q), c_sp, "numpy")
        if funcs:
            out[channel] = funcs
    return out


def _complex_f_from_g(g_val: float, G_val: float) -> complex:
    q_val = q_from_g(g_val)
    z_val = np.exp(1j * G_val)
    return (z_val - q_val) / (1.0 - q_val * z_val)


def evaluate_equinoctial_short_period(
    nu_val: float,
    p1_val: float,
    p2_val: float,
    q1_val: float,
    q2_val: float,
    G_val: float,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> dict[str, float]:
    """Evaluate SP corrections directly in equinoctial components.

    Returns {dp1, dp2, dq1, dq2, dM} — no polar (g, Psi, Q, Omega) round-trip.
    """
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    q_val = q_from_g(g_val)
    F_val = _complex_f_from_g(g_val, G_val)
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)

    # Compute exp(i*Omega) and w = exp(i*omega) from Cartesian components
    # (no atan2 needed — numerically stable at Q -> 0 and g -> 0)
    if Q_val > 0:
        exp_iOmega = (q2_val + 1j * q1_val) / Q_val
    else:
        exp_iOmega = 1.0 + 0j
    if g_val > 0:
        zeta_mean = p2_val + 1j * p1_val  # g * exp(i*Psi)
        w_val = zeta_mean / (g_val * exp_iOmega)  # exp(i*omega)
    else:
        w_val = 1.0 + 0j

    phi = np.arctan2(q_val * F_val.imag, 1.0 + q_val * F_val.real)

    dp1 = dp2 = dq1 = dq2 = dM_total = 0.0

    for n, jn_val in sorted(j_coeffs.items()):
        scale = jn_val * (re_val / a_val) ** n

        eqnoc = _lambdified_equinoctial_expressions(n)
        eqnoc_log = _lambdified_equinoctial_log_coefficients(n)

        if eqnoc is not None:
            # Zeta channel -> dp1, dp2
            if "zeta" in eqnoc:
                dzeta_red = 0.0j
                for m_val, func in eqnoc["zeta"].items():
                    dzeta_red += complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
                if "zeta" in eqnoc_log:
                    for m_val, c_log_func in eqnoc_log["zeta"].items():
                        dzeta_red += complex(c_log_func(q_val, Q_val)) * phi * (w_val ** m_val)
                dzeta = scale * dzeta_red * exp_iOmega
                dp1 += dzeta.imag
                dp2 += dzeta.real

            # Eta channel -> dq1, dq2
            if "eta" in eqnoc:
                deta_red = 0.0j
                for m_val, func in eqnoc["eta"].items():
                    deta_red += complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
                if "eta" in eqnoc_log:
                    for m_val, c_log_func in eqnoc_log["eta"].items():
                        deta_red += complex(c_log_func(q_val, Q_val)) * phi * (w_val ** m_val)
                deta = scale * deta_red * exp_iOmega
                dq1 += deta.imag
                dq2 += deta.real

        # M channel: reuse polar route (no polar singularity in M)
        polar_M = _lambdified_short_period_expressions(n).get("M", {})
        polar_M_log = _lambdified_log_coefficients(n).get("M", {})
        dM_contrib = 0.0j
        for m_val, func in polar_M.items():
            dM_contrib += complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
        for m_val, c_log_func in polar_M_log.items():
            dM_contrib += complex(c_log_func(q_val, Q_val)) * phi * (w_val ** m_val)
        dM_total += scale * float(dM_contrib.real)

    return {"dp1": dp1, "dp2": dp2, "dq1": dq1, "dq2": dq2, "dM": dM_total}


def evaluate_isolated_degree_mean_rates(
    n: int,
    nu_val: float,
    g_val: float,
    Q_val: float,
    omega_val: float,
    jn_val: float,
    re_val: float,
    mu_val: float,
) -> dict[str, float]:
    q_val = q_from_g(g_val)
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
    scale = jn_val * (re_val / a_val) ** n
    w_val = np.exp(1j * omega_val)
    coeffs = _lambdified_mean_coefficients(n)

    out = {}
    for variable in ("g", "Q", "Psi", "Omega", "M"):
        out[f"{variable}_dot"] = nu_val * scale * _evaluate_mean_laurent(coeffs[variable], q_val, Q_val, w_val)
    return out


def evaluate_isolated_degree_short_period(
    n: int,
    nu_val: float,
    g_val: float,
    Q_val: float,
    G_val: float,
    omega_val: float,
    jn_val: float,
    re_val: float,
    mu_val: float,
) -> dict[str, float]:
    q_val = q_from_g(g_val)
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
    scale = jn_val * (re_val / a_val) ** n
    F_val = _complex_f_from_g(g_val, G_val)
    w_val = np.exp(1j * omega_val)
    coeffs = _lambdified_short_period_expressions(n)
    log_coeffs = _lambdified_log_coefficients(n)

    # φ = arctan2(q·Im(F), 1 + q·Re(F)) — shared by all log contributions
    phi = np.arctan2(q_val * F_val.imag, 1.0 + q_val * F_val.real)

    out = {}
    for variable in ("g", "Q", "Psi", "Omega", "M"):
        total = 0.0j
        for m_val, func in coeffs[variable].items():
            total += complex(func(q_val, Q_val, F_val)) * (w_val**m_val)
        # Add log contributions
        if variable in log_coeffs:
            for m_val, c_log_func in log_coeffs[variable].items():
                total += complex(c_log_func(q_val, Q_val)) * phi * (w_val ** m_val)
        out[f"d{variable}"] = float((scale * total).real)
    return out


def evaluate_truncated_mean_rates(
    nu_val: float,
    g_val: float,
    Q_val: float,
    omega_val: float,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> dict[str, float]:
    totals = {f"{name}_dot": 0.0 for name in ("g", "Q", "Psi", "Omega", "M")}
    for n, jn_val in sorted(j_coeffs.items()):
        contrib = evaluate_isolated_degree_mean_rates(
            n,
            nu_val=nu_val,
            g_val=g_val,
            Q_val=Q_val,
            omega_val=omega_val,
            jn_val=jn_val,
            re_val=re_val,
            mu_val=mu_val,
        )
        for key in totals:
            totals[key] += contrib[key]
    return totals


def evaluate_truncated_short_period(
    nu_val: float,
    g_val: float,
    Q_val: float,
    G_val: float,
    omega_val: float,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> dict[str, float]:
    totals = {f"d{name}": 0.0 for name in ("g", "Q", "Psi", "Omega", "M")}
    for n, jn_val in sorted(j_coeffs.items()):
        contrib = evaluate_isolated_degree_short_period(
            n,
            nu_val=nu_val,
            g_val=g_val,
            Q_val=Q_val,
            G_val=G_val,
            omega_val=omega_val,
            jn_val=jn_val,
            re_val=re_val,
            mu_val=mu_val,
        )
        for key in totals:
            totals[key] += contrib[key]
    return totals


def _fast_mean_rhs_pqm(
    state_pqm: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float,
    mu_val: float,
) -> np.ndarray:
    """CSE-optimized mean RHS using pure Python math (no numpy overhead)."""
    nu_val = float(state_pqm[0])
    p1_val = float(state_pqm[1])
    p2_val = float(state_pqm[2])
    q1_val = float(state_pqm[4])
    q2_val = float(state_pqm[5])

    g_val = _math.hypot(p1_val, p2_val)
    Q_val = _math.hypot(q1_val, q2_val)
    Psi_val = _math.atan2(p1_val, p2_val)
    Omega_val = _math.atan2(q1_val, q2_val)
    omega_val = Psi_val - Omega_val

    # q from g: q = g / (1 + sqrt(1 - g^2))
    beta_val = _math.sqrt(max(1.0 - g_val * g_val, 0.0))
    q_val = g_val / (1.0 + beta_val)

    # Scale factors: Jn * (Re/a)^n
    a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
    ra = re_val / a_val
    ra2 = ra * ra
    s2 = j_coeffs[2] * ra2
    s3 = j_coeffs[3] * ra2 * ra
    s4 = j_coeffs[4] * ra2 * ra2
    s5 = j_coeffs[5] * ra2 * ra2 * ra

    g_dot, Q_dot, Psi_dot, Omega_dot, M_dot = _mean_rates_cse(
        q_val, Q_val, omega_val, s2, s3, s4, s5)

    # Scale by nu (CSE returns dimensionless rates)
    g_dot *= nu_val
    Q_dot *= nu_val
    Psi_dot *= nu_val
    Omega_dot *= nu_val
    M_dot *= nu_val

    sinPsi = _math.sin(Psi_val)
    cosPsi = _math.cos(Psi_val)
    sinOmega = _math.sin(Omega_val)
    cosOmega = _math.cos(Omega_val)

    p1_dot = g_dot * sinPsi + g_val * cosPsi * Psi_dot
    p2_dot = g_dot * cosPsi - g_val * sinPsi * Psi_dot
    q1_dot = Q_dot * sinOmega + Q_val * cosOmega * Omega_dot
    q2_dot = Q_dot * cosOmega - Q_val * sinOmega * Omega_dot

    return np.array([0.0, p1_dot, p2_dot, nu_val + M_dot, q1_dot, q2_dot])


def evaluate_truncated_mean_rhs_pqm(
    state_pqm: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    # Fast path: CSE-optimized when all 4 zonal degrees are present
    if _USE_HARDCODED and set(j_coeffs.keys()) == {2, 3, 4, 5}:
        return _fast_mean_rhs_pqm(state_pqm, j_coeffs, re_val, mu_val)

    nu_val, p1_val, p2_val, M_val, q1_val, q2_val = map(float, state_pqm)
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    Psi_val = float(np.arctan2(p1_val, p2_val))
    Omega_val = float(np.arctan2(q1_val, q2_val))
    omega_val = Psi_val - Omega_val

    rates = evaluate_truncated_mean_rates(
        nu_val=nu_val,
        g_val=g_val,
        Q_val=Q_val,
        omega_val=omega_val,
        j_coeffs=j_coeffs,
        re_val=re_val,
        mu_val=mu_val,
    )

    p1_dot = rates["g_dot"] * np.sin(Psi_val) + g_val * np.cos(Psi_val) * rates["Psi_dot"]
    p2_dot = rates["g_dot"] * np.cos(Psi_val) - g_val * np.sin(Psi_val) * rates["Psi_dot"]
    q1_dot = rates["Q_dot"] * np.sin(Omega_val) + Q_val * np.cos(Omega_val) * rates["Omega_dot"]
    q2_dot = rates["Q_dot"] * np.cos(Omega_val) - Q_val * np.sin(Omega_val) * rates["Omega_dot"]

    # M = L - Psi retains the Keplerian advance nu plus the averaged zonal drift.
    return np.array([0.0, p1_dot, p2_dot, nu_val + rates["M_dot"], q1_dot, q2_dot], dtype=float)


def osculating_to_mean_state(
    state_osc: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    nu_val, p1_val, p2_val, K_val, q1_val, q2_val = map(float, state_osc)
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    Psi_val = float(np.arctan2(p1_val, p2_val))
    Omega_val = float(np.arctan2(q1_val, q2_val))
    omega_val = Psi_val - Omega_val
    M_val = float(K_to_L(K_val, p1_val, p2_val) - Psi_val)
    G_val = float(K_val - Psi_val)

    corr = evaluate_truncated_short_period(
        nu_val=nu_val,
        g_val=g_val,
        Q_val=Q_val,
        G_val=G_val,
        omega_val=omega_val,
        j_coeffs=j_coeffs,
        re_val=re_val,
        mu_val=mu_val,
    )

    g_mean = g_val - corr["dg"]
    Psi_mean = Psi_val - corr["dPsi"]
    Q_mean = Q_val - corr["dQ"]
    Omega_mean = Omega_val - corr["dOmega"]
    M_mean = M_val - corr["dM"]

    return np.array(
        [
            nu_val,
            g_mean * np.sin(Psi_mean),
            g_mean * np.cos(Psi_mean),
            M_mean,
            Q_mean * np.sin(Omega_mean),
            Q_mean * np.cos(Omega_mean),
        ],
        dtype=float,
    )


def mean_to_osculating_state(
    state_mean: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    nu_val, p1_val, p2_val, M_val, q1_val, q2_val = map(float, state_mean)
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    Psi_val = float(np.arctan2(p1_val, p2_val))
    Omega_val = float(np.arctan2(q1_val, q2_val))
    omega_val = Psi_val - Omega_val
    L_mean = Psi_val + M_val
    K_mean = float(solve_kepler_gen(L_mean, p1_val, p2_val))
    G_mean = float(K_mean - Psi_val)

    corr = evaluate_truncated_short_period(
        nu_val=nu_val,
        g_val=g_val,
        Q_val=Q_val,
        G_val=G_mean,
        omega_val=omega_val,
        j_coeffs=j_coeffs,
        re_val=re_val,
        mu_val=mu_val,
    )

    g_osc = g_val + corr["dg"]
    Psi_osc = Psi_val + corr["dPsi"]
    Q_osc = Q_val + corr["dQ"]
    Omega_osc = Omega_val + corr["dOmega"]
    M_osc = M_val + corr["dM"]

    p1_osc = g_osc * np.sin(Psi_osc)
    p2_osc = g_osc * np.cos(Psi_osc)
    q1_osc = Q_osc * np.sin(Omega_osc)
    q2_osc = Q_osc * np.cos(Omega_osc)
    L_osc = Psi_osc + M_osc
    K_osc = float(solve_kepler_gen(L_osc, p1_osc, p2_osc))

    return np.array([nu_val, p1_osc, p2_osc, K_osc, q1_osc, q2_osc], dtype=float)


# --------------------------------------------------------------------------- #
#  Equinoctial (singularity-free) mean <-> osculating transformations
# --------------------------------------------------------------------------- #

def osculating_to_mean_state_equinoctial(
    state_osc: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    """Osculating -> mean using equinoctial SP (singularity-free at g=0, Q=0).

    Slow variables use equinoctial corrections. Fast phase (M) uses polar
    dPsi+dM to avoid ill-conditioned atan2 at g -> 0.
    """
    nu_val, p1_val, p2_val, K_val, q1_val, q2_val = map(float, state_osc)
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    Psi_val = float(np.arctan2(p1_val, p2_val))
    Omega_val = float(np.arctan2(q1_val, q2_val))
    omega_val = Psi_val - Omega_val
    M_val = float(K_to_L(K_val, p1_val, p2_val) - Psi_val)
    G_val = float(K_val - Psi_val)

    # Equinoctial corrections for slow variables
    corr_eq = evaluate_equinoctial_short_period(
        nu_val, p1_val, p2_val, q1_val, q2_val, G_val,
        j_coeffs, re_val, mu_val,
    )
    p1_mean = p1_val - corr_eq["dp1"]
    p2_mean = p2_val - corr_eq["dp2"]
    q1_mean = q1_val - corr_eq["dq1"]
    q2_mean = q2_val - corr_eq["dq2"]

    # Polar corrections for fast phase
    corr_polar = evaluate_truncated_short_period(
        nu_val, g_val, Q_val, G_val, omega_val,
        j_coeffs, re_val, mu_val,
    )
    M_mean = M_val - corr_polar["dPsi"] - corr_polar["dM"]
    # M_mean = L_mean - Psi_mean. Since L_osc = L_mean + dPsi + dM,
    # we have L_mean = L_osc - dPsi - dM = (Psi + M) - dPsi - dM.
    # So M_mean (as stored) = L_mean - Psi_mean_from_p1p2.
    # But we define M_mean = (Psi_osc + M_osc) - (dPsi + dM) - Psi_mean,
    # where Psi_mean = atan2(p1_mean, p2_mean). This introduces the atan2
    # extraction we're trying to avoid. Instead, store M_mean consistently:
    Psi_mean = float(np.arctan2(p1_mean, p2_mean))
    L_osc = Psi_val + M_val
    L_mean = L_osc - corr_polar["dPsi"] - corr_polar["dM"]
    M_mean = L_mean - Psi_mean

    return np.array([nu_val, p1_mean, p2_mean, M_mean, q1_mean, q2_mean], dtype=float)


def mean_to_osculating_state_equinoctial(
    state_mean: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    """Mean -> osculating using equinoctial SP (singularity-free at g=0, Q=0).

    Slow variables (p1,p2,q1,q2) use equinoctial corrections (no polar
    singularity). Fast phase L uses polar dPsi+dM (avoids ill-conditioned
    atan2 extraction from small p1,p2).
    """
    nu_val, p1_val, p2_val, M_val, q1_val, q2_val = map(float, state_mean)
    g_val = float(np.hypot(p1_val, p2_val))
    Q_val = float(np.hypot(q1_val, q2_val))
    Psi_val = float(np.arctan2(p1_val, p2_val))
    Omega_val = float(np.arctan2(q1_val, q2_val))
    omega_val = Psi_val - Omega_val
    L_mean = Psi_val + M_val
    K_mean = float(solve_kepler_gen(L_mean, p1_val, p2_val))
    G_mean = float(K_mean - Psi_val)

    # Equinoctial corrections for slow variables
    corr_eq = evaluate_equinoctial_short_period(
        nu_val, p1_val, p2_val, q1_val, q2_val, G_mean,
        j_coeffs, re_val, mu_val,
    )
    p1_osc = p1_val + corr_eq["dp1"]
    p2_osc = p2_val + corr_eq["dp2"]
    q1_osc = q1_val + corr_eq["dq1"]
    q2_osc = q2_val + corr_eq["dq2"]

    # Polar corrections for fast phase (dPsi + dM -> L_osc)
    corr_polar = evaluate_truncated_short_period(
        nu_val, g_val, Q_val, G_mean, omega_val,
        j_coeffs, re_val, mu_val,
    )
    L_osc = L_mean + corr_polar["dPsi"] + corr_polar["dM"]
    K_osc = float(solve_kepler_gen(L_osc, p1_osc, p2_osc))

    return np.array([nu_val, p1_osc, p2_osc, K_osc, q1_osc, q2_osc], dtype=float)


def mean_to_osculating_state_equinoctial_batch(
    mean_states: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    """Vectorized mean -> osculating using equinoctial SP.

    Parameters
    ----------
    mean_states : (N, 6) array of [nu, p1, p2, M, q1, q2]
    j_coeffs : {degree: Jn_value}

    Returns
    -------
    (N, 6) array of osculating [nu, p1, p2, K, q1, q2]
    """
    N = len(mean_states)
    nu_arr = mean_states[:, 0]
    p1_arr = mean_states[:, 1]
    p2_arr = mean_states[:, 2]
    M_arr = mean_states[:, 3]
    q1_arr = mean_states[:, 4]
    q2_arr = mean_states[:, 5]

    g_arr = np.hypot(p1_arr, p2_arr)
    Q_arr = np.hypot(q1_arr, q2_arr)
    Psi_arr = np.arctan2(p1_arr, p2_arr)

    L_mean = Psi_arr + M_arr
    K_mean = solve_kepler_gen(L_mean, p1_arr, p2_arr)
    G_mean = K_mean - Psi_arr

    q_arr = _q_from_g_arr(g_arr)
    F_arr = _complex_f_from_g_arr(g_arr, G_mean)
    a_arr = (mu_val / (nu_arr * nu_arr)) ** (1.0 / 3.0)

    # Compute exp(i*Omega) and w from Cartesian components (no atan2)
    Q_safe = np.where(Q_arr > 0, Q_arr, 1.0)
    exp_iOmega = (q2_arr + 1j * q1_arr) / Q_safe
    g_safe = np.where(g_arr > 0, g_arr, 1.0)
    zeta_mean = p2_arr + 1j * p1_arr
    w_arr = zeta_mean / (g_safe * exp_iOmega)
    # Fix degenerate cases
    w_arr = np.where(g_arr > 0, w_arr, 1.0 + 0j)
    exp_iOmega = np.where(Q_arr > 0, exp_iOmega, 1.0 + 0j)

    phi_arr = np.arctan2(q_arr * np.imag(F_arr), 1.0 + q_arr * np.real(F_arr))

    dp1 = np.zeros(N)
    dp2 = np.zeros(N)
    dq1 = np.zeros(N)
    dq2 = np.zeros(N)
    dM = np.zeros(N)

    for n, jn_val in sorted(j_coeffs.items()):
        scale_arr = jn_val * (re_val / a_arr) ** n

        eqnoc = _lambdified_equinoctial_expressions(n)
        eqnoc_log = _lambdified_equinoctial_log_coefficients(n)

        if eqnoc is not None:
            # Zeta channel
            if "zeta" in eqnoc:
                dzeta_red = np.zeros(N, dtype=complex)
                for m_val, func in eqnoc["zeta"].items():
                    dzeta_red += func(q_arr, Q_arr, F_arr) * (w_arr ** m_val)
                if "zeta" in eqnoc_log:
                    for m_val, c_log_func in eqnoc_log["zeta"].items():
                        dzeta_red += c_log_func(q_arr, Q_arr) * phi_arr * (w_arr ** m_val)
                dzeta = scale_arr * dzeta_red * exp_iOmega
                dp1 += np.imag(dzeta)
                dp2 += np.real(dzeta)

            # Eta channel
            if "eta" in eqnoc:
                deta_red = np.zeros(N, dtype=complex)
                for m_val, func in eqnoc["eta"].items():
                    deta_red += func(q_arr, Q_arr, F_arr) * (w_arr ** m_val)
                if "eta" in eqnoc_log:
                    for m_val, c_log_func in eqnoc_log["eta"].items():
                        deta_red += c_log_func(q_arr, Q_arr) * phi_arr * (w_arr ** m_val)
                deta = scale_arr * deta_red * exp_iOmega
                dq1 += np.imag(deta)
                dq2 += np.real(deta)

        # M channel: polar route
        polar_M = _lambdified_short_period_expressions(n).get("M", {})
        polar_M_log = _lambdified_log_coefficients(n).get("M", {})
        dM_c = np.zeros(N, dtype=complex)
        for m_val, func in polar_M.items():
            dM_c += func(q_arr, Q_arr, F_arr) * (w_arr ** m_val)
        for m_val, c_log_func in polar_M_log.items():
            dM_c += c_log_func(q_arr, Q_arr) * phi_arr * (w_arr ** m_val)
        dM += scale_arr * np.real(dM_c)

    # Slow variables from equinoctial corrections
    p1_osc = p1_arr + dp1
    p2_osc = p2_arr + dp2
    q1_osc = q1_arr + dq1
    q2_osc = q2_arr + dq2

    # Fast phase L from polar dPsi + dM (avoids ill-conditioned atan2 at g->0)
    omega_arr = Psi_arr - np.arctan2(q1_arr, q2_arr)
    polar_corr = evaluate_truncated_short_period_batch(
        nu_arr, g_arr, Q_arr, G_mean, omega_arr, j_coeffs, re_val, mu_val)
    L_osc = L_mean + polar_corr["dPsi"] + polar_corr["dM"]
    K_osc = solve_kepler_gen(L_osc, p1_osc, p2_osc)

    return np.column_stack([nu_arr, p1_osc, p2_osc, K_osc, q1_osc, q2_osc])


# --------------------------------------------------------------------------- #
#  Vectorized (batch) evaluators — Stage A performance optimization
# --------------------------------------------------------------------------- #


def _q_from_g_arr(g_arr: np.ndarray) -> np.ndarray:
    """Vectorized q_from_g: q = g / (1 + sqrt(1 - g^2))."""
    beta = np.sqrt(np.maximum(1.0 - g_arr * g_arr, 0.0))
    return g_arr / (1.0 + beta)


def _complex_f_from_g_arr(g_arr: np.ndarray, G_arr: np.ndarray) -> np.ndarray:
    """Vectorized F = (z - q) / (1 - q*z) where z = exp(i*G)."""
    q_arr = _q_from_g_arr(g_arr)
    z_arr = np.exp(1j * G_arr)
    return (z_arr - q_arr) / (1.0 - q_arr * z_arr)


def _evaluate_sp_batch_cfunc(
    N: int,
    cos_f_arr: np.ndarray,
    sin_f_arr: np.ndarray,
    q_arr: np.ndarray,
    Q_arr: np.ndarray,
    cos_omega_arr: np.ndarray,
    sin_omega_arr: np.ndarray,
    s2: float, s3: float, s4: float, s5: float,
) -> dict[str, np.ndarray]:
    """Evaluate SP corrections via heyoka cfunc (SIMD batch mode)."""
    cf, _ = _get_sp_cfunc()

    # heyoka batch: inputs (n_vars, N), pars (n_pars, N), outputs (n_outputs, N)
    inp = np.vstack([cos_f_arr, sin_f_arr, q_arr, Q_arr,
                     cos_omega_arr, sin_omega_arr])  # (6, N)
    pars = np.tile(np.array([[s2], [s3], [s4], [s5]]), (1, N))  # (4, N)
    out = np.empty((5, N))

    cf(inp, pars=pars, outputs=out)

    return {"dg": out[0], "dQ": out[1], "dPsi": out[2],
            "dOmega": out[3], "dM": out[4]}


def evaluate_truncated_short_period_batch(
    nu_arr: np.ndarray,
    g_arr: np.ndarray,
    Q_arr: np.ndarray,
    G_arr: np.ndarray,
    omega_arr: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> dict[str, np.ndarray]:
    """Vectorized short-period evaluation over N states.

    All input arrays have shape (N,).  Returns dict of (N,) correction arrays.
    Uses heyoka cfunc fast path when available (Stage C.2), otherwise falls
    back to the lambdified path.
    """
    N = len(nu_arr)

    # --- cfunc fast path ---
    if _HAS_SP_CFUNC and set(j_coeffs.keys()) == {2, 3, 4, 5}:
        q_arr_val = _q_from_g_arr(g_arr)
        F_arr = _complex_f_from_g_arr(g_arr, G_arr)
        cos_f_arr = np.real(F_arr)
        sin_f_arr = np.imag(F_arr)
        cos_omega_arr = np.cos(omega_arr)
        sin_omega_arr = np.sin(omega_arr)

        nu_val = float(nu_arr[0])
        a_val = (mu_val / (nu_val * nu_val)) ** (1.0 / 3.0)
        ra = re_val / a_val
        ra2 = ra * ra
        s2 = j_coeffs[2] * ra2
        s3 = j_coeffs[3] * ra2 * ra
        s4 = j_coeffs[4] * ra2 * ra2
        s5 = j_coeffs[5] * ra2 * ra2 * ra

        return _evaluate_sp_batch_cfunc(
            N, cos_f_arr, sin_f_arr, q_arr_val, Q_arr,
            cos_omega_arr, sin_omega_arr, s2, s3, s4, s5)

    # --- Lambdified fallback ---
    results = {f"d{v}": np.zeros(N) for v in ("g", "Q", "Psi", "Omega", "M")}

    q_arr = _q_from_g_arr(g_arr)
    F_arr = _complex_f_from_g_arr(g_arr, G_arr)
    w_arr = np.exp(1j * omega_arr)
    a_arr = (mu_val / (nu_arr * nu_arr)) ** (1.0 / 3.0)

    # φ for log terms (shared across all degrees)
    phi_arr = np.arctan2(q_arr * np.imag(F_arr), 1.0 + q_arr * np.real(F_arr))

    for n, jn_val in sorted(j_coeffs.items()):
        scale_arr = jn_val * (re_val / a_arr) ** n
        coeffs = _lambdified_short_period_expressions(n)
        log_coeffs = _lambdified_log_coefficients(n)

        for variable in ("g", "Q", "Psi", "Omega", "M"):
            total = np.zeros(N, dtype=complex)
            for m_val, func in coeffs[variable].items():
                total += func(q_arr, Q_arr, F_arr) * (w_arr ** m_val)
            # Add log contributions
            if variable in log_coeffs:
                for m_val, c_log_func in log_coeffs[variable].items():
                    total += c_log_func(q_arr, Q_arr) * phi_arr * (w_arr ** m_val)
            results[f"d{variable}"] += scale_arr * total.real

    return results


def mean_to_osculating_state_batch(
    mean_states: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> np.ndarray:
    """Vectorized mean -> osculating transformation for N states.

    Parameters
    ----------
    mean_states : (N, 6) array of [nu, p1, p2, M, q1, q2]
    j_coeffs : {degree: Jn_value}

    Returns
    -------
    (N, 6) array of osculating [nu, p1, p2, K, q1, q2]
    """
    nu_arr = mean_states[:, 0]
    p1_arr = mean_states[:, 1]
    p2_arr = mean_states[:, 2]
    M_arr = mean_states[:, 3]
    q1_arr = mean_states[:, 4]
    q2_arr = mean_states[:, 5]

    g_arr = np.hypot(p1_arr, p2_arr)
    Q_arr = np.hypot(q1_arr, q2_arr)
    Psi_arr = np.arctan2(p1_arr, p2_arr)
    Omega_arr = np.arctan2(q1_arr, q2_arr)
    omega_arr = Psi_arr - Omega_arr

    L_mean = Psi_arr + M_arr
    K_mean = solve_kepler_gen(L_mean, p1_arr, p2_arr)
    G_mean = K_mean - Psi_arr

    corr = evaluate_truncated_short_period_batch(
        nu_arr, g_arr, Q_arr, G_mean, omega_arr,
        j_coeffs, re_val, mu_val,
    )

    g_osc = g_arr + corr["dg"]
    Psi_osc = Psi_arr + corr["dPsi"]
    Q_osc = Q_arr + corr["dQ"]
    Omega_osc = Omega_arr + corr["dOmega"]
    M_osc = M_arr + corr["dM"]

    p1_osc = g_osc * np.sin(Psi_osc)
    p2_osc = g_osc * np.cos(Psi_osc)
    q1_osc = Q_osc * np.sin(Omega_osc)
    q2_osc = Q_osc * np.cos(Omega_osc)
    L_osc = Psi_osc + M_osc
    K_osc = solve_kepler_gen(L_osc, p1_osc, p2_osc)

    return np.column_stack([nu_arr, p1_osc, p2_osc, K_osc, q1_osc, q2_osc])


def _latex(expr: sp.Expr) -> str:
    return sp.latex(sp.factor(sp.cancel(sp.expand(expr))))


def _sorted_degree_keys(data: dict[int, object]) -> list[int]:
    return sorted(int(key) for key in data)


def _serialize_generated_file(
    mean_data: dict[int, dict[str, dict[int, str]]],
    short_data: dict[int, dict[str, dict[int, str]]],
    log_data: dict[int, dict[str, dict[int, str]]] | None = None,
) -> str:
    lines = ['"""Generated exact mixed-zonal short-period coefficients."""', "", "MEAN_DATA = {"]
    for n in _sorted_degree_keys(mean_data):
        lines.append(f"    {n}: {{")
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            coeffs = mean_data[n].get(variable, {})
            lines.append(f'        "{variable}": {{')
            for m_val in sorted(int(key) for key in coeffs):
                lines.append(f'            "{m_val}": {coeffs[m_val]!r},')
            lines.append("        },")
        lines.append("    },")
    lines.extend(["}", "", "SHORT_DATA = {"])
    for n in _sorted_degree_keys(short_data):
        lines.append(f"    {n}: {{")
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            coeffs = short_data[n].get(variable, {})
            lines.append(f'        "{variable}": {{')
            for m_val in sorted(int(key) for key in coeffs):
                lines.append(f'            "{m_val}": {coeffs[m_val]!r},')
            lines.append("        },")
        lines.append("    },")
    lines.extend(["}", ""])
    if log_data:
        lines.append("LOG_DATA = {")
        for n in _sorted_degree_keys(log_data):
            lines.append(f"    {n}: {{")
            for variable in ("g", "Q", "Psi", "Omega", "M"):
                coeffs = log_data[n].get(variable, {})
                if not coeffs:
                    continue
                lines.append(f'        "{variable}": {{')
                for m_val in sorted(int(key) for key in coeffs):
                    lines.append(f'            "{m_val}": {coeffs[m_val]!r},')
                lines.append("        },")
            lines.append("    },")
        lines.extend(["}", ""])
    return "\n".join(lines)


def _copy_generated_table(table: dict[int, dict[str, dict[str, str]]]) -> dict[int, dict[str, dict[int, str]]]:
    out: dict[int, dict[str, dict[int, str]]] = {}
    for degree_key, variable_map in table.items():
        n = int(degree_key)
        out[n] = {}
        for variable, coeffs in variable_map.items():
            out[n][variable] = {int(key): value for key, value in coeffs.items()}
    return out


def _load_generated_tables() -> tuple[
    dict[int, dict[str, dict[int, str]]],
    dict[int, dict[str, dict[int, str]]],
    dict[int, dict[str, dict[int, str]]],
]:
    mean_data = _copy_generated_table(_MEAN_DATA_STR)
    short_data = _copy_generated_table(_SHORT_DATA_STR)
    log_data = _copy_generated_table(_LOG_DATA_STR)
    if GENERATED_DATA.exists():
        module_ast = ast.parse(GENERATED_DATA.read_text())
        namespace: dict[str, object] = {}
        for node in module_ast.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"MEAN_DATA", "SHORT_DATA", "LOG_DATA"}:
                    namespace[name] = ast.literal_eval(node.value)
        if "MEAN_DATA" in namespace:
            mean_data = _copy_generated_table(namespace["MEAN_DATA"])
        if "SHORT_DATA" in namespace:
            short_data = _copy_generated_table(namespace["SHORT_DATA"])
        if "LOG_DATA" in namespace:
            log_data = _copy_generated_table(namespace["LOG_DATA"])
    return mean_data, short_data, log_data


def _persist_generated_tables(
    mean_data: dict[int, dict[str, dict[int, str]]],
    short_data: dict[int, dict[str, dict[int, str]]],
    log_data: dict[int, dict[str, dict[int, str]]] | None = None,
) -> None:
    GENERATED_DATA.write_text(_serialize_generated_file(mean_data, short_data, log_data))


def write_generated_data(
    degrees: tuple[int, ...] = (2, 3, 4, 5),
    variables: tuple[str, ...] = ("g", "Q", "Psi", "Omega", "M"),
    resume: bool = True,
) -> None:
    mean_data, short_data, log_data = _load_generated_tables() if resume else ({}, {}, {})
    selected_variables = tuple(variable for variable in ("g", "Q", "Psi", "Omega", "M") if variable in variables)

    for n in degrees:
        if n not in mean_data or not resume:
            mean_data[n] = {}
        if n not in short_data or not resume:
            short_data[n] = {}

        if not mean_data[n]:
            print(f"[generated] mean degree {n}", flush=True)
            mean_coeffs = _compute_isolated_mean_laurent_coefficients(n)
            mean_data[n] = {
                variable: {m_val: str(_clean(expr)) for m_val, expr in sorted(coeffs.items())}
                for variable, coeffs in mean_coeffs.items()
            }
            _persist_generated_tables(mean_data, short_data, log_data)

        for variable in selected_variables:
            if resume and short_data[n].get(variable):
                print(f"[generated] short degree {n} variable {variable} (cached)", flush=True)
                continue
            print(f"[generated] short degree {n} variable {variable}", flush=True)
            coeffs = _compute_isolated_short_period_expressions(variable, n)
            short_data[n][variable] = {m_val: str(expr) for m_val, expr in sorted(coeffs.items())}
            _persist_generated_tables(mean_data, short_data, log_data)

    print(f"Wrote {GENERATED_DATA}")


def _parse_csv_list(raw: str | None, cast: object = str) -> tuple[object, ...]:
    if raw is None:
        return ()
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def write_generated_data_legacy(degrees: tuple[int, ...] = (2, 3, 4, 5)) -> None:
    lines = [
        '"""Generated exact mixed-zonal short-period coefficients."""',
        "",
        "MEAN_DATA = {",
    ]
    for n in degrees:
        print(f"[generated] mean degree {n}", flush=True)
        mean_coeffs = _compute_isolated_mean_laurent_coefficients(n)
        lines.append(f"    {n}: {{")
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            coeffs = mean_coeffs[variable]
            lines.append(f'        "{variable}": {{')
            for m_val, expr in sorted(coeffs.items()):
                lines.append(f'            "{m_val}": {str(_clean(expr))!r},')
            lines.append("        },")
        lines.append("    },")
    lines.extend(["}", "", "SHORT_DATA = {"])
    for n in degrees:
        lines.append(f"    {n}: {{")
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            print(f"[generated] short degree {n} variable {variable}", flush=True)
            coeffs = _compute_isolated_short_period_expressions(variable, n)
            lines.append(f'        "{variable}": {{')
            for m_val, expr in sorted(coeffs.items()):
                lines.append(f'            "{m_val}": {str(_clean(expr))!r},')
            lines.append("        },")
        lines.append("    },")
    lines.extend(["}", ""])
    GENERATED_DATA.write_text("\n".join(lines))
    print(f"Wrote {GENERATED_DATA}")


def write_note() -> None:
    supports = {n: isolated_short_period_support(n) for n in (2, 3, 4, 5)}
    mean_m = isolated_mean_laurent_coefficients(3)["M"]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,longtable}",
        r"\begin{document}",
        r"\section*{Exact First-Order Mixed-Zonal GEqOE Short-Period Generator}",
        r"The mixed-zonal short-period map is built in the uniformly advancing",
        r"mean anomaly / longitude variable",
        r"\begin{equation}",
        r"L = K + p_1 \cos K - p_2 \sin K, \qquad M = L - \Psi,",
        r"\end{equation}",
        r"so that $\dot M_0 = \dot L_0 = \nu$ on the unperturbed ellipse.",
        r"Using $F=e^{if}$, $w=e^{i(\Psi-\Omega)}$,",
        r"\begin{equation}",
        r"D^{-1}(F;q) = \frac{(1+q^2)(1+qF)(F+q)}{(1-q^2)^2 F},",
        r"\qquad",
        r"\hat z = \frac{Q}{i(1+Q^2)}\left(wF - w^{-1}F^{-1}\right),",
        r"\end{equation}",
        r"the isolated degree-$n$ first-order frozen-state rates become finite",
        r"Laurent series in $(w,F)$. The periodic kernels are then obtained as",
        r"exact rational functions of $F$ for each finite $w$-harmonic, with the zero-mean gauge",
        r"$\langle u_1\rangle_M = 0$.",
        r"",
        r"For the fast phase the clean variable is $M$, not $K$ directly.",
        r"After reconstructing the osculating $(g,Q,\Psi,\Omega,M)$, one recovers",
        r"$L_{\mathrm{osc}}=\Psi_{\mathrm{osc}}+M_{\mathrm{osc}}$ and finally",
        r"$K_{\mathrm{osc}}$ from the generalized Kepler equation.",
        r"",
        r"\subsection*{Degree-wise structural formulas}",
        r"For an isolated zonal degree $n$ with $\lambda_n = J_n (R_e/a)^n$,",
        r"the dimensionless first-order rates are",
        r"\begin{align}",
        r"\frac{\dot g_n}{\nu\lambda_n} &= \frac{n-1}{2 i \beta} D^{-n} P_n(\hat z)\left(F-F^{-1}\right),\\",
        r"\frac{\dot Q_n}{\nu\lambda_n} &= -\frac{\delta}{2\beta} D^{-(n+1)} P_n'(\hat z)\cos u,\\",
        r"\frac{\dot\Omega_n}{\nu\lambda_n} &= -\frac{\delta}{2\beta Q} D^{-(n+1)} P_n'(\hat z)\sin u,\\",
        r"\frac{\dot M_n}{\nu\lambda_n} &= \frac{n-1}{2g} D^{-n} P_n(\hat z)\left(F+F^{-1}\right),\\",
        r"\frac{\dot\Psi_n}{\nu\lambda_n} &= -\frac{2n-1}{\beta}D^{-(n+1)}P_n(\hat z)",
        r"- \frac{n-1}{2g\beta}D^{-n}P_n(\hat z)\left(F+F^{-1}\right)",
        r"- \frac{\delta}{2\beta} D^{-(n+1)} \hat z P_n'(\hat z).",
        r"\end{align}",
        r"The exact mean coefficients follow from the $M$-average",
        r"$\langle F^k\rangle_M$ and the periodic kernels from the finite",
        r"linear system generated by $\partial_M F = i \beta D^{-2} F$.",
        r"",
        r"\subsection*{Support Summary}",
        r"\begin{longtable}{cccccc}",
        r"\toprule",
        r"degree & variable & $m_{\max}$ in $w$ & raw $k_{\max}$ in forcing & terms \\",
        r"\midrule",
    ]

    for n in (2, 3, 4, 5):
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            support = supports[n][variable]
            lines.append(
                rf"{n} & ${variable}$ & {support['m_max']} & {support['k_raw_max']} & {support['terms']} \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"",
            r"\subsection*{Example: degree-$3$ mean-anomaly rate}",
            r"For $J_3$ the exact mean $M$ drift is",
            r"\begin{equation}",
            rf"\frac{{\dot{{\bar M}}_{{J_3}}}}{{\nu J_3 (R_e/a)^3}} = {_latex(sum(mean_m[m] * w**m for m in sorted(mean_m)))}.",
            r"\end{equation}",
            r"",
            r"The implementation in",
            r"\texttt{scripts/zonal\_short\_period\_general.py}",
            r"exposes exact symbolic-to-numeric evaluators for the degree-wise and",
            r"truncated mixed-zonal mean rates, inverse map, and forward",
            r"reconstruction.",
            r"\end{document}",
        ]
    )

    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="Generate the cached coefficient file only.")
    parser.add_argument(
        "--degrees",
        type=str,
        default="2,3,4,5",
        help="Comma-separated zonal degrees to generate.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="g,Q,Psi,Omega,M",
        help="Comma-separated variables to generate in the short-period cache.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rebuild the generated cache file from scratch instead of resuming.",
    )
    args = parser.parse_args()

    degrees = tuple(int(item) for item in _parse_csv_list(args.degrees, int))
    variables = tuple(str(item) for item in _parse_csv_list(args.variables, str))
    write_generated_data(degrees=degrees, variables=variables, resume=not args.no_resume)
    if not args.data_only:
        write_note()


if __name__ == "__main__":
    main()
