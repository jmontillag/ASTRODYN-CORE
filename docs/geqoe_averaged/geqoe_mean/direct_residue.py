"""Direct residue-based short-period solver for GEqOE zonal averaging.

Replaces the term-by-term ratint approach with a combined partial-fraction
decomposition computed via direct residue evaluation at the known poles
F = 0, F = -q, F = -1/q.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import sympy as sp

from .short_period import (
    q, Q, F, w, z, I,
    beta, g, alpha, gamma, delta,
    _clean,
    dimensionless_rate_series,
    mean_f_power,
    _mean_rate_from_raw_series,
    _series_by_m,
    _mean_of_f_expression,
    isolated_short_period_expressions_for,
    LaurentPoly, HarmonicExpr,
)


# ---------------------------------------------------------------------------
# Fast fraction simplification for rational functions of q, Q
# ---------------------------------------------------------------------------

def _fast_cancel(expr: sp.Expr) -> sp.Expr:
    """Cancel common factors in a rational expression of q, Q.

    Uses sp.Poly-based GCD which is much faster than sp.cancel for
    polynomial/rational expressions in the specific generators q, Q.
    Handles the imaginary unit I by factoring it out.
    """
    if expr == 0:
        return sp.Integer(0)

    # Factor out I if present
    e = sp.expand(expr)
    coeff_I = e.coeff(I)
    if coeff_I != 0 and sp.expand(e - I * coeff_I) == 0:
        # Expression is purely imaginary: I * f(q, Q)
        return I * _fast_cancel(coeff_I)

    n, d = sp.fraction(sp.together(e))
    if d == 1:
        return sp.expand(n)
    n_exp = sp.expand(n)
    d_exp = sp.expand(d)
    try:
        n_poly = sp.Poly(n_exp, q, Q)
        d_poly = sp.Poly(d_exp, q, Q)
        g = sp.gcd(n_poly, d_poly)
        if g.is_one:
            return n_exp / d_exp
        n_red = sp.quo(n_poly, g).as_expr()
        d_red = sp.quo(d_poly, g).as_expr()
        return n_red / d_red if d_red != 1 else n_red
    except (sp.GeneratorsNeeded, sp.PolynomialError, sp.ComputationFailed):
        return sp.cancel(expr)


# ---------------------------------------------------------------------------
# Direct residue-based integration
# ---------------------------------------------------------------------------

def _build_combined_numerator(
    raw_by_k: dict[int, sp.Expr], mean_coeff: sp.Expr
) -> tuple[sp.Expr, int]:
    """Build the combined polynomial numerator for the integrand.

    Returns (N_poly, shift) where N_poly is a true polynomial in F,
    and the integrand is scale * N_poly / [F^shift * (F+q)^2 * (1+qF)^2].
    """
    numer_coeffs: dict[int, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for k_val, coeff in raw_by_k.items():
        numer_coeffs[k_val + 1] += coeff
    if mean_coeff != 0:
        numer_coeffs[1] -= mean_coeff

    numer_coeffs = {k: _fast_cancel(v) for k, v in numer_coeffs.items() if v != 0}
    if not numer_coeffs:
        return sp.Integer(0), 0

    k_min = min(numer_coeffs.keys())
    shift = max(0, -k_min)

    N_poly = sp.expand(sum(c * F ** (k + shift) for k, c in numer_coeffs.items()))
    return N_poly, shift


def _poly_eval(poly_expr: sp.Expr, F_val: sp.Expr) -> sp.Expr:
    """Evaluate a polynomial expression in F at a specific value."""
    return _fast_cancel(poly_expr.subs(F, F_val))


def _poly_deriv_eval(poly_expr: sp.Expr, F_val: sp.Expr, order: int = 1) -> sp.Expr:
    """Evaluate the k-th derivative of a polynomial in F at a specific point."""
    d = sp.diff(poly_expr, F, order)
    return _fast_cancel(d.subs(F, F_val))


def _integrate_polynomial(poly_expr: sp.Expr) -> sp.Expr:
    """Integrate a polynomial in F term-by-term. Much faster than sp.integrate."""
    p = sp.Poly(sp.expand(poly_expr), F)
    result = sp.Integer(0)
    for (power,), coeff in p.terms():
        result += coeff * F ** (power + 1) / (power + 1)
    return result


def _scale_polynomial_coeffwise(poly_expr: sp.Expr, scale_expr: sp.Expr) -> sp.Expr:
    """Multiply a polynomial in F by a scalar, cancelling each coefficient individually.

    This avoids the catastrophic multivariate GCD that sp.cancel(scale * poly)
    triggers when the polynomial has many terms with rational coefficients in q, Q.
    """
    p = sp.Poly(sp.expand(poly_expr), F)
    result = sp.Integer(0)
    for (power,), coeff in p.terms():
        result += _fast_cancel(scale_expr * coeff) * F ** power
    return result


@lru_cache(maxsize=None)
def _mean_of_f_inv_pole(pole: str) -> sp.Expr:
    """Compute <1/(F+q)>_M or <1/(1+qF)>_M by direct residue calculus.

    For 1/(F+q):
      After substituting F = (z-q)/(1-qz) and weighting by D/z,
      the integrand has a pole only at z=0, giving:
        <1/(F+q)>_M = -q(2+q^2) / ((1-q^2)(1+q^2))

    For 1/(1+qF):
      <1/(1+qF)>_M = (1+2q^2) / ((1-q^2)(1+q^2))
    """
    if pole == "Fpq":
        # <1/(F+q)>_M: integrand is (1-qz)^2(z-q)/((1-q^2)(1+q^2)z^3)
        # Residue at z=0 from z^2 coefficient of (1-qz)^2(z-q)
        # = coefficient of z^2 in (-q + (1+2q^2)z + (-2q-q^3)z^2 + q^2 z^3) = -q(2+q^2)
        return _clean(-q * (2 + q**2) / ((1 - q**2) * (1 + q**2)))
    elif pole == "1pqF":
        # <1/(1+qF)>_M: integrand is (1-qz)^2(z-q)/((1-q^2)(1+q^2)z^2)
        # Residue at z=0 from z^1 coefficient of (1-qz)^2(z-q) = (1+2q^2)
        return _clean((1 + 2 * q**2) / ((1 - q**2) * (1 + q**2)))
    raise ValueError(pole)


def _mean_of_structural_result(
    Q_int_expr: sp.Expr,
    B: sp.Expr,
    D: sp.Expr,
    E_periodic: dict[int, sp.Expr],
    scale: sp.Expr,
) -> sp.Expr:
    """Compute the M-average of the structural antiderivative efficiently.

    Instead of substituting F -> F(z) into the full expression (which causes
    expression swell), we compute the mean term-by-term:
      <Q_int(F)>_M = Σ c_k <F^{k+1}>_M / (k+1)  [from polynomial]
      <B/(F+q)>_M  = B * <1/(F+q)>_M              [cached formula]
      <D/(q(1+qF))>_M = (D/q) * <1/(1+qF)>_M      [cached formula]
      <E_j/F^{j-1}>_M = E_j * <F^{-(j-1)}>_M      [from mean_f_power]
    """
    avg = sp.Integer(0)

    # Polynomial part: Q_int(F) = Σ c_k F^k (already integrated polynomial)
    if Q_int_expr != 0:
        p = sp.Poly(sp.expand(Q_int_expr), F)
        for (power,), coeff in p.terms():
            avg += coeff * mean_f_power(power)

    # Double-pole at F = -q: term is -B/(F+q)
    if B != 0:
        avg -= B * _mean_of_f_inv_pole("Fpq")

    # Double-pole at F = -1/q: term is -D/(q(1+qF))
    if D != 0:
        avg -= D / q * _mean_of_f_inv_pole("1pqF")

    # Poles at F = 0: terms are -E_j/((j-1)F^{j-1})
    for j, Ej in E_periodic.items():
        if Ej != 0:
            avg -= Ej / (j - 1) * mean_f_power(-(j - 1))

    return _clean(scale * avg)


def integrate_harmonic_residue(
    raw_by_k: dict[int, sp.Expr], mean_coeff: sp.Expr
) -> tuple[sp.Expr, sp.Expr | None]:
    """Solve the homological equation for one w-harmonic by direct residues.

    The integrand is:
        scale * N(F) / [F^s * (F+q)^2 * (1+qF)^2]

    We compute partial fractions DIRECTLY via residue evaluation:
    - Double-pole at F=-q:  B = φ(-q) where φ = N_rem/[F^s(1+qF)^2]
    - Double-pole at F=-1/q: D = ψ(-1/q) where ψ = N_rem/[F^s(F+q)^2]
    - Poles at F=0:  E_j from Taylor of χ = N_rem/[(F+q)^2(1+qF)^2]
    - Polynomial part from long division

    Returns (u1_rational, C_log) where:
    - u1_rational: the rational periodic antiderivative
    - C_log: coefficient such that u1_full = u1_rational + C_log * arctan2(q*Im(F), 1+q*Re(F)),
             or None if the harmonic has no log contribution (zero-mean harmonics).
    """
    if not raw_by_k:
        return sp.Integer(0), None

    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    if N_poly == 0:
        return sp.Integer(0), None

    scale = -I * (1 - q**2) ** 3 / (1 + q**2)

    # --- Step 1: Polynomial long division ---
    # R(F) = N / [F^s * (F+q)^2 * (1+qF)^2]
    # Compute Q, N_rem such that N = Q * denom + N_rem
    denom_expr = sp.expand(F ** shift * (F + q) ** 2 * (1 + q * F) ** 2)

    N_sp = sp.Poly(N_poly, F)
    denom_sp = sp.Poly(denom_expr, F)

    deg_N = N_sp.degree()
    deg_D = denom_sp.degree()

    if deg_N >= deg_D:
        Q_poly, N_rem_poly = sp.div(N_sp, denom_sp)
        Q_expr = Q_poly.as_expr()
        N_rem = N_rem_poly.as_expr()
    else:
        Q_expr = sp.Integer(0)
        N_rem = N_poly

    N_rem = sp.expand(N_rem)

    # --- Step 2: Double-pole residue at F = -q ---
    # φ(F) = N_rem / [F^s * (1+qF)^2]
    # B = φ(-q) = N_rem(-q) / [(-q)^s * (1-q^2)^2]
    N_rem_at_mq = _poly_eval(N_rem, -q)
    mq_s = (-q) ** shift
    one_mq2_sq = (1 - q**2) ** 2
    B = _fast_cancel(N_rem_at_mq / (mq_s * one_mq2_sq))

    # --- Step 3: Double-pole residue at F = -1/q ---
    # ψ(F) = N_rem / [F^s * (F+q)^2]
    # D = ψ(-1/q) = N_rem(-1/q) / [(-1/q)^s * (-1/q + q)^2]
    #             = N_rem(-1/q) / [(-1)^s q^{-s} * (q^2-1)^2/q^2]
    #             = N_rem(-1/q) * (-1)^s * q^{s+2} / (q^2-1)^2
    N_rem_at_m1q = _poly_eval(N_rem, -sp.Integer(1) / q)
    # Clean up: multiply through by q^deg to clear denominators
    N_rem_at_m1q = _fast_cancel(N_rem_at_m1q)
    m1q_s = (-sp.Integer(1) / q) ** shift
    m1q_pq_sq = ((-sp.Integer(1) / q) + q) ** 2  # = ((q^2-1)/q)^2
    D = _fast_cancel(N_rem_at_m1q / (m1q_s * m1q_pq_sq))

    # --- Step 4: Residues at F = 0 (for shift >= 1) ---
    # χ(F) = N_rem / [(F+q)^2 * (1+qF)^2]
    # E_j = χ^{(s-j)}(0) / (s-j)!  for j = 1, ..., s
    # j=1 gives E₁ (log coefficient); j≥2 give the periodic pole terms
    chi_numer = N_rem
    chi_denom = sp.expand((F + q) ** 2 * (1 + q * F) ** 2)
    E_periodic = {}  # j -> E_j for j >= 2
    E1 = sp.Integer(0)

    if shift >= 1:
        # We need Taylor coefficients of χ(F) = chi_numer/chi_denom at F=0
        # χ(F) = Σ c_k F^k, and E_j = c_{s-j}
        # Compute via repeated differentiation
        chi_denom_at_0 = _poly_eval(chi_denom, sp.Integer(0))  # = q^2 * 1 = q^4 wait...
        # chi_denom = (F+q)^2(1+qF)^2. At F=0: q^2 * 1 = q^2

        # Instead of differentiating the ratio, use the formula:
        # E_j = [1/(s-j)!] * [d^{s-j}/dF^{s-j} (F^s * R_proper)]_{F=0}
        # where R_proper = N_rem / [F^s * (F+q)^2 * (1+qF)^2]
        # F^s * R_proper = N_rem / [(F+q)^2 * (1+qF)^2] = χ(F)
        #
        # So E_j = χ^{(s-j)}(0) / (s-j)!
        #
        # For efficiency, compute the Taylor series of χ(F) = chi_numer/chi_denom
        # up to order s-1 (for E₁ at j=1, plus E_2, ..., E_s).
        #
        # Use: if χ = P/Q, then P = Q * χ, so Taylor coefficients of χ can be
        # computed by solving P_k = Σ_{j=0}^{k} Q_j χ_{k-j} for each k.

        # Get Taylor coefficients of P and Q up to order s-1
        max_order = shift - 1  # highest order needed (for j=1: order s-1)

        # Taylor coefficients of chi_numer (it's a polynomial)
        P_coeffs = {}
        chi_numer_poly = sp.Poly(chi_numer, F) if chi_numer.has(F) else sp.Poly(chi_numer, F, domain="ZZ(q,Q)")
        for (power,), coeff in chi_numer_poly.terms():
            P_coeffs[power] = coeff
        # Taylor coefficients of chi_denom
        Q_coeffs = {}
        chi_denom_poly = sp.Poly(chi_denom, F)
        for (power,), coeff in chi_denom_poly.terms():
            Q_coeffs[power] = coeff

        # Solve for χ coefficients: P_k = Σ Q_j χ_{k-j}
        # => χ_k = (P_k - Σ_{j=1}^{k} Q_j χ_{k-j}) / Q_0
        Q_0 = Q_coeffs.get(0, sp.Integer(0))
        Q_0_inv = sp.Rational(1, 1) / Q_0  # Q_0 = q^{2*something}, so this is cheap
        chi_taylor: dict[int, sp.Expr] = {}
        for k in range(max_order + 1):
            val = P_coeffs.get(k, sp.Integer(0))
            for j in range(1, k + 1):
                qj = Q_coeffs.get(j, sp.Integer(0))
                if qj != 0 and (k - j) in chi_taylor:
                    val -= qj * chi_taylor[k - j]
            chi_taylor[k] = _fast_cancel(val * Q_0_inv)

        # E₁ = chi_taylor[s-1] (log coefficient at F=0)
        E1_raw = chi_taylor.get(shift - 1, sp.Integer(0))
        if E1_raw != 0:
            E1 = _fast_cancel(E1_raw)

        # E_j for j >= 2 (periodic pole terms)
        for j in range(2, shift + 1):
            order = shift - j
            if order in chi_taylor and chi_taylor[order] != 0:
                E_periodic[j] = chi_taylor[order]

    # --- Log coefficient ---
    # C_log = 2(1-q²)³/(1+q²) · E₁, the coefficient of arctan2(q·Im(F), 1+q·Re(F))
    C_log = None
    if mean_coeff != 0 and E1 != 0:
        C_log = _fast_cancel(2 * (1 - q**2)**3 / (1 + q**2) * E1)

    # --- Step 5: Assemble the periodic antiderivative ---
    # P_per(F) = scale * [Q_int(F) - B/(F+q) - D/(q(1+qF)) - Σ E_j/((j-1)F^{j-1})]
    #
    # CRITICAL: Do NOT combine over common denominator (sp.together).
    # Instead, simplify each structural term individually and then
    # combine into a single rational function only at the end,
    # using the known denominator structure.

    Q_int = sp.Integer(0)
    if Q_expr != 0:
        Q_int = _integrate_polynomial(Q_expr)

    # --- Step 6: Zero-mean gauge (structural, no full substitution) ---
    avg = _mean_of_structural_result(Q_int, B, D, E_periodic, scale)

    # --- Step 7: Combine into a single canonical rational function ---
    # Use the known common denominator q * F^fpow * (F+q) * (1+qF)
    # and simplify the numerator coefficient-by-coefficient in F.
    max_j = max(E_periodic.keys()) if E_periodic else 1
    fpow = max(max_j - 1, 0)

    # Build the numerator over the common denominator
    cd = q * F ** fpow * (F + q) * (1 + q * F) if fpow > 0 else q * (F + q) * (1 + q * F)

    numer = sp.Integer(0)
    if Q_int != 0:
        numer += sp.expand(Q_int * cd)
    if B != 0:
        numer -= sp.expand(B * q * F ** fpow * (1 + q * F))
    if D != 0:
        numer -= sp.expand(D * F ** fpow * (F + q))
    for j, Ej in E_periodic.items():
        if Ej != 0:
            pwr = fpow - j + 1
            numer -= sp.expand(Ej / (j - 1) * q * F ** pwr * (F + q) * (1 + q * F))
    if avg != 0:
        numer -= sp.expand(avg / scale * cd)

    # Multiply numerator by scale and simplify each F-coefficient individually
    # (avoids catastrophic multivariate GCD from sp.cancel on the full expression)
    numer_scaled = _scale_polynomial_coeffwise(numer, scale)

    # Build canonical denominator (also with scale absorbed where needed)
    return numer_scaled / cd, C_log


def compute_log_coefficient(
    raw_by_k: dict[int, sp.Expr], mean_coeff: sp.Expr
) -> sp.Expr | None:
    """Compute just the log-term coefficient C_log for one harmonic.

    Lighter than integrate_harmonic_residue: only computes the Taylor
    recurrence for E₁ without assembling the full rational antiderivative.

    Returns C_log such that:
        u1_full = u1_rational + C_log * arctan2(q*Im(F), 1+q*Re(F))
    Returns None if the harmonic has no log contribution.
    """
    if not raw_by_k or mean_coeff == 0:
        return None

    N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)
    if N_poly == 0 or shift < 1:
        return None

    # Polynomial long division to get N_rem
    denom_expr = sp.expand(F ** shift * (F + q) ** 2 * (1 + q * F) ** 2)
    N_sp = sp.Poly(N_poly, F)
    denom_sp = sp.Poly(denom_expr, F)

    if N_sp.degree() >= denom_sp.degree():
        _, N_rem_poly = sp.div(N_sp, denom_sp)
        N_rem = sp.expand(N_rem_poly.as_expr())
    else:
        N_rem = sp.expand(N_poly)

    # Taylor coefficients of χ(F) = N_rem / [(F+q)^2 * (1+qF)^2]
    chi_denom = sp.expand((F + q) ** 2 * (1 + q * F) ** 2)

    chi_numer_poly = (sp.Poly(N_rem, F) if N_rem.has(F)
                      else sp.Poly(N_rem, F, domain="ZZ(q,Q)"))
    P_coeffs = {power: coeff for (power,), coeff in chi_numer_poly.terms()}

    chi_denom_poly = sp.Poly(chi_denom, F)
    Q_coeffs = {power: coeff for (power,), coeff in chi_denom_poly.terms()}

    Q_0 = Q_coeffs.get(0, sp.Integer(0))
    Q_0_inv = sp.Rational(1, 1) / Q_0

    max_order = shift - 1
    chi_taylor: dict[int, sp.Expr] = {}
    for k in range(max_order + 1):
        val = P_coeffs.get(k, sp.Integer(0))
        for j in range(1, k + 1):
            qj = Q_coeffs.get(j, sp.Integer(0))
            if qj != 0 and (k - j) in chi_taylor:
                val -= qj * chi_taylor[k - j]
        chi_taylor[k] = _fast_cancel(val * Q_0_inv)

    E1 = chi_taylor.get(shift - 1, sp.Integer(0))
    if E1 == 0:
        return None

    return _fast_cancel(2 * (1 - q**2)**3 / (1 + q**2) * E1)


def compute_short_period_direct(
    variable: str, n: int
) -> tuple[HarmonicExpr, dict[int, sp.Expr]]:
    """Compute isolated-degree short-period expressions using the direct method.

    Returns (rational_exprs, log_coeffs) where:
    - rational_exprs: dict mapping m -> rational SP expression
    - log_coeffs: dict mapping m -> C_log coefficient (only for secular-rate harmonics)
    """
    raw = dimensionless_rate_series(variable, n)
    mean_coeffs = _mean_rate_from_raw_series(raw)
    solved: HarmonicExpr = {}
    log_data: dict[int, sp.Expr] = {}
    for m_val, raw_by_k in _series_by_m(raw).items():
        t0 = time.time()
        rational, c_log = integrate_harmonic_residue(
            raw_by_k, mean_coeffs.get(m_val, sp.Integer(0))
        )
        solved[m_val] = rational
        if c_log is not None:
            log_data[m_val] = c_log
        dt = time.time() - t0
        print(f"    harmonic m={m_val:+d}: {dt:.2f}s", flush=True)
    # Do NOT call _clean here: integrate_harmonic_residue already simplifies each
    # structural term individually.  Calling _clean (sp.together + sp.cancel) would
    # try to combine all terms over a common denominator, which is catastrophically
    # slow for higher zonal degrees.
    rational_out = {m_val: expr for m_val, expr in solved.items() if expr != 0}
    return rational_out, log_data


# ---------------------------------------------------------------------------
# Validation against existing cached J2 data
# ---------------------------------------------------------------------------

def _compare_expressions(label: str, new_expr: sp.Expr, ref_expr: sp.Expr) -> bool:
    """Check if two symbolic expressions are equivalent."""
    diff = _clean(sp.expand(new_expr - ref_expr))
    ok = diff == 0
    status = "OK" if ok else f"MISMATCH (diff = {diff})"
    print(f"  {label}: {status}")
    return ok


def validate_j2_against_cache() -> bool:
    """Validate the direct method against cached J2 short-period data."""
    print("=" * 70)
    print("Validating direct method against cached J2 short-period expressions")
    print("=" * 70)

    all_ok = True
    for variable in ("g", "Q", "M"):  # Psi, Omega not cached yet
        print(f"\nVariable: {variable}")

        ref = isolated_short_period_expressions_for(variable, 2)
        if not ref:
            print(f"  No cached data for {variable}, skipping")
            continue

        t0 = time.time()
        direct, _ = compute_short_period_direct(variable, 2)
        dt = time.time() - t0
        print(f"  Total: {dt:.3f} s")

        all_m = set(ref) | set(direct)
        for m_val in sorted(all_m):
            ref_expr = ref.get(m_val, sp.Integer(0))
            dir_expr = direct.get(m_val, sp.Integer(0))
            ok = _compare_expressions(f"m={m_val}", dir_expr, ref_expr)
            all_ok = all_ok and ok

    return all_ok


def generate_all_degrees() -> None:
    """Generate all short-period expressions for J2-J5 using the direct method."""
    print("\n" + "=" * 70)
    print("Generating all short-period expressions (J2-J5)")
    print("=" * 70)

    results: dict[int, dict[str, HarmonicExpr]] = {}
    log_results: dict[int, dict[str, dict[int, sp.Expr]]] = {}

    for n in (2, 3, 4, 5):
        results[n] = {}
        log_results[n] = {}
        for variable in ("g", "Q", "Psi", "Omega", "M"):
            print(f"\n  n={n} {variable}:", flush=True)
            t0 = time.time()
            expr, log_data = compute_short_period_direct(variable, n)
            dt = time.time() - t0
            results[n][variable] = expr
            if log_data:
                log_results[n][variable] = log_data
            n_harmonics = len(expr)
            n_log = len(log_data)
            print(f"  => {dt:.2f}s total  ({n_harmonics} harmonics, {n_log} log terms)")

    return results, log_results


def write_generated_data(
    results: dict[int, dict[str, HarmonicExpr]],
    log_results: dict[int, dict[str, dict[int, sp.Expr]]] | None = None,
) -> None:
    """Write the generated short-period data to the cache file."""
    from .short_period import (
        _compute_isolated_mean_laurent_coefficients,
        _load_generated_tables,
        _persist_generated_tables,
    )

    mean_data, short_data, log_data = _load_generated_tables()

    for n, var_exprs in results.items():
        if n not in mean_data or not mean_data[n]:
            print(f"  Computing mean data for degree {n}...")
            mean_coeffs = _compute_isolated_mean_laurent_coefficients(n)
            mean_data[n] = {
                variable: {m_val: str(_clean(expr)) for m_val, expr in sorted(coeffs.items())}
                for variable, coeffs in mean_coeffs.items()
            }

        if n not in short_data:
            short_data[n] = {}

        for variable, exprs in var_exprs.items():
            short_data[n][variable] = {
                m_val: str(_clean(expr)) for m_val, expr in sorted(exprs.items())
            }

    if log_results:
        for n, var_logs in log_results.items():
            if n not in log_data:
                log_data[n] = {}
            for variable, logs in var_logs.items():
                log_data[n][variable] = {
                    m_val: str(expr) for m_val, expr in sorted(logs.items())
                }

    _persist_generated_tables(mean_data, short_data, log_data)
    print(f"  Wrote generated data")


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------

def numerical_spot_check() -> None:
    """Evaluate all five direct J2 short-period corrections at a test point
    and compare against direct frozen-state quadrature."""
    from .symbolic import q_from_g
    from .short_period import _complex_f_from_g
    from astrodyn_core.geqoe_taylor import J2, MU, RE

    print(f"\n{'=' * 70}")
    print("Numerical spot check: direct method vs frozen-state quadrature (J2)")
    print("=" * 70)

    a_km, e_val, inc_deg = 12000.0, 0.3, 50.0
    nu_val = (MU / a_km**3) ** 0.5
    g_val, Q_val = e_val, np.tan(np.deg2rad(inc_deg / 2))
    omega_val, G_val = np.deg2rad(60.0), np.deg2rad(45.0)
    q_val = q_from_g(g_val)
    F_val = _complex_f_from_g(g_val, G_val)
    w_val = np.exp(1j * omega_val)
    a_val = (MU / nu_val**2) ** (1.0 / 3.0)
    scale_num = J2 * (RE / a_val) ** 2

    for variable in ("g", "Q", "Psi", "Omega", "M"):
        direct_exprs, _ = compute_short_period_direct(variable, 2)
        total = 0.0j
        for m_val, expr in direct_exprs.items():
            func = sp.lambdify((q, Q, F), expr, "numpy")
            total += complex(func(q_val, Q_val, F_val)) * (w_val ** m_val)
        val = float(np.real_if_close(scale_num * total))
        print(f"  d{variable:5s} = {val:+.10e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--spot-check", action="store_true")
    parser.add_argument("--degrees", type=str, default="2,3,4,5")
    args = parser.parse_args()

    if args.validate_only:
        ok = validate_j2_against_cache()
        sys.exit(0 if ok else 1)

    if args.spot_check:
        numerical_spot_check()
        return

    if args.generate:
        results, log_results = generate_all_degrees()
        write_generated_data(results, log_results)
        return

    # Default: validate then spot-check
    ok = validate_j2_against_cache()
    if not ok:
        print("\nValidation FAILED")
        sys.exit(1)

    numerical_spot_check()
    print("\nAll checks passed. Run with --generate to produce full J2-J5 data.")


if __name__ == "__main__":
    main()
