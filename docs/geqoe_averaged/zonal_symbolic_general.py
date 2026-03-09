#!/usr/bin/env python
"""Exact symbolic averaged GEqOE zonal generator for isolated degree n.

The derivation uses two exact steps:
1. Expand P_n(sin(i) sin(u)) and P'_n(sin(i) sin(u)) as Laurent series in
   w = exp(i omega), where omega = Psi - Omega and exp(i f) = (z-q)/(1-q z).
2. Evaluate the one-revolution average coefficient-by-coefficient through the
   single interior pole z = q.

This yields exact finite-harmonic averaged drift formulas for any isolated
zonal degree n >= 2.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/zonal_symbolic_general.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import sympy as sp


OUT_TEX = Path(__file__).with_name("zonal_symbolic_general.tex")

q, Q, z = sp.symbols("q Q z", positive=True, real=True)
I = sp.I

beta = (1 - q**2) / (1 + q**2)
gamma = 1 + Q**2
delta = 1 - Q**2


def _clean(expr: sp.Expr) -> sp.Expr:
    return sp.factor(sp.cancel(sp.simplify(expr)))


@lru_cache(maxsize=None)
def legendre_laurent_coefficients(n: int, derivative: bool = False) -> dict[int, sp.Expr]:
    """Coefficient map m -> C_{n,m}(Q) in the exact Laurent expansion.

    With
        C(Q) = Q / (i (1 + Q^2)),
        F(z;q) = (z - q) / (1 - q z),

    this routine returns the exact coefficients in

        P_n(C(Q) (w F - w^{-1} F^{-1}))
        = sum_m C_{n,m}(Q) w^m F^m,

    or the analogous expansion for P'_n when derivative=True.
    """
    x = sp.symbols("x", real=True)
    poly_expr = sp.diff(sp.legendre(n, x), x) if derivative else sp.legendre(n, x)
    poly = sp.Poly(sp.expand(poly_expr), x)
    c_q = Q / (I * (1 + Q**2))

    coeffs: dict[int, sp.Expr] = {}
    for (power,), poly_coeff in poly.terms():
        for m in range(-power, power + 1, 2):
            j = (power + m) // 2
            term = poly_coeff * c_q**power * sp.binomial(power, j) * (-1) ** (power - j)
            coeffs[m] = coeffs.get(m, sp.Integer(0)) + term

    return {m: _clean(val) for m, val in coeffs.items() if val != 0}


def _pole_value(order: int, analytic_part: sp.Expr) -> sp.Expr:
    if order <= 0:
        return sp.Integer(0)
    deriv_order = order - 1
    deriv = sp.diff(analytic_part, z, deriv_order)
    return deriv.subs(z, q) / sp.factorial(deriv_order)


@lru_cache(maxsize=None)
def kernel_core(n: int, m: int) -> sp.Expr:
    """Exact z = q residue of D^{-n} F^m / z."""
    order = n - m
    if order <= 0:
        return sp.Integer(0)
    analytic = z ** (n - 1) / (1 - q * z) ** (n + m)
    return _clean((1 + q**2) ** n * _pole_value(order, analytic))


@lru_cache(maxsize=None)
def kernel_g(n: int, m: int) -> sp.Expr:
    """Exact z = q residue kernel for the magnitude drift."""
    order = n - m
    if order <= 0:
        return sp.Integer(0)
    analytic = z ** (n - 2) * (z**2 - 1) / (1 - q * z) ** (n + m)
    pref = (n - 1) * (1 + q**2) ** n / (2 * I)
    return _clean(pref * _pole_value(order, analytic))


@lru_cache(maxsize=None)
def kernel_psi_p(n: int, m: int) -> sp.Expr:
    """Exact z = q residue kernel for the P_n part of Psi_dot."""
    order = n - m
    if order <= 0:
        return sp.Integer(0)
    poly = 4 * n * q * z + (n - 1) * (1 + q**2) * (z**2 + 1)
    analytic = z ** (n - 2) * poly / (1 - q * z) ** (n + m)
    pref = -(1 + q**2) ** (n + 1) / (4 * q * (1 - q**2))
    return _clean(pref * _pole_value(order, analytic))


@lru_cache(maxsize=None)
def averaged_laurent_coefficients(n: int) -> dict[str, dict[int, sp.Expr]]:
    """Exact Laurent coefficients in w = exp(i omega) for isolated J_n."""
    p_coeffs = legendre_laurent_coefficients(n, derivative=False)
    dp_coeffs = legendre_laurent_coefficients(n, derivative=True)

    m_vals = set(p_coeffs)
    for m in dp_coeffs:
        m_vals.add(m - 1)
        m_vals.add(m + 1)

    g_coeffs: dict[int, sp.Expr] = {}
    q_coeffs: dict[int, sp.Expr] = {}
    psi_coeffs: dict[int, sp.Expr] = {}
    omega_coeffs: dict[int, sp.Expr] = {}

    for m in sorted(m_vals):
        core = kernel_core(n, m)
        g_expr = p_coeffs.get(m, sp.Integer(0)) * kernel_g(n, m)

        dp_combo_plus = dp_coeffs.get(m - 1, sp.Integer(0)) + dp_coeffs.get(m + 1, sp.Integer(0))
        dp_combo_minus = dp_coeffs.get(m - 1, sp.Integer(0)) - dp_coeffs.get(m + 1, sp.Integer(0))

        q_expr = -delta / (4 * beta) * dp_combo_plus * core
        psi_expr = (
            p_coeffs.get(m, sp.Integer(0)) * kernel_psi_p(n, m)
            - Q * delta / (2 * I * gamma * beta) * dp_combo_minus * core
        )
        omega_expr = -delta / (4 * I * beta * Q) * dp_combo_minus * core

        if g_expr != 0:
            g_coeffs[m] = _clean(g_expr)
        if q_expr != 0:
            q_coeffs[m] = _clean(q_expr)
        if psi_expr != 0:
            psi_coeffs[m] = _clean(psi_expr)
        if omega_expr != 0:
            omega_coeffs[m] = _clean(omega_expr)

    return {
        "g": g_coeffs,
        "Q": q_coeffs,
        "Psi": psi_coeffs,
        "Omega": omega_coeffs,
    }


def _real_harmonic_pair(coeffs: dict[int, sp.Expr], m: int) -> tuple[sp.Expr, sp.Expr]:
    c_pos = coeffs.get(m, sp.Integer(0))
    c_neg = coeffs.get(-m, sp.Integer(0))
    cos_coeff = _clean(c_pos + c_neg)
    sin_coeff = _clean(I * (c_pos - c_neg))
    return cos_coeff, sin_coeff


@lru_cache(maxsize=None)
def harmonic_coefficients(n: int) -> dict[str, dict[int, sp.Expr] | sp.Expr]:
    """Exact real Fourier coefficients for the isolated averaged J_n drift."""
    coeffs = averaged_laurent_coefficients(n)
    out: dict[str, dict[int, sp.Expr] | sp.Expr] = {
        "g": {},
        "Q": {},
        "Psi": {},
        "Omega": {},
    }

    if n % 2:
        for m in range(1, n, 2):
            g_cos, _ = _real_harmonic_pair(coeffs["g"], m)
            q_cos, _ = _real_harmonic_pair(coeffs["Q"], m)
            _, psi_sin = _real_harmonic_pair(coeffs["Psi"], m)
            _, omega_sin = _real_harmonic_pair(coeffs["Omega"], m)
            if g_cos != 0:
                out["g"][m] = g_cos
            if q_cos != 0:
                out["Q"][m] = q_cos
            if psi_sin != 0:
                out["Psi"][m] = psi_sin
            if omega_sin != 0:
                out["Omega"][m] = omega_sin
    else:
        psi0 = _clean(coeffs["Psi"].get(0, sp.Integer(0)))
        omega0 = _clean(coeffs["Omega"].get(0, sp.Integer(0)))
        if psi0 != 0:
            out["Psi"][0] = psi0
        if omega0 != 0:
            out["Omega"][0] = omega0
        for m in range(2, n, 2):
            _, g_sin = _real_harmonic_pair(coeffs["g"], m)
            _, q_sin = _real_harmonic_pair(coeffs["Q"], m)
            psi_cos, _ = _real_harmonic_pair(coeffs["Psi"], m)
            omega_cos, _ = _real_harmonic_pair(coeffs["Omega"], m)
            if g_sin != 0:
                out["g"][m] = g_sin
            if q_sin != 0:
                out["Q"][m] = q_sin
            if psi_cos != 0:
                out["Psi"][m] = psi_cos
            if omega_cos != 0:
                out["Omega"][m] = omega_cos

    return out


def eps_n(n: int, nu_sym: sp.Symbol, a_sym: sp.Symbol, re_sym: sp.Symbol, jn_sym: sp.Symbol) -> sp.Expr:
    return nu_sym * jn_sym * (re_sym / a_sym) ** n


def _latex(expr: sp.Expr) -> str:
    return sp.latex(_clean(expr))


def _append_degree_block(lines: list[str], n: int, nu_sym: sp.Symbol, a_sym: sp.Symbol, re_sym: sp.Symbol) -> None:
    coeffs = harmonic_coefficients(n)
    jn_sym = sp.symbols(f"J_{n}", real=True)
    eps = eps_n(n, nu_sym, a_sym, re_sym, jn_sym)

    lines.extend(
        [
            "",
            rf"\subsection*{{$J_{n}$ explicit coefficients}}",
            r"\begin{equation}",
            rf"\varepsilon_{n} = {sp.latex(eps)}.",
            r"\end{equation}",
        ]
    )

    if n % 2:
        g_terms = " + ".join(
            rf"\left({_latex(coeffs['g'][m])}\right)\cos({m}\omega)" for m in sorted(coeffs["g"])
        )
        q_terms = " + ".join(
            rf"\left({_latex(coeffs['Q'][m])}\right)\cos({m}\omega)" for m in sorted(coeffs["Q"])
        )
        psi_terms = " + ".join(
            rf"\left({_latex(coeffs['Psi'][m])}\right)\sin({m}\omega)" for m in sorted(coeffs["Psi"])
        )
        omega_terms = " + ".join(
            rf"\left({_latex(coeffs['Omega'][m])}\right)\sin({m}\omega)" for m in sorted(coeffs["Omega"])
        )
    else:
        g_terms = " + ".join(
            rf"\left({_latex(coeffs['g'][m])}\right)\sin({m}\omega)" for m in sorted(coeffs["g"])
        )
        q_terms = " + ".join(
            rf"\left({_latex(coeffs['Q'][m])}\right)\sin({m}\omega)" for m in sorted(coeffs["Q"])
        )
        psi_parts = []
        if 0 in coeffs["Psi"]:
            psi_parts.append(rf"\left({_latex(coeffs['Psi'][0])}\right)")
        psi_parts.extend(
            rf"\left({_latex(coeffs['Psi'][m])}\right)\cos({m}\omega)"
            for m in sorted(k for k in coeffs["Psi"] if k != 0)
        )
        omega_parts = []
        if 0 in coeffs["Omega"]:
            omega_parts.append(rf"\left({_latex(coeffs['Omega'][0])}\right)")
        omega_parts.extend(
            rf"\left({_latex(coeffs['Omega'][m])}\right)\cos({m}\omega)"
            for m in sorted(k for k in coeffs["Omega"] if k != 0)
        )
        psi_terms = " + ".join(psi_parts) if psi_parts else "0"
        omega_terms = " + ".join(omega_parts) if omega_parts else "0"

    lines.extend(
        [
            r"\begingroup\footnotesize",
            r"\begin{align}",
            rf"\dot{{\bar g}}_{{J_{n}}} &= \varepsilon_{n} \left({g_terms if g_terms else '0'}\right),\\",
            rf"\dot{{\bar Q}}_{{J_{n}}} &= \varepsilon_{n} \left({q_terms if q_terms else '0'}\right),\\",
            rf"\dot{{\bar\Psi}}_{{J_{n}}} &= \varepsilon_{n} \left({psi_terms if psi_terms else '0'}\right),\\",
            rf"\dot{{\bar\Omega}}_{{J_{n}}} &= \varepsilon_{n} \left({omega_terms if omega_terms else '0'}\right).",
            r"\end{align}",
            r"\endgroup",
        ]
    )


def write_note() -> None:
    nu_sym, a_sym, re_sym = sp.symbols("nu a R_e", positive=True, real=True)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=1.5cm]{geometry}",
        r"\usepackage{amsmath,amssymb,bm}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
        r"\small",
        r"\section*{Exact Symbolic Averaged GEqOE Zonal Generator}",
        r"For the isolated degree-$n$ zonal contribution we define",
        r"\begin{equation}",
        r"F(z;q)=\frac{z-q}{1-qz}, \qquad",
        r"D(z;q)=\frac{(z-q)(1-qz)}{(1+q^2)z}, \qquad",
        r"C(Q)=\frac{Q}{i(1+Q^2)},",
        r"\end{equation}",
        r"with $z=e^{iG}$, $q=(1-\beta)/g=g/(1+\beta)$, and $\omega=\Psi-\Omega$.",
        r"The exact Laurent expansion step is",
        r"\begin{align}",
        r"P_n\!\left(C(Q)\left[wF-w^{-1}F^{-1}\right]\right) &= \sum_{m\equiv n\!\!\!\!\pmod{2}} \Pi_{n,m}(Q)\,w^m F^m,\\",
        r"P_n'\!\left(C(Q)\left[wF-w^{-1}F^{-1}\right]\right) &= \sum_{m\equiv n-1\!\!\!\!\pmod{2}} \Delta_{n,m}(Q)\,w^m F^m,",
        r"\end{align}",
        r"where $w=e^{i\omega}$ and the coefficients are the exact finite sums",
        r"\begin{align}",
        r"\Pi_{n,m}(Q) &= \sum_{\substack{r=0\\ r\equiv m\!\!\!\!\pmod{2}}}^{n}",
        r"a_{n,r}\,C(Q)^r",
        r"\binom{r}{\frac{r+m}{2}}(-1)^{\frac{r-m}{2}},\\",
        r"\Delta_{n,m}(Q) &= \sum_{\substack{r=0\\ r\equiv m\!\!\!\!\pmod{2}}}^{n-1}",
        r"b_{n,r}\,C(Q)^r",
        r"\binom{r}{\frac{r+m}{2}}(-1)^{\frac{r-m}{2}},",
        r"\end{align}",
        r"with $P_n(x)=\sum_r a_{n,r}x^r$ and $P_n'(x)=\sum_r b_{n,r}x^r$.",
        r"The anomaly average then collapses to the single interior pole $z=q$.",
        r"For the recurring kernels one obtains the exact derivative formulas",
        r"\begin{align}",
        r"\mathcal{R}_{n,m}(q) &= \frac{(1+q^2)^n}{(n-m-1)!}",
        r"\left.\frac{d^{\,n-m-1}}{dz^{\,n-m-1}}\left(\frac{z^{n-1}}{(1-qz)^{n+m}}\right)\right|_{z=q},\\",
        r"\mathcal{G}_{n,m}(q) &= \frac{(n-1)(1+q^2)^n}{2i\,(n-m-1)!}",
        r"\left.\frac{d^{\,n-m-1}}{dz^{\,n-m-1}}\left(\frac{z^{n-2}(z^2-1)}{(1-qz)^{n+m}}\right)\right|_{z=q},\\",
        r"\mathcal{P}_{n,m}(q) &= -\frac{(1+q^2)^{n+1}}{4q(1-q^2)(n-m-1)!}",
        r"\left.\frac{d^{\,n-m-1}}{dz^{\,n-m-1}}",
        r"\left(\frac{z^{n-2}\left[4nqz+(n-1)(1+q^2)(z^2+1)\right]}{(1-qz)^{n+m}}\right)\right|_{z=q}.",
        r"\end{align}",
        r"These formulas immediately show that the isolated degree-$n$ averaged drift cannot reach harmonic $n$:",
        r"the residue vanishes whenever $m\ge n$, so the highest surviving harmonic is $n-2$.",
    ]

    _append_degree_block(lines, 5, nu_sym, a_sym, re_sym)

    lines.extend([r"\end{document}"])
    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    write_note()
