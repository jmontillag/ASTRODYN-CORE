#!/usr/bin/env python
"""Write explicit symbolic J3/J4 averaged GEqOE coefficient formulas to LaTeX.

These formulas were derived through the residue route in z = exp(i G) after:
1. expanding the zonal forcing in the relative angle u = omega + f,
2. projecting onto the parity-allowed omega harmonics,
3. evaluating the resulting anomaly averages as residues at z = 0 and z = q.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/zonal_symbolic_coeffs.py
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp


OUT_TEX = Path(__file__).with_name("zonal_symbolic_coeffs.tex")

q, Q, omega = sp.symbols("q Q omega", positive=True, real=True)
nu, a, Re, J3, J4 = sp.symbols("nu a R_e J_3 J_4", positive=True, real=True)

eps3 = nu * J3 * (Re / a) ** 3
eps4 = nu * J4 * (Re / a) ** 4

G31 = -3 * Q * (q**2 + 1) ** 4 * (Q**2 - Q - 1) * (Q**2 + Q - 1) / (
    (Q**2 + 1) ** 3 * (q - 1) ** 4 * (q + 1) ** 4
)
Q31 = -3 * q * (Q - 1) * (Q + 1) * (q**2 + 1) ** 5 * (Q**2 - Q - 1) * (Q**2 + Q - 1) / (
    2 * (Q**2 + 1) ** 2 * (q - 1) ** 6 * (q + 1) ** 6
)
O31 = -3 * q * (Q - 1) * (Q + 1) * (q**2 + 1) ** 5 * (Q**4 - 13 * Q**2 + 1) / (
    2 * Q * (Q**2 + 1) ** 2 * (q - 1) ** 6 * (q + 1) ** 6
)
P31 = 3 * Q * (q**2 + 1) ** 5 * (
    -2 * Q**6 * q**2 + Q**4 * q**4 + 46 * Q**4 * q**2 + Q**4
    - 3 * Q**2 * q**4 - 82 * Q**2 * q**2 - 3 * Q**2 + q**4 + 20 * q**2 + 1
) / (2 * q * (Q**2 + 1) ** 3 * (q - 1) ** 6 * (q + 1) ** 6)

G42 = -15 * Q**2 * q * (q**2 + 1) ** 5 * (3 * Q**4 - 8 * Q**2 + 3) / (
    2 * (Q**2 + 1) ** 4 * (q - 1) ** 6 * (q + 1) ** 6
)
Q42 = -15 * Q * q**2 * (Q - 1) * (Q + 1) * (q**2 + 1) ** 6 * (3 * Q**4 - 8 * Q**2 + 3) / (
    4 * (Q**2 + 1) ** 3 * (q - 1) ** 8 * (q + 1) ** 8
)
O40 = -15 * (Q - 1) * (Q + 1) * (q**2 + 1) ** 6 * (Q**4 - 5 * Q**2 + 1) * (q**4 + 8 * q**2 + 1) / (
    4 * (Q**2 + 1) ** 3 * (q - 1) ** 8 * (q + 1) ** 8
)
O42 = 15 * q**2 * (Q - 1) * (Q + 1) * (q**2 + 1) ** 6 * (3 * Q**4 - 22 * Q**2 + 3) / (
    4 * (Q**2 + 1) ** 3 * (q - 1) ** 8 * (q + 1) ** 8
)
P40 = -15 * (q**2 + 1) ** 6 * (
    3 * Q**8 * q**4 + 21 * Q**8 * q**2 + 3 * Q**8
    - 28 * Q**6 * q**4 - 176 * Q**6 * q**2 - 28 * Q**6
    + 48 * Q**4 * q**4 + 276 * Q**4 * q**2 + 48 * Q**4
    - 18 * Q**2 * q**4 - 96 * Q**2 * q**2 - 18 * Q**2 + q**4 + 5 * q**2 + 1
) / (4 * (Q**2 + 1) ** 4 * (q - 1) ** 8 * (q + 1) ** 8)
P42 = -15 * Q**2 * (q**2 + 1) ** 6 * (
    -6 * Q**6 * q**2 + 3 * Q**4 * q**4 + 86 * Q**4 * q**2 + 3 * Q**4
    - 8 * Q**2 * q**4 - 146 * Q**2 * q**2 - 8 * Q**2 + 3 * q**4 + 42 * q**2 + 3
) / (4 * (Q**2 + 1) ** 4 * (q - 1) ** 8 * (q + 1) ** 8)


def _latex(expr: sp.Expr) -> str:
    return sp.latex(sp.factor(expr))


def _write_note() -> None:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=1.5cm]{geometry}",
        r"\usepackage{amsmath,amssymb,bm}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
        r"\small",
        r"\section*{Explicit Symbolic Averaged GEqOE Zonal Coefficients}",
        r"This note records the first non-$J_2$ symbolic averaged GEqOE zonal coefficients obtained through the residue route.",
        r"We use",
        r"\begin{equation}",
        r"q = \frac{1-\beta}{g} = \frac{g}{1+\beta}, \qquad",
        r"g = \frac{2q}{1+q^2}, \qquad",
        r"\beta = \frac{1-q^2}{1+q^2}, \qquad",
        r"Q=\tan\frac{i}{2}, \qquad",
        r"\omega = \Psi - \Omega,",
        r"\end{equation}",
        r"and the degree-scaled small parameters",
        r"\begin{equation}",
        rf"\varepsilon_3 = {sp.latex(eps3)}, \qquad \varepsilon_4 = {sp.latex(eps4)}.",
        r"\end{equation}",
        r"",
        r"\subsection*{$J_3$ contribution}",
        r"\begin{align}",
        r"\dot{\bar g}_{J_3} &= \varepsilon_3 \mathcal{G}_{3,1}(q,Q)\cos\omega,\\",
        r"\dot{\bar Q}_{J_3} &= \varepsilon_3 \mathcal{Q}_{3,1}(q,Q)\cos\omega,\\",
        r"\dot{\bar\Psi}_{J_3} &= \varepsilon_3 \mathcal{P}_{3,1}(q,Q)\sin\omega,\\",
        r"\dot{\bar\Omega}_{J_3} &= \varepsilon_3 \mathcal{O}_{3,1}(q,Q)\sin\omega.",
        r"\end{align}",
        r"\begingroup\footnotesize",
        r"\begin{align}",
        rf"\mathcal{{G}}_{{3,1}}(q,Q) &= {_latex(G31)},\\",
        rf"\mathcal{{Q}}_{{3,1}}(q,Q) &= {_latex(Q31)},\\",
        rf"\mathcal{{P}}_{{3,1}}(q,Q) &= {_latex(P31)},\\",
        rf"\mathcal{{O}}_{{3,1}}(q,Q) &= {_latex(O31)}.",
        r"\end{align}",
        r"\endgroup",
        r"",
        r"\subsection*{$J_4$ contribution}",
        r"\begin{align}",
        r"\dot{\bar g}_{J_4} &= \varepsilon_4 \mathcal{G}_{4,2}(q,Q)\sin(2\omega),\\",
        r"\dot{\bar Q}_{J_4} &= \varepsilon_4 \mathcal{Q}_{4,2}(q,Q)\sin(2\omega),\\",
        r"\dot{\bar\Psi}_{J_4} &= \varepsilon_4 \mathcal{P}_{4,0}(q,Q) + \varepsilon_4 \mathcal{P}_{4,2}(q,Q)\cos(2\omega),\\",
        r"\dot{\bar\Omega}_{J_4} &= \varepsilon_4 \mathcal{O}_{4,0}(q,Q) + \varepsilon_4 \mathcal{O}_{4,2}(q,Q)\cos(2\omega).",
        r"\end{align}",
        r"\begingroup\footnotesize",
        r"\begin{align}",
        rf"\mathcal{{G}}_{{4,2}}(q,Q) &= {_latex(G42)},\\",
        rf"\mathcal{{Q}}_{{4,2}}(q,Q) &= {_latex(Q42)},\\",
        rf"\mathcal{{P}}_{{4,0}}(q,Q) &= {_latex(P40)},\\",
        rf"\mathcal{{P}}_{{4,2}}(q,Q) &= {_latex(P42)},\\",
        rf"\mathcal{{O}}_{{4,0}}(q,Q) &= {_latex(O40)},\\",
        rf"\mathcal{{O}}_{{4,2}}(q,Q) &= {_latex(O42)}.",
        r"\end{align}",
        r"\endgroup",
        r"",
        r"These formulas match the parity structure inferred earlier from the finite Fourier model:",
        r"$J_3$ produces first-harmonic odd-angle coupling, while $J_4$ produces constant and second-harmonic even-angle coupling.",
        r"\end{document}",
    ]

    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    _write_note()
