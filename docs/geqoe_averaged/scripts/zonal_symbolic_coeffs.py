#!/usr/bin/env python
"""Write explicit symbolic averaged GEqOE zonal coefficients to LaTeX.

This note is generated from the exact degree-n symbolic machinery in
``scripts/zonal_symbolic_general.py``. It records explicit closed formulas for
J3, J4, and J5.

Run:
  conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/zonal_symbolic_coeffs.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp

SCRIPT_DIR = Path(__file__).resolve().parent
DOC_DIR = SCRIPT_DIR.parent
if str(DOC_DIR) not in sys.path:
    sys.path.insert(0, str(DOC_DIR))

from geqoe_mean.symbolic import Q, harmonic_coefficients, q


OUT_TEX = DOC_DIR / "zonal_symbolic_coeffs.tex"

omega = sp.symbols("omega", real=True)
nu, a, Re = sp.symbols("nu a R_e", positive=True, real=True)
J3, J4, J5 = sp.symbols("J_3 J_4 J_5", real=True)

eps3 = nu * J3 * (Re / a) ** 3
eps4 = nu * J4 * (Re / a) ** 4
eps5 = nu * J5 * (Re / a) ** 5

_j3 = harmonic_coefficients(3)
_j4 = harmonic_coefficients(4)

G31 = _j3["g"][1]
Q31 = _j3["Q"][1]
P31 = _j3["Psi"][1]
O31 = _j3["Omega"][1]

G42 = _j4["g"][2]
Q42 = _j4["Q"][2]
P40 = _j4["Psi"][0]
P42 = _j4["Psi"][2]
O40 = _j4["Omega"][0]
O42 = _j4["Omega"][2]


def _latex(expr: sp.Expr) -> str:
    return sp.latex(sp.factor(sp.cancel(sp.simplify(expr))))


def _write_note() -> None:
    j5 = harmonic_coefficients(5)

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,landscape,margin=1.5cm]{geometry}",
        r"\usepackage{amsmath,amssymb,bm}",
        r"\allowdisplaybreaks",
        r"\begin{document}",
        r"\small",
        r"\section*{Explicit Symbolic Averaged GEqOE Zonal Coefficients}",
        r"This note records explicit closed formulas generated from the exact degree-$n$ symbolic residue machinery.",
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
        rf"\varepsilon_3 = {sp.latex(eps3)}, \qquad \varepsilon_4 = {sp.latex(eps4)}, \qquad \varepsilon_5 = {sp.latex(eps5)}.",
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
        r"\subsection*{$J_5$ contribution}",
        r"\begin{align}",
        r"\dot{\bar g}_{J_5} &= \varepsilon_5 \mathcal{G}_{5,1}(q,Q)\cos\omega + \varepsilon_5 \mathcal{G}_{5,3}(q,Q)\cos(3\omega),\\",
        r"\dot{\bar Q}_{J_5} &= \varepsilon_5 \mathcal{Q}_{5,1}(q,Q)\cos\omega + \varepsilon_5 \mathcal{Q}_{5,3}(q,Q)\cos(3\omega),\\",
        r"\dot{\bar\Psi}_{J_5} &= \varepsilon_5 \mathcal{P}_{5,1}(q,Q)\sin\omega + \varepsilon_5 \mathcal{P}_{5,3}(q,Q)\sin(3\omega),\\",
        r"\dot{\bar\Omega}_{J_5} &= \varepsilon_5 \mathcal{O}_{5,1}(q,Q)\sin\omega + \varepsilon_5 \mathcal{O}_{5,3}(q,Q)\sin(3\omega).",
        r"\end{align}",
        r"\begingroup\footnotesize",
        r"\begin{align}",
        rf"\mathcal{{G}}_{{5,1}}(q,Q) &= {_latex(j5['g'][1])},\\",
        rf"\mathcal{{G}}_{{5,3}}(q,Q) &= {_latex(j5['g'][3])},\\",
        rf"\mathcal{{Q}}_{{5,1}}(q,Q) &= {_latex(j5['Q'][1])},\\",
        rf"\mathcal{{Q}}_{{5,3}}(q,Q) &= {_latex(j5['Q'][3])},\\",
        rf"\mathcal{{P}}_{{5,1}}(q,Q) &= {_latex(j5['Psi'][1])},\\",
        rf"\mathcal{{P}}_{{5,3}}(q,Q) &= {_latex(j5['Psi'][3])},\\",
        rf"\mathcal{{O}}_{{5,1}}(q,Q) &= {_latex(j5['Omega'][1])},\\",
        rf"\mathcal{{O}}_{{5,3}}(q,Q) &= {_latex(j5['Omega'][3])}.",
        r"\end{align}",
        r"\endgroup",
        r"",
        r"The exact generator also explains a sharper structural fact than the earlier numerical probe:",
        r"for an isolated degree-$n$ zonal term, the highest surviving averaged harmonic is $n-2$, not $n$.",
        r"\end{document}",
    ]

    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    _write_note()
