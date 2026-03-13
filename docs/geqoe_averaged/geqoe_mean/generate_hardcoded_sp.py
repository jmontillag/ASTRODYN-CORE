"""Generate CSE-optimized hardcoded short-period batch function.

NOTE: This script is SUPERSEDED by Stage C.2 (heyoka SP cfunc) which is
implemented directly in ``heyoka_compiled.py``.  The cfunc approach
recursively converts SHORT_DATA SymPy expressions to a heyoka expression
DAG (handling complex F→(cos_f, sin_f) and I natively), then LLVM-compiles
the entire SP map into a single SIMD-vectorized cfunc.  Build time is ~9s,
and the cfunc evaluates 3201 points in ~8ms (86x faster than the lambdified
path).  This script is retained only for reference.

Original approach (broken): F → cos_f + I*sin_f substitution followed by
sp.expand — hangs for ~20 min on large n=5 expressions.

Usage:
    conda run -n astrodyn-core-env python docs/geqoe_averaged/geqoe_mean/generate_hardcoded_sp.py
"""

from __future__ import annotations

import ast
import textwrap
import time
from pathlib import Path

import sympy as sp

# ── Symbols ──────────────────────────────────────────────────────────────

q_sym = sp.Symbol("q", real=True, positive=True)
Q_sym = sp.Symbol("Q", real=True, positive=True)
cos_f_sym = sp.Symbol("cos_f", real=True)
sin_f_sym = sp.Symbol("sin_f", real=True)
omega_sym = sp.Symbol("omega", real=True)
F_sym = sp.Symbol("F")  # placeholder, substituted away
s2, s3, s4, s5 = sp.symbols("s2 s3 s4 s5", real=True)

_SYMPIFY_LOCALS = {"q": q_sym, "Q": Q_sym, "F": F_sym, "I": sp.I}
_SCALE_MAP = {2: s2, 3: s3, 4: s4, 5: s5}
_VARIABLES = ("g", "Q", "Psi", "Omega", "M")


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_short_data() -> dict:
    """Load SHORT_DATA from generated_coefficients.py via AST."""
    data_file = Path(__file__).resolve().parent / "generated_coefficients.py"
    tree = ast.parse(data_file.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "SHORT_DATA"):
            return ast.literal_eval(node.value)
    raise RuntimeError("SHORT_DATA not found in generated_coefficients.py")


def _convert_one_expression(expr_str: str) -> tuple[sp.Expr, sp.Expr]:
    """Convert one SP expression from F-domain to real/imag parts.

    Returns (re_part, im_part) where both are functions of (q, Q, cos_f, sin_f).
    """
    expr = sp.sympify(expr_str, locals=_SYMPIFY_LOCALS)
    if expr == 0:
        return sp.Integer(0), sp.Integer(0)

    # Substitute F → cos_f + I*sin_f
    expr_trig = expr.subs(F_sym, cos_f_sym + sp.I * sin_f_sym)

    # Expand to separate real and imaginary components
    expr_expanded = sp.expand(expr_trig)

    re_part = sp.re(expr_expanded)
    im_part = sp.im(expr_expanded)

    return re_part, im_part


def _build_sp_correction_expressions(short_data: dict) -> dict[str, sp.Expr]:
    """Build 5 combined SP correction expressions in real arithmetic.

    Each correction is a function of (q, Q, cos_f, sin_f, omega, s2..s5).
    """
    corrections = {v: sp.Integer(0) for v in _VARIABLES}
    n_converted = 0
    n_total = sum(
        len(var_map.get(v, {}))
        for var_map in short_data.values()
        for v in _VARIABLES
    )

    for n_key in sorted(short_data.keys(), key=int):
        n = int(n_key)
        scale = _SCALE_MAP[n]
        var_map = short_data[n_key]

        for variable in _VARIABLES:
            coeffs = var_map.get(variable, {})
            for m_str, expr_str in coeffs.items():
                m = int(m_str)
                n_converted += 1
                t0 = time.time()

                re_part, im_part = _convert_one_expression(expr_str)

                if re_part == 0 and im_part == 0:
                    continue

                # Re(c_m * w^m) = Re(c_m)*cos(m*omega) - Im(c_m)*sin(m*omega)
                if m == 0:
                    contrib = re_part
                else:
                    contrib = sp.Integer(0)
                    if re_part != 0:
                        contrib += re_part * sp.cos(m * omega_sym)
                    if im_part != 0:
                        contrib -= im_part * sp.sin(m * omega_sym)

                corrections[variable] += scale * contrib

                dt = time.time() - t0
                if dt > 0.5:
                    print(f"    n={n}, {variable:6s}, m={m:+2d}: {dt:.1f}s "
                          f"[{n_converted}/{n_total}]")

    return corrections


# ── Code emission ────────────────────────────────────────────────────────

def _emit_expr(expr: sp.Expr) -> str:
    """Convert a SymPy expression to numpy-compatible Python string."""
    from sympy.printing.pycode import PythonCodePrinter

    class _Printer(PythonCodePrinter):
        def _print_Symbol(self, expr):
            return expr.name

    return _Printer().doprint(expr)


def generate_sp_batch_module(short_data: dict) -> str:
    """Generate the full hardcoded_sp.py source code."""
    print("  Converting 88 SP expressions to real trig...")
    t0 = time.time()
    corrections = _build_sp_correction_expressions(short_data)
    print(f"    done in {time.time() - t0:.1f}s")

    # Simplify before CSE
    print("  Simplifying combined expressions...")
    t0 = time.time()
    exprs = []
    labels = []
    for variable in _VARIABLES:
        e = corrections[variable]
        # cancel simplifies rational expressions; expand_trig is not needed
        # since we use cos_f/sin_f symbols directly
        e = sp.cancel(e)
        exprs.append(e)
        labels.append(f"d{variable}" if variable != "Q" else "dQ_rate")
    print(f"    done in {time.time() - t0:.1f}s")

    print(f"  Running CSE on {len(exprs)} expressions...")
    t0 = time.time()
    replacements, reduced = sp.cse(exprs, optimizations="basic")
    print(f"    CSE found {len(replacements)} shared subexpressions "
          f"in {time.time() - t0:.1f}s")

    # ── Emit Python module ──
    lines = [
        '"""CSE-optimized short-period corrections (auto-generated, do not edit).',
        '',
        'Generated by generate_hardcoded_sp.py.  Pure Python + math arithmetic.',
        'Provides short_period_cse() returning dimensionless SP corrections',
        '(dg, dQ, dPsi, dOmega, dM) before nu-scaling.',
        '"""',
        '',
        'import math',
        '',
        '',
        'def short_period_cse(cos_f, sin_f, q, Q, omega,',
        '                     s2, s3, s4, s5):',
        '    """Compute dimensionless short-period corrections via CSE.',
        '',
        '    Parameters',
        '    ----------',
        '    cos_f, sin_f : cos/sin of true anomaly',
        '    q : eccentricity parameter g/(1+beta)',
        '    Q : inclination parameter ||(q1,q2)||',
        '    omega : argument of perigee (Psi - Omega)',
        '    s2..s5 : scale factors Jn*(Re/a)^n',
        '',
        '    Returns',
        '    -------',
        '    (dg, dQ, dPsi, dOmega, dM) dimensionless corrections',
        '    """',
    ]

    # Emit CSE intermediates
    for sym, expr in replacements:
        lines.append(f"    {sym.name} = {_emit_expr(expr)}")

    lines.append("")
    for label, expr in zip(labels, reduced):
        lines.append(f"    {label} = {_emit_expr(expr)}")

    lines.append("")
    lines.append("    return dg, dQ_rate, dPsi, dOmega, dM")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Stage B.2: Generating CSE-optimized hardcoded short-period corrections...")
    short_data = _load_short_data()

    # Count non-zero terms
    nz = sum(
        1 for n_data in short_data.values()
        for v_data in n_data.values()
        for expr_str in v_data.values()
        if expr_str != "0" and expr_str != "'0'"
    )
    print(f"  Non-zero SP coefficients: {nz}")

    source = generate_sp_batch_module(short_data)

    out_path = Path(__file__).resolve().parent / "hardcoded_sp.py"
    out_path.write_text(source)
    print(f"  Wrote {out_path} ({len(source)} bytes)")

    # Quick smoke test
    print("  Smoke test...")
    ns = {}
    exec(compile(source, str(out_path), "exec"), ns)
    fn = ns["short_period_cse"]
    result = fn(0.9, 0.4, 0.01, 0.1, 0.5, 1e-3, 1e-6, 1e-9, 1e-12)
    print(f"    short_period_cse(cos_f=0.9, sin_f=0.4, ...) = {result}")
    print("  Done.")


if __name__ == "__main__":
    main()
