#!/usr/bin/env python3
"""Verify that the B̂ residue at F = -1/q vanishes for all zonal degrees.

This script checks the key structural property of the GEqOE short-period
decomposition: the residue D (= B̂ in the paper) at the pole F = -1/q is
identically zero in (q, Q) for every forcing variable and every ω-harmonic,
for zonal degrees n = 2, ..., N_max.

Physical motivation: F = -1/q corresponds to a/r → 0 (r → ∞), where the
zonal potential vanishes.  Since all GEqOE forcing functions contain (a/r)^n
as a factor, they vanish at this point.  The residue D measures whether the
partial fraction decomposition has a contribution from the pole at 1+qF = 0,
and its vanishing is required for the short-period map to be a rational
(not logarithmic) function of the complex eccentric longitude F.

The verification is fast because it only evaluates the numerator polynomial
N(F) at F = -1/q — no full antidifferentiation is needed.

Usage:
    conda run -n astrodyn-core-env python scripts/verify_bhat_residue.py [--nmax 15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure geqoe_mean is importable
PKG_DIR = Path(__file__).resolve().parent.parent / "geqoe_mean"
sys.path.insert(0, str(PKG_DIR.parent))

import sympy as sp

from geqoe_mean.short_period import (
    dimensionless_rate_series,
    q,
    Q,
    F,
    mean_f_power,
)
from geqoe_mean.direct_residue import (
    _build_combined_numerator,
    _fast_cancel,
    _poly_eval,
)


def _series_by_m(poly: dict) -> dict[int, dict[int, sp.Expr]]:
    """Group Laurent polynomial by ω-harmonic m."""
    from collections import defaultdict

    out: dict[int, dict[int, sp.Expr]] = defaultdict(dict)
    for (m, k), coeff in poly.items():
        out[m][k] = coeff
    return dict(out)


def _mean_rate_from_raw_series(raw: dict) -> dict[int, sp.Expr]:
    """Compute mean rates from raw Laurent series."""
    from collections import defaultdict

    out: defaultdict[int, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (m_val, k_val), coeff in raw.items():
        out[m_val] += coeff * mean_f_power(k_val)
    clean = sp.cancel
    return {m_val: clean(sp.together(v)) for m_val, v in out.items() if v != 0}


def verify_d_residue(n: int, variable: str) -> dict[int, sp.Expr]:
    """Check D residue at F = -1/q for degree n and variable.

    Returns dict mapping ω-harmonic m → D value (should be 0 for all).
    """
    raw = dimensionless_rate_series(variable, n)
    mean_coeffs = _mean_rate_from_raw_series(raw)
    by_m = _series_by_m(raw)

    results = {}
    for m_val, raw_by_k in sorted(by_m.items()):
        mean_coeff = mean_coeffs.get(m_val, sp.Integer(0))
        N_poly, shift = _build_combined_numerator(raw_by_k, mean_coeff)

        if N_poly == 0:
            results[m_val] = sp.Integer(0)
            continue

        # Polynomial long division: N = Q * denom + N_rem
        denom_expr = sp.expand(F**shift * (F + q) ** 2 * (1 + q * F) ** 2)
        N_sp = sp.Poly(N_poly, F)
        denom_sp = sp.Poly(denom_expr, F)

        if N_sp.degree() >= denom_sp.degree():
            _, N_rem_poly = sp.div(N_sp, denom_sp)
            N_rem = N_rem_poly.as_expr()
        else:
            N_rem = N_poly

        N_rem = sp.expand(N_rem)

        # Evaluate N_rem at F = -1/q
        N_rem_at_m1q = _poly_eval(N_rem, -sp.Integer(1) / q)
        N_rem_at_m1q = _fast_cancel(N_rem_at_m1q)

        # D = N_rem(-1/q) / [(-1/q)^s * ((-1/q)+q)^2]
        m1q_s = (-sp.Integer(1) / q) ** shift
        m1q_pq_sq = ((-sp.Integer(1) / q) + q) ** 2  # = ((q^2-1)/q)^2

        if N_rem_at_m1q == 0:
            D = sp.Integer(0)
        else:
            D = _fast_cancel(N_rem_at_m1q / (m1q_s * m1q_pq_sq))

        results[m_val] = D

    return results


VARIABLES = ["g", "Q", "Psi", "Omega", "M"]


def main():
    parser = argparse.ArgumentParser(description="Verify B̂ residue vanishing")
    parser.add_argument("--nmax", type=int, default=10, help="Maximum zonal degree")
    args = parser.parse_args()

    print(f"Verifying B̂ (D residue at F = -1/q) = 0 for n = 2 .. {args.nmax}")
    print("=" * 70)

    all_pass = True
    total_checks = 0

    for n in range(2, args.nmax + 1):
        t0 = time.time()
        n_checks = 0
        n_pass = True

        for var in VARIABLES:
            d_results = verify_d_residue(n, var)
            for m_val, D in sorted(d_results.items()):
                total_checks += 1
                n_checks += 1
                if D != 0:
                    print(f"  FAIL: n={n}, var={var}, m={m_val}: D = {D}")
                    n_pass = False
                    all_pass = False

        elapsed = time.time() - t0
        status = "PASS" if n_pass else "FAIL"
        print(f"  n = {n:2d}:  {status}  ({n_checks} harmonics checked, {elapsed:.1f}s)")

    print("=" * 70)
    print(f"Total: {total_checks} residues checked, n = 2 .. {args.nmax}")
    if all_pass:
        print("RESULT: ALL B̂ residues vanish identically in (q, Q).")
    else:
        print("RESULT: SOME B̂ residues are NONZERO — conjecture is FALSE.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
