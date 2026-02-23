"""Side-by-side comparison of legacy vs staged GEqOE J2 Taylor propagator.

Prints full-precision output for several dt values between 10 and 300 s,
across all four Taylor orders, so that bit-level parity can be inspected
visually.
"""

from __future__ import annotations

import numpy as np

from astrodyn_core.propagation.geqoe.conversion import rv2geqoe
from astrodyn_core.propagation.geqoe.core import j2_taylor_propagator

# ── Physical constants (Earth) ──────────────────────────────────────
J2 = 1.08262668e-3
R_eq = 6_378_137.0          # m
mu = 3.986004418e14          # m^3 s^-2
body = (J2, R_eq, mu)

# ── Reference Cartesian state → GEqOE ──────────────────────────────
rv0 = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7500.0, 1000.0])
eq0_tuple = rv2geqoe(t=0.0, y=rv0, p=body)
eq0 = np.hstack([e.flatten() for e in eq0_tuple])

# ── Time grid ───────────────────────────────────────────────────────
dt_values = np.array([10.0, 120.0, 300.0])

# ── GEqOE element labels ───────────────────────────────────────────
LABELS = ["nu", "q1", "q2", "p1", "p2", "Lr"]

SEP = "-" * 120


def fmt(v: float) -> str:
    """Full-precision representation (17 significant digits)."""
    return f"{v: .17e}"


def run_order(order: int) -> None:
    print(f"\n{'=' * 120}")
    print(f"  ORDER {order}")
    print(f"{'=' * 120}")

    y_leg, stm_leg, mc_leg = j2_taylor_propagator(
        dt=dt_values, y0=eq0, p=body, order=order, backend="legacy",
    )
    y_stg, stm_stg, mc_stg = j2_taylor_propagator(
        dt=dt_values, y0=eq0, p=body, order=order, backend="staged",
    )

    for idx, dt in enumerate(dt_values):
        print(f"\n  dt = {dt:.1f} s")
        print(f"  {'elem':<4}  {'legacy':>24}  {'staged':>24}  {'diff':>24}")
        print(f"  {SEP}")
        for j, lbl in enumerate(LABELS):
            vl = y_leg[idx, j]
            vs = y_stg[idx, j]
            d = vs - vl
            print(f"  {lbl:<4}  {fmt(vl)}  {fmt(vs)}  {fmt(d)}")

        # Full 6x6 STM comparison
        print(f"\n  STM [6x6] (legacy → staged → diff):")
        for r in range(6):
            for c in range(6):
                sl = stm_leg[r, c, idx]
                ss = stm_stg[r, c, idx]
                d = ss - sl
                tag = " *" if d != 0.0 else ""
                print(f"    [{r},{c}]  {fmt(sl)}  {fmt(ss)}  {fmt(d)}{tag}")


def main() -> None:
    np.set_printoptions(precision=17, linewidth=200)

    print("GEqOE J2 Taylor Propagator — Legacy vs Staged comparison")
    print(f"Initial Cartesian state: {rv0}")
    print(f"Initial GEqOE state:     {eq0}")
    print(f"Body:  J2={J2},  Re={R_eq} m,  mu={mu} m^3/s^2")
    print(f"dt values (s): {dt_values}")

    for order in range(1, 5):
        run_order(order)

    print(f"\n{'=' * 120}")
    print("Done.")


if __name__ == "__main__":
    main()
