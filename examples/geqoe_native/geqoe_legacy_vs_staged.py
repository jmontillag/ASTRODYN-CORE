"""Side-by-side comparison of legacy vs refactored GEqOE J2 Taylor propagator.

Two sections:

1. **Parity check** -- prints full-precision output for several dt values
   between 10 and 300 s, across all four Taylor orders, so that bit-level
   parity can be inspected visually.

2. **Multi-epoch benchmark** -- measures the wall-clock time of three
   propagation strategies when the same initial state is evaluated at many
   different time points:

   * ``legacy``   -- original monolithic propagator (recomputes everything per call)
   * ``monolithic`` -- new refactored propagator called once with the full grid
   * ``staged``   -- new two-stage API: ``prepare_taylor_coefficients`` once,
                     then ``evaluate_taylor`` per time grid

   The staged approach should be noticeably faster for large numbers of
   independent calls because the expensive coefficient computation is paid
   only once.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from astrodyn_core.propagation.geqoe._legacy_loader import import_legacy_module
from astrodyn_core.propagation.geqoe.conversion import rv2geqoe
from astrodyn_core.propagation.geqoe.core import (
    evaluate_taylor,
    j2_taylor_propagator,
    prepare_taylor_coefficients,
)

legacy_propagator = import_legacy_module("temp_mosaic_modules.geqoe_utils.propagator")

# -- Physical constants (Earth) ------------------------------------------
J2 = 1.08262668e-3
R_eq = 6_378_137.0          # m
mu = 3.986004418e14          # m^3 s^-2
body = (J2, R_eq, mu)

# -- Reference Cartesian state -> GEqOE ---------------------------------
rv0 = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7500.0, 1000.0])
eq0_tuple = rv2geqoe(t=0.0, y=rv0, p=body)
eq0 = np.hstack([e.flatten() for e in eq0_tuple])

# -- Time grid for parity check -----------------------------------------
dt_values = np.array([10.0, 120.0, 300.0])

# -- GEqOE element labels -----------------------------------------------
LABELS = ["nu", "q1", "q2", "p1", "p2", "Lr"]

SEP = "-" * 120


def fmt(v: float) -> str:
    """Full-precision representation (17 significant digits)."""
    return f"{v: .17e}"


# -----------------------------------------------------------------------
# Section 1: Parity check
# -----------------------------------------------------------------------

def run_order_parity(order: int) -> None:
    print(f"\n{'=' * 120}")
    print(f"  ORDER {order}")
    print(f"{'=' * 120}")

    y_leg, stm_leg, mc_leg = legacy_propagator.j2_taylor_propagator(
        dt=dt_values, y0=eq0, p=body, order=order,
    )
    y_stg, stm_stg, mc_stg = j2_taylor_propagator(
        dt=dt_values, y0=eq0, p=body, order=order,
    )

    for idx, dt in enumerate(dt_values):
        print(f"\n  dt = {dt:.1f} s")
        print(f"  {'elem':<4}  {'legacy':>24}  {'refactored':>24}  {'diff':>24}")
        print(f"  {SEP}")
        for j, lbl in enumerate(LABELS):
            vl = y_leg[idx, j]
            vs = y_stg[idx, j]
            d = vs - vl
            print(f"  {lbl:<4}  {fmt(vl)}  {fmt(vs)}  {fmt(d)}")

        # Full 6x6 STM comparison
        print(f"\n  STM [6x6] (legacy -> refactored -> diff):")
        for r in range(6):
            for c in range(6):
                sl = stm_leg[r, c, idx]
                ss = stm_stg[r, c, idx]
                d = ss - sl
                tag = " *" if d != 0.0 else ""
                print(f"    [{r},{c}]  {fmt(sl)}  {fmt(ss)}  {fmt(d)}{tag}")


# -----------------------------------------------------------------------
# Section 2: Multi-epoch benchmark
# -----------------------------------------------------------------------

def _timeit(fn, n_calls: int) -> float:
    """Return wall-clock seconds for ``n_calls`` calls to ``fn()``."""
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    return time.perf_counter() - t0


def run_benchmark(order: int = 4, n_grids: int = 50, grid_size: int = 20, n_repeat: int = 5) -> None:
    """Benchmark legacy / monolithic / staged across repeated independent time grids.

    The scenario modelled here mirrors a realistic trajectory-analysis workflow:
    the *same* initial state is propagated to many *different* time grids
    (e.g. successive sensor-tasking windows, Monte-Carlo sample times, or
    filter update epochs).  Each grid is a small batch of epochs.

    Three strategies are timed for ``n_grids`` independent grids of
    ``grid_size`` epochs each:

    ``legacy``
        Each grid calls the original monolithic propagator from scratch
        (coefficient computation + evaluation every time).

    ``monolithic``
        Each grid calls the refactored ``j2_taylor_propagator`` from scratch
        (same cost structure as legacy -- no pre-computation).

    ``staged``
        Coefficients are computed **once** with
        ``prepare_taylor_coefficients``; each grid calls only
        ``evaluate_taylor`` (cheap polynomial evaluation).

    Parameters
    ----------
    order:
        Taylor expansion order (1-4).
    n_grids:
        Number of independent time grids (call groups).
    grid_size:
        Number of epochs per grid.
    n_repeat:
        Timing repetitions; the minimum is reported (warm-cache estimate).
    """
    rng = np.random.default_rng(42)
    # Generate ``n_grids`` independent time grids
    grids = [rng.uniform(10.0, 3600.0, size=grid_size) for _ in range(n_grids)]

    header = (
        f"\n{'=' * 72}\n"
        f"  MULTI-GRID BENCHMARK  |  order={order}"
        f"  |  {n_grids} grids x {grid_size} epochs/grid\n"
        f"{'=' * 72}"
    )
    print(header)

    # ----------------------------------------------------------------
    # Strategy A: legacy -- recompute from scratch for every grid
    # ----------------------------------------------------------------
    def _legacy_all_grids():
        for grid in grids:
            legacy_propagator.j2_taylor_propagator(
                dt=grid, y0=eq0, p=body, order=order,
            )

    # ----------------------------------------------------------------
    # Strategy B: monolithic -- refactored, same cost structure
    # ----------------------------------------------------------------
    def _monolithic_all_grids():
        for grid in grids:
            j2_taylor_propagator(dt=grid, y0=eq0, p=body, order=order)

    # ----------------------------------------------------------------
    # Strategy C: staged -- prepare coefficients once, evaluate per grid
    # ----------------------------------------------------------------
    def _staged_all_grids():
        coeffs = prepare_taylor_coefficients(y0=eq0, p=body, order=order)
        for grid in grids:
            evaluate_taylor(coeffs, grid)

    strategies = [
        ("legacy   (per-grid)",     _legacy_all_grids),
        ("monolithic (per-grid)",   _monolithic_all_grids),
        ("staged  (per-grid)",      _staged_all_grids),
    ]

    times: dict[str, float] = {}
    for label, fn in strategies:
        best = float("inf")
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        times[label] = best

    baseline = times["legacy   (per-grid)"]
    print(f"\n  {'Strategy':<28}  {'Time (ms)':>10}  {'vs legacy':>10}")
    print(f"  {'-' * 28}  {'-' * 10}  {'-' * 10}")
    for label, t in times.items():
        speedup = baseline / t
        marker = "  <-- baseline" if label == "legacy   (per-grid)" else (
            f"  {speedup:.2f}x faster" if speedup >= 1.0 else f"  {1/speedup:.2f}x slower"
        )
        print(f"  {label:<28}  {t * 1000:>10.2f}  {speedup:>9.2f}x{marker}")

    # Verify numerical parity for the last grid
    last_grid = grids[-1]
    y_mono, _, _ = j2_taylor_propagator(dt=last_grid, y0=eq0, p=body, order=order)
    coeffs_ref = prepare_taylor_coefficients(y0=eq0, p=body, order=order)
    y_staged, _, _ = evaluate_taylor(coeffs_ref, last_grid)

    max_diff = np.max(np.abs(y_staged - y_mono))
    print(f"\n  Numerical parity (staged vs monolithic, max |diff|): {max_diff:.2e}")
    if max_diff < 1e-10:
        print("  [OK] Results are numerically identical.")
    else:
        print("  [WARN] Non-trivial difference detected!", file=sys.stderr)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    np.set_printoptions(precision=17, linewidth=200)

    print("GEqOE J2 Taylor Propagator -- Legacy vs Refactored comparison")
    print(f"Initial Cartesian state: {rv0}")
    print(f"Initial GEqOE state:     {eq0}")
    print(f"Body:  J2={J2},  Re={R_eq} m,  mu={mu} m^3/s^2")
    print(f"dt values (s): {dt_values}")

    # --- Section 1: parity ---
    print("\n" + "=" * 120)
    print("  SECTION 1: Bit-level parity check (legacy vs refactored)")
    print("=" * 120)
    for order in range(1, 5):
        run_order_parity(order)

    # --- Section 2: benchmark ---
    print("\n" + "=" * 120)
    print("  SECTION 2: Multi-epoch performance benchmark")
    print("  (staged = prepare_taylor_coefficients once + evaluate_taylor per call)")
    print("=" * 120)
    for order in range(1, 5):
        run_benchmark(order=order, n_grids=50, grid_size=20, n_repeat=5)

    print(f"\n{'=' * 120}")
    print("Done.")


if __name__ == "__main__":
    main()
