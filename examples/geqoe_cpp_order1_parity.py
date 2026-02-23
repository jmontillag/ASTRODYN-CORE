"""Order-1 parity check: Python staged GEqOE propagator vs C++ staged implementation.

This example compares the previous Python staged implementation
(`astrodyn_core.propagation.geqoe.core`) against the new C++ staged
implementation (`astrodyn_core.geqoe_cpp`) for multiple dt grids.

Run:
    conda run -n astrodyn-core-env python examples/geqoe_cpp_order1_parity.py
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from astrodyn_core.geqoe_cpp import evaluate_taylor_cpp, prepare_taylor_coefficients_cpp
from astrodyn_core.propagation.geqoe import core as py_core
from astrodyn_core.propagation.geqoe.conversion import rv2geqoe

_J2 = 0.0010826266835531513
_Re = 6378137.0
_mu = 3.986004418e14


def _reference_cart_state() -> np.ndarray:
    return np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


def _to_geqoe_state(y0_cart: np.ndarray) -> np.ndarray:
    eq0_tuple = rv2geqoe(t=0.0, y=y0_cart, p=(_J2, _Re, _mu))
    return np.hstack([component.flatten() for component in eq0_tuple])


def _dt_cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    return {
        "short_uniform": np.array([0.0, 10.0, 30.0, 60.0], dtype=float),
        "mixed_sparse": np.array([0.0, 15.0, 75.0, 600.0], dtype=float),
        "dense_window": np.linspace(0.0, 120.0, 25, dtype=float),
        "random_sorted": np.sort(rng.uniform(0.0, 900.0, 20)).astype(float),
    }


def _compare_case(name: str, dt: np.ndarray, eq0: np.ndarray) -> None:
    coeffs_py = py_core.prepare_taylor_coefficients(y0=eq0, p=(_J2, _Re, _mu), order=1)
    y_py, stm_py, map_py = py_core.evaluate_taylor(coeffs_py, dt)

    coeffs_cpp = prepare_taylor_coefficients_cpp(eq0, _J2, _Re, _mu, order=1)
    y_cpp, stm_cpp, map_cpp = evaluate_taylor_cpp(coeffs_cpp, dt)

    npt.assert_allclose(y_cpp, y_py, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(stm_cpp, stm_py, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(map_cpp, map_py, rtol=1e-12, atol=1e-12)

    print(
        f"[{name}] OK  "
        f"max|dy|={np.max(np.abs(y_cpp - y_py)):.3e}, "
        f"max|dSTM|={np.max(np.abs(stm_cpp - stm_py)):.3e}, "
        f"max|dMap|={np.max(np.abs(map_cpp - map_py)):.3e}"
    )


def main() -> None:
    y0_cart = _reference_cart_state()
    eq0 = _to_geqoe_state(y0_cart)

    print("Comparing Python staged Order-1 vs C++ staged Order-1 across dt grids...")
    for case_name, dt in _dt_cases().items():
        _compare_case(case_name, dt, eq0)

    print("All Order-1 parity checks passed.")


if __name__ == "__main__":
    main()
