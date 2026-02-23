from __future__ import annotations

import numpy as np
import numpy.testing as npt

from astrodyn_core.geqoe_cpp import (
    evaluate_cart_taylor_cpp,
    evaluate_taylor_cpp,
    prepare_cart_coefficients_cpp,
    prepare_taylor_coefficients_cpp,
)
from astrodyn_core.propagation.geqoe import core as py_core
from astrodyn_core.propagation.geqoe.conversion import rv2geqoe

_J2 = 0.0010826266835531513
_Re = 6378137.0
_mu = 3.986004418e14


def _reference_cart_state() -> np.ndarray:
    return np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


def _to_eq_state(y0_cart: np.ndarray) -> np.ndarray:
    eq0_tuple = rv2geqoe(t=0.0, y=y0_cart, p=(_J2, _Re, _mu))
    return np.hstack([elem.flatten() for elem in eq0_tuple])


def test_staged_order1_geqoe_parity_vs_python() -> None:
    eq0 = _to_eq_state(_reference_cart_state())
    dt = np.array([0.0, 30.0, 120.0, 600.0], dtype=float)

    coeffs_cpp = prepare_taylor_coefficients_cpp(eq0, _J2, _Re, _mu, order=1)
    y_cpp, stm_cpp, map_cpp = evaluate_taylor_cpp(coeffs_cpp, dt)

    coeffs_py = py_core.prepare_taylor_coefficients(y0=eq0, p=(_J2, _Re, _mu), order=1)
    y_py, stm_py, map_py = py_core.evaluate_taylor(coeffs_py, dt)

    npt.assert_allclose(y_cpp, y_py, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(stm_cpp, stm_py, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(map_cpp, map_py, rtol=1e-12, atol=1e-12)


def test_staged_order1_cartesian_parity_vs_python() -> None:
    y0 = _reference_cart_state()
    tspan = np.array([0.0, 30.0, 120.0, 600.0], dtype=float)

    coeffs_cpp, peq_py_0_cpp = prepare_cart_coefficients_cpp(y0, _J2, _Re, _mu, order=1)
    y_cpp, stm_cpp = evaluate_cart_taylor_cpp(coeffs_cpp, peq_py_0_cpp, tspan)

    coeffs_py, peq_py_0_py = py_core.prepare_cart_coefficients(y0_cart=y0, p=(_J2, _Re, _mu), order=1)
    y_py, stm_py = py_core.evaluate_cart_taylor(coeffs_py, peq_py_0_py, tspan)

    npt.assert_allclose(y_cpp, y_py, rtol=1e-11, atol=1e-10)
    npt.assert_allclose(stm_cpp, stm_py, rtol=1e-11, atol=1e-10)
