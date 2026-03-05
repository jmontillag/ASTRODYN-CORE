"""Tests for GEqOE Taylor truncation error estimation utilities."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from astrodyn_core.propagation.geqoe.conversion import rv2geqoe
from astrodyn_core.propagation.geqoe.error import compute_max_dt, estimate_position_error
from astrodyn_core.propagation.geqoe.jacobians import get_pYpEq
from astrodyn_core.propagation.geqoe.core import (
    prepare_cart_coefficients,
    evaluate_cart_taylor,
)

_J2 = 0.0010826266835531513
_Re = 6378137.0
_mu = 3.986004418e14
_time_scale = (_Re**3 / _mu) ** 0.5


def _reference_state() -> np.ndarray:
    return np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


def _make_map_and_jac(order: int = 4):
    """Prepare map_components and pYpEq for the reference state."""
    y0 = _reference_state()
    p = (_J2, _Re, _mu)
    coeffs, _ = prepare_cart_coefficients(y0, p, order=order)
    eq0_tuple = rv2geqoe(t=0.0, y=y0, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])
    pYpEq = get_pYpEq(t=0.0, y=eq0, p=p)
    if pYpEq.ndim == 3:
        pYpEq = pYpEq[0]
    return coeffs.map_components, pYpEq


class TestEstimatePositionError:
    def test_zero_dt_gives_zero_error(self):
        mc, jac = _make_map_and_jac()
        err = estimate_position_error(mc, jac, 0.0, _time_scale, 4)
        assert err == 0.0

    def test_error_increases_with_dt(self):
        mc, jac = _make_map_and_jac()
        e1 = estimate_position_error(mc, jac, 60.0, _time_scale, 4)
        e2 = estimate_position_error(mc, jac, 300.0, _time_scale, 4)
        assert e2 > e1 > 0.0

    def test_higher_order_lower_error_at_moderate_dt(self):
        """Higher order should give lower estimated error at moderate dt."""
        y0 = _reference_state()
        p = (_J2, _Re, _mu)
        errors = {}
        for order in (2, 3, 4):
            coeffs, _ = prepare_cart_coefficients(y0, p, order=order)
            eq0_tuple = rv2geqoe(t=0.0, y=y0, p=p)
            eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])
            pYpEq = get_pYpEq(t=0.0, y=eq0, p=p)
            if pYpEq.ndim == 3:
                pYpEq = pYpEq[0]
            errors[order] = estimate_position_error(
                coeffs.map_components, pYpEq, 120.0, _time_scale, order
            )
        assert errors[4] < errors[3] < errors[2]


class TestComputeMaxDt:
    def test_positive_dt(self):
        mc, jac = _make_map_and_jac()
        dt = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=1.0)
        assert dt > 0.0

    def test_tighter_tolerance_shorter_step(self):
        mc, jac = _make_map_and_jac()
        dt_tight = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=0.1)
        dt_loose = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=10.0)
        assert dt_loose > dt_tight

    def test_safety_factor(self):
        mc, jac = _make_map_and_jac()
        dt_safe = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=1.0, safety_factor=0.5)
        dt_full = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=1.0, safety_factor=1.0)
        # Direct formula: dt_safe / dt_full is exactly the safety factor.
        npt.assert_allclose(dt_safe, dt_full * 0.5, rtol=1e-12)

    def test_error_at_dt_max_below_tolerance(self):
        """Error at dt_max should not exceed pos_tol."""
        mc, jac = _make_map_and_jac()
        pos_tol = 1.0
        sf = 0.8
        dt = compute_max_dt(mc, jac, _time_scale, 4, pos_tol=pos_tol, safety_factor=sf)
        err = estimate_position_error(mc, jac, dt, _time_scale, 4)
        assert err < pos_tol, f"error {err} exceeds pos_tol {pos_tol}"
        # Direct formula: error at dt_max is exactly pos_tol * sf^order.
        expected = pos_tol * sf**4
        npt.assert_allclose(err, expected, rtol=1e-12)

    def test_higher_order_longer_step(self):
        """Higher order should yield a longer step for the same tolerance."""
        y0 = _reference_state()
        p = (_J2, _Re, _mu)
        steps = {}
        for order in (2, 3, 4):
            coeffs, _ = prepare_cart_coefficients(y0, p, order=order)
            eq0_tuple = rv2geqoe(t=0.0, y=y0, p=p)
            eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])
            pYpEq = get_pYpEq(t=0.0, y=eq0, p=p)
            if pYpEq.ndim == 3:
                pYpEq = pYpEq[0]
            steps[order] = compute_max_dt(
                coeffs.map_components, pYpEq, _time_scale, order, pos_tol=1.0
            )
        assert steps[4] > steps[3] > steps[2]
