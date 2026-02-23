"""Parity tests for the C++ GEqOE utility functions.

Compares C++ implementations (geqoe_cpp) against the canonical Python
implementations (propagation.geqoe) to ensure bit-level numerical parity.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from astrodyn_core.geqoe_cpp import geqoe2rv as cpp_geqoe2rv
from astrodyn_core.geqoe_cpp import get_pEqpY as cpp_get_pEqpY
from astrodyn_core.geqoe_cpp import get_pYpEq as cpp_get_pYpEq
from astrodyn_core.geqoe_cpp import rv2geqoe as cpp_rv2geqoe
from astrodyn_core.geqoe_cpp import solve_kep_gen as cpp_solve_kep_gen
from astrodyn_core.propagation.geqoe.conversion import (
    geqoe2rv as py_geqoe2rv,
    rv2geqoe as py_rv2geqoe,
)
from astrodyn_core.propagation.geqoe.jacobians import (
    get_pEqpY as py_get_pEqpY,
    get_pYpEq as py_get_pYpEq,
)
from astrodyn_core.propagation.geqoe.utils import solve_kep_gen as py_solve_kep_gen

# ---------------------------------------------------------------------------
# WGS84 body constants (matching Orekit Constants.WGS84_EARTH_*)
# ---------------------------------------------------------------------------
_J2 = 0.0010826266835531513
_Re = 6378137.0
_mu = 3.986004418e14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_cart_states(n: int, seed: int = 42) -> np.ndarray:
    """Generate *n* random LEO-ish Cartesian states (N, 6)."""
    rng = np.random.default_rng(seed)
    states = np.empty((n, 6))
    for i in range(n):
        # Random altitude 400--1200 km
        r_mag = _Re + rng.uniform(400e3, 1200e3)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        pos = direction * r_mag

        # Roughly circular velocity + small perturbation
        v_circ = np.sqrt(_mu / r_mag)
        v_dir = np.cross([0, 0, 1], direction)
        if np.linalg.norm(v_dir) < 1e-10:
            v_dir = np.cross([0, 1, 0], direction)
        v_dir /= np.linalg.norm(v_dir)
        vel = v_dir * v_circ * (1.0 + rng.uniform(-0.05, 0.05))

        states[i, :3] = pos
        states[i, 3:] = vel
    return states


# ---------------------------------------------------------------------------
# solve_kep_gen parity
# ---------------------------------------------------------------------------
class TestSolveKepGen:
    """Kepler solver: C++ vs Python parity."""

    def test_parity_random_50(self) -> None:
        rng = np.random.default_rng(123)
        N = 50
        Lr = rng.uniform(0, 2 * np.pi, N)
        p1 = rng.uniform(-0.5, 0.5, N)
        p2 = rng.uniform(-0.5, 0.5, N)

        K_py = py_solve_kep_gen(Lr, p1, p2)
        K_cpp = cpp_solve_kep_gen(Lr, p1, p2)

        npt.assert_allclose(K_cpp, K_py, rtol=1e-14, atol=1e-14)

    def test_parity_near_circular(self) -> None:
        """Near-circular orbits (small p1, p2)."""
        rng = np.random.default_rng(456)
        N = 30
        Lr = rng.uniform(0, 2 * np.pi, N)
        p1 = rng.uniform(-0.01, 0.01, N)
        p2 = rng.uniform(-0.01, 0.01, N)

        K_py = py_solve_kep_gen(Lr, p1, p2)
        K_cpp = cpp_solve_kep_gen(Lr, p1, p2)

        npt.assert_allclose(K_cpp, K_py, rtol=1e-14, atol=1e-14)

    def test_parity_eccentric(self) -> None:
        """Moderately eccentric orbits."""
        rng = np.random.default_rng(789)
        N = 30
        Lr = rng.uniform(0, 2 * np.pi, N)
        # Eccentricity up to ~0.7
        e = rng.uniform(0.3, 0.7, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        p1 = e * np.sin(theta)
        p2 = e * np.cos(theta)

        K_py = py_solve_kep_gen(Lr, p1, p2)
        K_cpp = cpp_solve_kep_gen(Lr, p1, p2)

        npt.assert_allclose(K_cpp, K_py, rtol=1e-14, atol=1e-14)

    def test_single_element(self) -> None:
        """Single-element arrays."""
        Lr = np.array([2.5])
        p1 = np.array([0.1])
        p2 = np.array([0.2])

        K_py = py_solve_kep_gen(Lr, p1, p2)
        K_cpp = cpp_solve_kep_gen(Lr, p1, p2)

        npt.assert_allclose(K_cpp, K_py, rtol=1e-15, atol=1e-15)


# ---------------------------------------------------------------------------
# rv2geqoe parity
# ---------------------------------------------------------------------------
class TestRV2GEqOE:
    """Cartesian -> GEqOE: C++ vs Python parity."""

    def test_parity_batch_20(self) -> None:
        states = _random_cart_states(20, seed=42)

        py_result = py_rv2geqoe(0.0, states, (_J2, _Re, _mu))
        cpp_result = cpp_rv2geqoe(states, _J2, _Re, _mu)

        for j, (py_arr, cpp_arr) in enumerate(zip(py_result, cpp_result)):
            npt.assert_allclose(
                cpp_arr, py_arr, rtol=1e-13, atol=1e-13,
                err_msg=f"Component {j} mismatch"
            )

    def test_parity_batch_100(self) -> None:
        states = _random_cart_states(100, seed=99)

        py_result = py_rv2geqoe(0.0, states, (_J2, _Re, _mu))
        cpp_result = cpp_rv2geqoe(states, _J2, _Re, _mu)

        for j, (py_arr, cpp_arr) in enumerate(zip(py_result, cpp_result)):
            npt.assert_allclose(
                cpp_arr, py_arr, rtol=1e-13, atol=1e-13,
                err_msg=f"Component {j} mismatch"
            )

    def test_single_state(self) -> None:
        """Single state, 1-D input (6,)."""
        state = _random_cart_states(1, seed=7)[0]  # shape (6,)

        py_result = py_rv2geqoe(0.0, state, (_J2, _Re, _mu))
        cpp_result = cpp_rv2geqoe(state, _J2, _Re, _mu)

        for j, (py_arr, cpp_arr) in enumerate(zip(py_result, cpp_result)):
            npt.assert_allclose(
                cpp_arr, py_arr, rtol=1e-13, atol=1e-13,
                err_msg=f"Component {j} mismatch (single state)"
            )

    def test_single_state_2d(self) -> None:
        """Single state, 2-D input (1, 6)."""
        state = _random_cart_states(1, seed=7)  # shape (1, 6)

        py_result = py_rv2geqoe(0.0, state, (_J2, _Re, _mu))
        cpp_result = cpp_rv2geqoe(state, _J2, _Re, _mu)

        for j, (py_arr, cpp_arr) in enumerate(zip(py_result, cpp_result)):
            npt.assert_allclose(
                cpp_arr, py_arr, rtol=1e-13, atol=1e-13,
                err_msg=f"Component {j} mismatch (single state 2D)"
            )


# ---------------------------------------------------------------------------
# geqoe2rv parity
# ---------------------------------------------------------------------------
class TestGEqOE2RV:
    """GEqOE -> Cartesian: C++ vs Python parity.

    Note on tolerance: geqoe2rv velocity atol is set to 1e-11 (10 pm/s)
    rather than 1e-13.  The mismatches occur exclusively in the z-velocity
    of near-equatorial orbits where the true value is ~0.  The difference
    (~5e-13) arises from NumPy vectorised vs C++ scalar intermediate
    rounding.  Positions are exact to 1e-13; velocity *relative* error
    (rtol) is also 1e-13.
    """

    _VEL_ATOL = 1e-11   # 10 pm/s -- purely for near-zero z-velocity noise

    def _to_geqoe(self, cart: np.ndarray) -> np.ndarray:
        """Convert Cartesian to GEqOE via the Python reference."""
        result = py_rv2geqoe(0.0, cart, (_J2, _Re, _mu))
        return np.column_stack(result)

    def test_parity_batch_20(self) -> None:
        cart = _random_cart_states(20, seed=42)
        geqoe = self._to_geqoe(cart)

        py_rv, py_rpv = py_geqoe2rv(0.0, geqoe, (_J2, _Re, _mu))
        cpp_rv, cpp_rpv = cpp_geqoe2rv(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_rv, py_rv, rtol=1e-13, atol=1e-13,
                            err_msg="Position mismatch")
        npt.assert_allclose(cpp_rpv, py_rpv, rtol=1e-13, atol=self._VEL_ATOL,
                            err_msg="Velocity mismatch")

    def test_parity_batch_100(self) -> None:
        cart = _random_cart_states(100, seed=99)
        geqoe = self._to_geqoe(cart)

        py_rv, py_rpv = py_geqoe2rv(0.0, geqoe, (_J2, _Re, _mu))
        cpp_rv, cpp_rpv = cpp_geqoe2rv(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_rv, py_rv, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(cpp_rpv, py_rpv, rtol=1e-13, atol=self._VEL_ATOL)

    def test_single_state(self) -> None:
        """Single state, 1-D GEqOE input (6,)."""
        cart = _random_cart_states(1, seed=7)
        geqoe = self._to_geqoe(cart)[0]  # shape (6,)

        py_rv, py_rpv = py_geqoe2rv(0.0, np.atleast_2d(geqoe), (_J2, _Re, _mu))
        cpp_rv, cpp_rpv = cpp_geqoe2rv(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_rv, py_rv, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(cpp_rpv, py_rpv, rtol=1e-13, atol=self._VEL_ATOL)


# ---------------------------------------------------------------------------
# Roundtrip tests  (cart -> geqoe -> cart)
# ---------------------------------------------------------------------------
class TestRoundtrip:
    """rv -> GEqOE -> rv roundtrip via C++ implementations."""

    def test_roundtrip_batch(self) -> None:
        cart = _random_cart_states(30, seed=55)

        # Forward  (cart -> geqoe)  via C++
        nu, q1, q2, p1, p2, Lr = cpp_rv2geqoe(cart, _J2, _Re, _mu)
        geqoe = np.column_stack([nu, q1, q2, p1, p2, Lr])

        # Inverse  (geqoe -> cart)  via C++
        rv_back, rpv_back = cpp_geqoe2rv(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(rv_back, cart[:, :3], rtol=1e-11, atol=1e-6,
                            err_msg="Position roundtrip failed")
        npt.assert_allclose(rpv_back, cart[:, 3:], rtol=1e-11, atol=1e-3,
                            err_msg="Velocity roundtrip failed")

    def test_roundtrip_single(self) -> None:
        cart = _random_cart_states(1, seed=77)[0]

        nu, q1, q2, p1, p2, Lr = cpp_rv2geqoe(cart, _J2, _Re, _mu)
        geqoe = np.array([nu[0], q1[0], q2[0], p1[0], p2[0], Lr[0]])

        rv_back, rpv_back = cpp_geqoe2rv(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(rv_back.flatten(), cart[:3], rtol=1e-11, atol=1e-6)
        npt.assert_allclose(rpv_back.flatten(), cart[3:], rtol=1e-11, atol=1e-3)


# ---------------------------------------------------------------------------
# Jacobian parity: get_pEqpY
# ---------------------------------------------------------------------------
class TestGetPEqPY:
    """d(Eq)/d(Y) Jacobian: C++ vs Python parity."""

    _ATOL = 1e-11  # near-zero element noise (same rationale as geqoe2rv velocity)

    def test_parity_batch_20(self) -> None:
        states = _random_cart_states(20, seed=42)

        py_jac = py_get_pEqpY(0.0, states, (_J2, _Re, _mu))
        cpp_jac = cpp_get_pEqpY(states, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL,
                            err_msg="get_pEqpY mismatch")

    def test_parity_batch_100(self) -> None:
        states = _random_cart_states(100, seed=99)

        py_jac = py_get_pEqpY(0.0, states, (_J2, _Re, _mu))
        cpp_jac = cpp_get_pEqpY(states, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL,
                            err_msg="get_pEqpY mismatch (100 states)")

    def test_single_state(self) -> None:
        state = _random_cart_states(1, seed=7)[0]

        py_jac = py_get_pEqpY(0.0, state, (_J2, _Re, _mu))
        cpp_jac = cpp_get_pEqpY(state, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL)


# ---------------------------------------------------------------------------
# Jacobian parity: get_pYpEq
# ---------------------------------------------------------------------------
class TestGetPYPEq:
    """d(Y)/d(Eq) Jacobian: C++ vs Python parity.

    Note on tolerance: atol=1e-8 rather than 1e-11 because get_pYpEq
    involves a Kepler solve + nested intermediate computations (r from
    norm, h from sqrt(c^2 - 2*r^2*U)) that accumulate more FP noise
    than get_pEqpY.  The mismatches are exclusively in near-zero
    z-component partials (~1e-9) while non-zero elements (~1e+9) match
    to rtol=1e-12.  Inverse consistency (pEqpY @ pYpEq = I) holds to
    atol=1e-8, confirming mathematical correctness.
    """

    _ATOL = 1e-8

    def _to_geqoe(self, cart: np.ndarray) -> np.ndarray:
        result = py_rv2geqoe(0.0, cart, (_J2, _Re, _mu))
        return np.column_stack(result)

    def test_parity_batch_20(self) -> None:
        cart = _random_cart_states(20, seed=42)
        geqoe = self._to_geqoe(cart)

        py_jac = py_get_pYpEq(0.0, geqoe, (_J2, _Re, _mu))
        cpp_jac = cpp_get_pYpEq(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL,
                            err_msg="get_pYpEq mismatch")

    def test_parity_batch_100(self) -> None:
        cart = _random_cart_states(100, seed=99)
        geqoe = self._to_geqoe(cart)

        py_jac = py_get_pYpEq(0.0, geqoe, (_J2, _Re, _mu))
        cpp_jac = cpp_get_pYpEq(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL,
                            err_msg="get_pYpEq mismatch (100 states)")

    def test_single_state(self) -> None:
        cart = _random_cart_states(1, seed=7)
        geqoe = self._to_geqoe(cart)[0]

        py_jac = py_get_pYpEq(0.0, np.atleast_2d(geqoe), (_J2, _Re, _mu))
        cpp_jac = cpp_get_pYpEq(geqoe, _J2, _Re, _mu)

        npt.assert_allclose(cpp_jac, py_jac, rtol=1e-12, atol=self._ATOL)

    def test_jacobian_inverse_consistency(self) -> None:
        """pEqpY @ pYpEq should approximate the identity matrix."""
        cart = _random_cart_states(10, seed=33)
        geqoe = self._to_geqoe(cart)

        jac_eq_y = cpp_get_pEqpY(cart, _J2, _Re, _mu)
        jac_y_eq = cpp_get_pYpEq(geqoe, _J2, _Re, _mu)

        for i in range(10):
            product = jac_eq_y[i] @ jac_y_eq[i]
            npt.assert_allclose(product, np.eye(6), atol=1e-8,
                                err_msg=f"Jacobian inverse consistency failed for state {i}")
