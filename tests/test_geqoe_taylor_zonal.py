"""Tests for Phase 7a: Zonal harmonic perturbations in GEqOE Taylor propagator.

Validates:
1. Legendre polynomial recurrence
2. ZonalPerturbation with J2-only matches J2Perturbation
3. Numeric gradient vs finite differences
4. J2+J3+J4 propagation vs Cowell ground truth
5. Higher harmonics produce measurable effects
"""

import numpy as np
import pytest

import heyoka as hy

from astrodyn_core.geqoe_taylor.constants import MU, J2, J3, J4, RE, A_J2
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.perturbations.zonal import (
    ZonalPerturbation, _legendre_P, _legendre_P_and_deriv,
)
from astrodyn_core.geqoe_taylor.perturbations.composite import CompositePerturbation
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.integrator import build_state_integrator
from astrodyn_core.geqoe_taylor.rhs import _is_j2_only, _can_use_zonal_path

# Paper case (a): LEO circular i=45deg
R0_A = np.array([7178.1366, 0.0, 0.0])
V0_A = np.array([0.0, 5.269240572916780, 5.269240572916780])

# Reference final state after 12 days, J2 only (Appendix C)
RF_REF = np.array([-5398.929377366906, -390.257240638229, -4693.719111636971])
VF_REF = np.array([2.214482567493, -6.845637008953, -1.977748618717])

PERT_J2 = J2Perturbation()


def _propagate_cowell_zonal(r0, v0, t_final, j_coeffs, mu=MU, re=RE, tol=1e-15):
    """Cowell ground truth with arbitrary zonal harmonics."""
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    r2 = x * x + y * y + z * z
    r = hy.sqrt(r2)

    # Two-body
    r3 = r2 * r
    ax = -mu * x / r3
    ay = -mu * y / r3
    az = -mu * z / r3

    # Zonal perturbation: acceleration = -grad(U)
    pert = ZonalPerturbation(j_coeffs, mu, re)
    dUdx, dUdy, dUdz = pert.grad_U_expr(x, y, z, r, 0.0, {})

    ax = ax - dUdx
    ay = ay - dUdy
    az = az - dUdz

    sys = [
        (x, vx), (y, vy), (z, vz),
        (vx, ax), (vy, ay), (vz, az),
    ]

    ic = list(r0) + list(v0)
    ta = hy.taylor_adaptive(sys, ic, tol=tol)
    ta.propagate_until(t_final)

    return ta.state[:3].copy(), ta.state[3:].copy()


class TestLegendrePolynomials:
    """Verify Bonnet recurrence for Legendre polynomials."""

    def test_P0(self):
        assert _legendre_P(0, 0.5) == pytest.approx(1.0)

    def test_P1(self):
        assert _legendre_P(1, 0.7) == pytest.approx(0.7)

    def test_P2(self):
        x = 0.6
        expected = (3 * x**2 - 1) / 2
        assert _legendre_P(2, x) == pytest.approx(expected)

    def test_P3(self):
        x = 0.4
        expected = (5 * x**3 - 3 * x) / 2
        assert _legendre_P(3, x) == pytest.approx(expected)

    def test_P4(self):
        x = 0.3
        expected = (35 * x**4 - 30 * x**2 + 3) / 8
        assert _legendre_P(4, x) == pytest.approx(expected)


class TestLegendreDerivatives:
    """Verify differentiated Bonnet recurrence."""

    def test_dP0(self):
        _, dP = _legendre_P_and_deriv(0, 0.5)
        assert dP == pytest.approx(0.0)

    def test_dP1(self):
        _, dP = _legendre_P_and_deriv(1, 0.7)
        assert dP == pytest.approx(1.0)

    def test_dP2(self):
        x = 0.6
        P, dP = _legendre_P_and_deriv(2, x)
        assert P == pytest.approx((3 * x**2 - 1) / 2)
        assert dP == pytest.approx(3 * x)

    def test_dP3(self):
        x = 0.4
        P, dP = _legendre_P_and_deriv(3, x)
        assert P == pytest.approx((5 * x**3 - 3 * x) / 2)
        assert dP == pytest.approx((15 * x**2 - 3) / 2)

    def test_dP4(self):
        x = 0.3
        P, dP = _legendre_P_and_deriv(4, x)
        assert P == pytest.approx((35 * x**4 - 30 * x**2 + 3) / 8)
        # P'_4 = (140*x^3 - 60*x) / 8 = (35*x^3 - 15*x)/2
        assert dP == pytest.approx((140 * x**3 - 60 * x) / 8)


class TestZonalConstruction:
    """ZonalPerturbation construction and properties."""

    def test_uses_zonal_fast_path(self):
        zonal = ZonalPerturbation({2: J2})
        assert _can_use_zonal_path(zonal) is True
        assert _is_j2_only(zonal) is False

    def test_composite_falls_back_to_general(self):
        """Composite with zonal should NOT use zonal fast path (no delegation)."""
        zonal = ZonalPerturbation({2: J2})
        comp = CompositePerturbation(conservative=[zonal])
        assert _can_use_zonal_path(comp) is False
        assert _is_j2_only(comp) is False

    def test_empty_coeffs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ZonalPerturbation({})

    def test_degree_below_2_raises(self):
        with pytest.raises(ValueError, match="degree must be >= 2"):
            ZonalPerturbation({1: 0.5})


class TestZonalJ2Only:
    """ZonalPerturbation with only J2 should match J2Perturbation."""

    def test_potential_matches(self):
        zonal = ZonalPerturbation({2: J2})
        r_test = np.array([7000.0, 500.0, 3000.0])

        u_j2 = PERT_J2.U_numeric(r_test)
        u_zonal = zonal.U_numeric(r_test)
        assert u_zonal == pytest.approx(u_j2, rel=1e-12)

    def test_propagation_matches_12day(self):
        """12-day propagation should match paper reference to ~1e-5 km."""
        zonal = ZonalPerturbation({2: J2})
        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)

        ta, _ = build_state_integrator(zonal, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(12.0 * 86400.0)

        rf, vf = geqoe2cart(ta.state, MU, zonal)
        pos_err = np.linalg.norm(rf - RF_REF)
        vel_err = np.linalg.norm(vf - VF_REF)

        assert pos_err < 1e-5, f"Position error {pos_err:.2e} km"
        assert vel_err < 1e-8, f"Velocity error {vel_err:.2e} km/s"

    def test_nu_conserved(self):
        """nu should be conserved (E_dot=0 for conservative time-independent)."""
        zonal = ZonalPerturbation({2: J2})
        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)
        nu0 = ic[0]

        ta, _ = build_state_integrator(zonal, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(86400.0)

        assert ta.state[0] == pytest.approx(nu0, abs=1e-14)


class TestZonalGradient:
    """Verify symbolic gradient via finite differences of U_numeric."""

    def test_gradient_fd_j2(self):
        self._check_gradient_fd(ZonalPerturbation({2: J2}))

    def test_gradient_fd_j2_j3_j4(self):
        self._check_gradient_fd(ZonalPerturbation({2: J2, 3: J3, 4: J4}))

    def _check_gradient_fd(self, pert):
        r_test = np.array([7000.0, 500.0, 3000.0])
        eps = 1e-6

        grad_fd = np.zeros(3)
        for i in range(3):
            r_plus = r_test.copy()
            r_plus[i] += eps
            r_minus = r_test.copy()
            r_minus[i] -= eps
            grad_fd[i] = (pert.U_numeric(r_plus) - pert.U_numeric(r_minus)) / (2 * eps)

        # Evaluate symbolic gradient via cfunc
        x, y, z = hy.make_vars("x", "y", "z")
        r = hy.sqrt(x * x + y * y + z * z)
        dUdx, dUdy, dUdz = pert.grad_U_expr(x, y, z, r, 0.0, {})

        cf = hy.cfunc([dUdx, dUdy, dUdz], [x, y, z])
        grad_sym = cf(r_test)

        np.testing.assert_allclose(grad_sym, grad_fd, rtol=5e-5)


class TestZonalHigherOrder:
    """Tests with J2+J3+J4 zonal harmonics."""

    def test_higher_harmonics_measurable_effect(self):
        """J3+J4 should produce a measurable difference from J2-only."""
        t_final = 12.0 * 86400.0

        # J2-only (start from same Cartesian state)
        ic_j2 = cart2geqoe(R0_A, V0_A, MU, PERT_J2)
        ta_j2, _ = build_state_integrator(PERT_J2, ic_j2, tol=1e-15)
        ta_j2.propagate_until(t_final)
        rf_j2, _ = geqoe2cart(ta_j2.state, MU, PERT_J2)

        # J2+J3+J4 (start from same Cartesian state, different elements)
        zonal = ZonalPerturbation({2: J2, 3: J3, 4: J4})
        ic_z = cart2geqoe(R0_A, V0_A, MU, zonal)
        ta_z, _ = build_state_integrator(zonal, ic_z, tol=1e-15, compact_mode=True)
        ta_z.propagate_until(t_final)
        rf_z, _ = geqoe2cart(ta_z.state, MU, zonal)

        diff = np.linalg.norm(rf_z - rf_j2)
        assert diff > 0.01, f"J3+J4 effect too small: {diff:.2e} km"
        assert diff < 100.0, f"J3+J4 effect too large: {diff:.2e} km"

    def test_vs_cowell_ground_truth(self):
        """J2+J3+J4 GEqOE matches Cowell ground truth."""
        j_coeffs = {2: J2, 3: J3, 4: J4}
        zonal = ZonalPerturbation(j_coeffs)
        ic = cart2geqoe(R0_A, V0_A, MU, zonal)
        t_final = 12.0 * 86400.0

        # GEqOE propagation
        ta, _ = build_state_integrator(zonal, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(t_final)
        rf_geqoe, vf_geqoe = geqoe2cart(ta.state, MU, zonal)

        # Cowell ground truth
        rf_cow, vf_cow = _propagate_cowell_zonal(R0_A, V0_A, t_final, j_coeffs)

        pos_err = np.linalg.norm(rf_geqoe - rf_cow)
        vel_err = np.linalg.norm(vf_geqoe - vf_cow)

        assert pos_err < 1e-4, f"Position error vs Cowell: {pos_err:.2e} km"
        assert vel_err < 1e-7, f"Velocity error vs Cowell: {vel_err:.2e} km/s"

    def test_nu_conserved_higher_order(self):
        """nu conserved for J2+J3+J4 (conservative, time-independent)."""
        zonal = ZonalPerturbation({2: J2, 3: J3, 4: J4})
        ic = cart2geqoe(R0_A, V0_A, MU, zonal)
        nu0 = ic[0]

        ta, _ = build_state_integrator(zonal, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(86400.0)

        assert ta.state[0] == pytest.approx(nu0, abs=1e-14)
