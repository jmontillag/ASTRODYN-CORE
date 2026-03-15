"""Tests for the GEqOE Taylor propagator (heyoka-based).

Validates conversions, propagation, and STM against the reference paper:
Baù, Hernando-Ayuso & Bombardelli (2021), Celest. Mech. Dyn. Astr. 133:50.
"""

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor.constants import MU, J2, RE, A_J2
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.integrator import (
    build_state_integrator,
    build_stm_integrator,
    propagate,
    propagate_grid,
    extract_stm,
)
from astrodyn_core.geqoe_taylor.utils import K_to_L, solve_kepler_gen
from astrodyn_core.geqoe_taylor.cowell import propagate_cowell_heyoka

# Paper case (a): LEO circular i=45°
R0_A = np.array([7178.1366, 0.0, 0.0])
V0_A = np.array([0.0, 5.269240572916780, 5.269240572916780])

# Reference final state after 12 days, J2 only (Appendix C)
RF_REF = np.array([-5398.929377366906, -390.257240638229, -4693.719111636971])
VF_REF = np.array([2.214482567493, -6.845637008953, -1.977748618717])

PERT = J2Perturbation()


class TestConversions:
    """Conversion round-trip and paper value checks."""

    def test_cart2geqoe_case_a(self):
        """Verify GEqOE values match Table 3 of the paper."""
        geqoe = cart2geqoe(R0_A, V0_A, MU, PERT)
        nu, p1, p2, K, q1, q2 = geqoe

        assert nu == pytest.approx(1.03946e-3, rel=1e-4)
        assert p1 == pytest.approx(0.0, abs=1e-12)
        assert p2 == pytest.approx(-8.5476e-4, rel=1e-3)
        assert K == pytest.approx(0.0, abs=1e-12)
        assert q1 == pytest.approx(0.0, abs=1e-12)
        assert q2 == pytest.approx(0.41421356, rel=1e-6)

    def test_roundtrip_case_a(self):
        """Cart -> GEqOE -> Cart round-trip at machine precision."""
        geqoe = cart2geqoe(R0_A, V0_A, MU, PERT)
        r2, v2 = geqoe2cart(geqoe, MU, PERT)

        assert np.linalg.norm(r2 - R0_A) < 1e-10  # km
        assert np.linalg.norm(v2 - V0_A) < 1e-13  # km/s

    def test_roundtrip_elliptical(self):
        """Round-trip for an eccentric orbit (e~0.3, i=60°)."""
        a = 10000.0  # km
        e = 0.3
        i = np.radians(60.0)

        # Build Cartesian state at periapsis
        rp = a * (1 - e)
        vp = np.sqrt(MU * (2.0 / rp - 1.0 / a))
        r_vec = np.array([rp, 0.0, 0.0])
        v_vec = np.array([0.0, vp * np.cos(i), vp * np.sin(i)])

        geqoe = cart2geqoe(r_vec, v_vec, MU, PERT)
        r2, v2 = geqoe2cart(geqoe, MU, PERT)

        assert np.linalg.norm(r2 - r_vec) < 1e-9
        assert np.linalg.norm(v2 - v_vec) < 1e-12

    def test_roundtrip_retrograde(self):
        """Round-trip for a retrograde orbit (i=135°)."""
        r_vec = np.array([8000.0, 0.0, 0.0])
        v_circ = np.sqrt(MU / 8000.0)
        i = np.radians(135.0)
        v_vec = np.array([0.0, v_circ * np.cos(i), v_circ * np.sin(i)])

        geqoe = cart2geqoe(r_vec, v_vec, MU, PERT)
        r2, v2 = geqoe2cart(geqoe, MU, PERT)

        assert np.linalg.norm(r2 - r_vec) < 1e-9
        assert np.linalg.norm(v2 - v_vec) < 1e-12


class TestKeplerEquation:
    """Generalized Kepler equation utilities."""

    def test_K_to_L_roundtrip(self):
        """K -> L -> K round-trip."""
        K0 = 1.5
        p1, p2 = 0.1, -0.2
        L = K_to_L(K0, p1, p2)
        K_back = solve_kepler_gen(L, p1, p2)
        assert K_back == pytest.approx(K0, abs=1e-14)

    def test_solve_kepler_gen_zero(self):
        """K=0 when L=0 and p1=0."""
        K = solve_kepler_gen(0.0, 0.0, -8.5e-4)
        assert abs(K) < 1e-14


class TestPropagation:
    """Propagation accuracy against paper reference values."""

    def test_12day_case_a(self):
        """12-day propagation matches Appendix C to < 1e-6 km."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)
        ta, _ = build_state_integrator(PERT, ic, tol=1e-15)
        ta.propagate_until(12.0 * 86400.0)

        rf, vf = geqoe2cart(ta.state, MU, PERT)
        assert np.linalg.norm(rf - RF_REF) < 1e-6  # km
        assert np.linalg.norm(vf - VF_REF) < 1e-9  # km/s

    def test_nu_conserved(self):
        """nu (generalized mean motion) is exactly conserved for J2."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)
        nu0 = ic[0]

        ta, _ = build_state_integrator(PERT, ic, tol=1e-15)
        ta.propagate_until(86400.0)  # 1 day

        assert ta.state[0] == pytest.approx(nu0, abs=1e-20)

    def test_step_propagation(self):
        """Step-by-step propagation gives same result as propagate_until."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)

        # propagate_until
        ta1, _ = build_state_integrator(PERT, ic, tol=1e-15)
        ta1.propagate_until(3600.0)

        # Step-by-step
        ta2, _ = build_state_integrator(PERT, ic, tol=1e-15)
        times, states = propagate(ta2, 3600.0)

        np.testing.assert_allclose(ta1.state, states[-1], atol=1e-14)

    def test_propagate_grid(self):
        """propagate_grid returns states at requested times."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)
        ta, _ = build_state_integrator(PERT, ic, tol=1e-15)

        t_grid = np.linspace(0.0, 3600.0, 10)
        states = propagate_grid(ta, t_grid)

        assert states.shape == (10, 6)
        # First row should be IC
        np.testing.assert_allclose(states[0], ic, atol=1e-12)


class TestSTM:
    """STM validation against finite differences."""

    def test_stm_vs_finite_diff(self):
        """STM matches central finite differences to < 1e-5."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)

        # Variational integrator
        ta_v, _ = build_stm_integrator(PERT, ic, tol=1e-15)
        T_orb = 2 * np.pi / ic[0]
        ta_v.propagate_until(T_orb)
        _, phi = extract_stm(ta_v.state)

        # Finite differences
        delta = 1e-7
        phi_fd = np.zeros((6, 6))
        for j in range(6):
            scale = max(abs(ic[j]), 1e-6)
            dj = delta * scale
            ic_p = ic.copy()
            ic_m = ic.copy()
            ic_p[j] += dj
            ic_m[j] -= dj

            ta_p, _ = build_state_integrator(PERT, ic_p, tol=1e-15)
            ta_m, _ = build_state_integrator(PERT, ic_m, tol=1e-15)
            ta_p.propagate_until(T_orb)
            ta_m.propagate_until(T_orb)
            phi_fd[:, j] = (ta_p.state - ta_m.state) / (2 * dj)

        rel_err = np.linalg.norm(phi - phi_fd) / np.linalg.norm(phi)
        assert rel_err < 1e-5

    def test_stm_identity_at_t0(self):
        """STM is identity at t=0."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT)
        ta_v, _ = build_stm_integrator(PERT, ic, tol=1e-15)
        _, phi = extract_stm(ta_v.state)
        np.testing.assert_allclose(phi, np.eye(6), atol=1e-15)


class TestCowellGroundTruth:
    """Validate GEqOE against independent Cowell (Cartesian) integration."""

    def test_12day_vs_cowell_heyoka(self):
        """GEqOE Taylor matches heyoka Cowell ground truth to < 1e-6 km."""
        r_cow, v_cow = propagate_cowell_heyoka(R0_A, V0_A, 12.0 * 86400.0)

        ic = cart2geqoe(R0_A, V0_A, MU, PERT)
        ta, _ = build_state_integrator(PERT, ic, tol=1e-15)
        ta.propagate_until(12.0 * 86400.0)
        rf, vf = geqoe2cart(ta.state, MU, PERT)

        assert np.linalg.norm(rf - r_cow) < 1e-6  # km
        assert np.linalg.norm(vf - v_cow) < 1e-9  # km/s

    def test_cowell_vs_paper(self):
        """Cowell ground truth is consistent with paper Dromo reference."""
        r_cow, v_cow = propagate_cowell_heyoka(R0_A, V0_A, 12.0 * 86400.0)

        assert np.linalg.norm(r_cow - RF_REF) < 1e-5  # km
        assert np.linalg.norm(v_cow - VF_REF) < 1e-8  # km/s
