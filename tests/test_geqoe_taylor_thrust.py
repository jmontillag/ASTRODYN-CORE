"""Tests for Phase 8a: continuous thrust in the GEqOE Taylor propagator."""

from __future__ import annotations

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor.constants import G0_MPS2, MU
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.cowell import propagate_cowell_heyoka_general
from astrodyn_core.geqoe_taylor.integrator import (
    build_state_integrator,
    build_thrust_sensitivity_integrator,
    build_thrust_state_integrator,
    build_thrust_stm_integrator,
    extract_stm,
    extract_variational_matrices,
)
from astrodyn_core.geqoe_taylor.perturbations.composite import CompositePerturbation
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.perturbations.thrust import ContinuousThrustPerturbation
from astrodyn_core.geqoe_taylor.thrust import ConstantRTNThrustLaw

# Paper case (a): LEO circular i=45deg
R0_A = np.array([7178.1366, 0.0, 0.0])
V0_A = np.array([0.0, 5.269240572916780, 5.269240572916780])


def _build_mass_state(perturbation, mass_kg: float) -> np.ndarray:
    geqoe = cart2geqoe(R0_A, V0_A, MU, perturbation)
    return np.concatenate([geqoe, [mass_kg]])


def _specific_energy(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    r = np.linalg.norm(r_vec)
    return np.dot(v_vec, v_vec) / 2.0 - MU / r


def _semi_major_axis(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    eps = _specific_energy(r_vec, v_vec)
    return -MU / (2.0 * eps)


class TestContinuousThrustCore:
    def test_requires_mass_augmented_builder(self):
        law = ConstantRTNThrustLaw(thrust_t_newtons=0.2, isp_s=2000.0)
        pert = CompositePerturbation(
            non_conservative=[ContinuousThrustPerturbation(law)]
        )
        ic = cart2geqoe(R0_A, V0_A, MU, pert)

        with pytest.raises(ValueError, match="mass-augmented GEqOE system"):
            build_state_integrator(pert, ic, tol=1e-15)

    def test_runtime_parameters_exposed(self):
        law = ConstantRTNThrustLaw(
            thrust_r_newtons=0.1,
            thrust_t_newtons=0.2,
            thrust_n_newtons=0.3,
            isp_s=1800.0,
        )
        pert = CompositePerturbation(
            conservative=[J2Perturbation()],
            non_conservative=[ContinuousThrustPerturbation(law)],
        )
        ic = _build_mass_state(pert, 450.0)

        ta, par_map = build_thrust_state_integrator(pert, ic, tol=1e-15)

        assert "mu" in par_map
        assert "thrust.r_newtons" in par_map
        assert "thrust.t_newtons" in par_map
        assert "thrust.n_newtons" in par_map
        assert "thrust.isp_s" in par_map
        assert ta.state[6] == pytest.approx(450.0)

    def test_mass_depletion_matches_closed_form(self):
        thrust_newtons = 0.4
        isp_s = 2200.0
        mass_kg = 500.0
        t_final = 7200.0

        law = ConstantRTNThrustLaw(thrust_t_newtons=thrust_newtons, isp_s=isp_s)
        pert = CompositePerturbation(
            non_conservative=[ContinuousThrustPerturbation(law)]
        )
        ic = _build_mass_state(pert, mass_kg)

        ta, _ = build_thrust_state_integrator(pert, ic, tol=1e-15)
        ta.propagate_until(t_final)

        expected_mass = mass_kg - thrust_newtons * t_final / (G0_MPS2 * isp_s)
        assert ta.state[6] == pytest.approx(expected_mass, rel=0.0, abs=1e-10)

    def test_constant_tangential_thrust_increases_orbital_energy(self):
        mass_kg = 500.0
        t_final = 5.0 * 3600.0

        thrust_law = ConstantRTNThrustLaw(
            thrust_t_newtons=0.5,
            isp_s=2200.0,
        )
        thrust_pert = CompositePerturbation(
            non_conservative=[ContinuousThrustPerturbation(thrust_law)]
        )
        zero_pert = CompositePerturbation()

        ic_thrust = _build_mass_state(thrust_pert, mass_kg)
        ic_zero = cart2geqoe(R0_A, V0_A, MU, zero_pert)

        ta_thrust, _ = build_thrust_state_integrator(
            thrust_pert, ic_thrust, tol=1e-15
        )
        ta_zero, _ = build_state_integrator(zero_pert, ic_zero, tol=1e-15)

        ta_thrust.propagate_until(t_final)
        ta_zero.propagate_until(t_final)

        r_thrust, v_thrust = geqoe2cart(ta_thrust.state, MU, thrust_pert)
        r_zero, v_zero = geqoe2cart(ta_zero.state, MU, zero_pert)

        a_initial = _semi_major_axis(R0_A, V0_A)
        a_zero = _semi_major_axis(r_zero, v_zero)
        a_thrust = _semi_major_axis(r_thrust, v_thrust)

        energy_initial = _specific_energy(R0_A, V0_A)
        energy_thrust = _specific_energy(r_thrust, v_thrust)

        assert abs(a_zero - a_initial) < 1e-7
        assert a_thrust > a_initial + 5.0
        assert energy_thrust > energy_initial

    def test_thrust_stm_identity_at_t0(self):
        law = ConstantRTNThrustLaw(thrust_t_newtons=0.2, isp_s=2000.0)
        pert = CompositePerturbation(
            non_conservative=[ContinuousThrustPerturbation(law)]
        )
        ic = _build_mass_state(pert, 500.0)

        ta, _ = build_thrust_stm_integrator(pert, ic, tol=1e-15)
        y, phi = extract_stm(ta.state, state_dim=7)

        np.testing.assert_allclose(y, ic, atol=1e-14)
        np.testing.assert_allclose(phi, np.eye(7), atol=1e-14)

    def test_thrust_param_sensitivity_identity_at_t0(self):
        law = ConstantRTNThrustLaw(
            thrust_r_newtons=0.1,
            thrust_t_newtons=0.2,
            thrust_n_newtons=0.05,
            isp_s=2100.0,
        )
        pert = CompositePerturbation(
            non_conservative=[ContinuousThrustPerturbation(law)]
        )
        ic = _build_mass_state(pert, 500.0)

        ta, par_map = build_thrust_sensitivity_integrator(pert, ic, tol=1e-15)
        y, phi_x, phi_p, param_names = extract_variational_matrices(
            ta.state, state_dim=7, par_map=par_map
        )

        np.testing.assert_allclose(y, ic, atol=1e-14)
        np.testing.assert_allclose(phi_x, np.eye(7), atol=1e-14)
        np.testing.assert_allclose(phi_p, np.zeros((7, len(param_names))), atol=1e-14)
        assert param_names == [
            "mu",
            "thrust.r_newtons",
            "thrust.t_newtons",
            "thrust.n_newtons",
            "thrust.isp_s",
        ]

    def test_thrust_param_sensitivities_vs_finite_difference(self):
        base_kwargs = dict(
            thrust_r_newtons=0.03,
            thrust_t_newtons=0.28,
            thrust_n_newtons=0.02,
            isp_s=2000.0,
        )
        mass_kg = 500.0
        t_final = 2400.0

        def build_pert(**kwargs):
            return CompositePerturbation(
                conservative=[J2Perturbation()],
                non_conservative=[ContinuousThrustPerturbation(ConstantRTNThrustLaw(**kwargs))],
            )

        pert = build_pert(**base_kwargs)
        ic = _build_mass_state(pert, mass_kg)
        ta, par_map = build_thrust_sensitivity_integrator(
            pert, ic, tol=1e-15, compact_mode=True
        )
        ta.propagate_until(t_final)
        y, _, phi_p, param_names = extract_variational_matrices(
            ta.state, state_dim=7, par_map=par_map
        )
        col_t = param_names.index("thrust.t_newtons")
        col_isp = param_names.index("thrust.isp_s")

        # Tangential thrust has the strongest orbital effect and is a good
        # regression target for the full 7-state endpoint Jacobian.
        dt = 1.0e-4
        kwargs_p = dict(base_kwargs)
        kwargs_m = dict(base_kwargs)
        kwargs_p["thrust_t_newtons"] += dt
        kwargs_m["thrust_t_newtons"] -= dt

        ta_p, _ = build_thrust_state_integrator(
            build_pert(**kwargs_p), ic, tol=1e-15, compact_mode=True
        )
        ta_m, _ = build_thrust_state_integrator(
            build_pert(**kwargs_m), ic, tol=1e-15, compact_mode=True
        )
        ta_p.propagate_until(t_final)
        ta_m.propagate_until(t_final)
        fd_t = (ta_p.state - ta_m.state) / (2.0 * dt)

        np.testing.assert_allclose(phi_p[:, col_t], fd_t, rtol=2e-4, atol=2e-8)

        # Isp mainly enters through mass depletion; validate that coupling too.
        di = 1.0
        kwargs_p = dict(base_kwargs)
        kwargs_m = dict(base_kwargs)
        kwargs_p["isp_s"] += di
        kwargs_m["isp_s"] -= di

        ta_p, _ = build_thrust_state_integrator(
            build_pert(**kwargs_p), ic, tol=1e-15, compact_mode=True
        )
        ta_m, _ = build_thrust_state_integrator(
            build_pert(**kwargs_m), ic, tol=1e-15, compact_mode=True
        )
        ta_p.propagate_until(t_final)
        ta_m.propagate_until(t_final)
        fd_isp = (ta_p.state - ta_m.state) / (2.0 * di)

        np.testing.assert_allclose(phi_p[6, col_isp], fd_isp[6], rtol=1e-7, atol=1e-11)
        assert abs(phi_p[6, col_isp]) > 0.0

    def test_j2_plus_thrust_matches_cowell(self):
        law = ConstantRTNThrustLaw(
            thrust_r_newtons=0.08,
            thrust_t_newtons=0.25,
            thrust_n_newtons=0.04,
            isp_s=1900.0,
        )
        pert = CompositePerturbation(
            conservative=[J2Perturbation()],
            non_conservative=[ContinuousThrustPerturbation(law)],
        )
        ic = _build_mass_state(pert, 500.0)
        t_final = 5400.0

        ta, _ = build_thrust_state_integrator(pert, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(t_final)
        r_geqoe, v_geqoe = geqoe2cart(ta.state, MU, pert)
        m_geqoe = float(ta.state[6])

        r_cow, v_cow, m_cow = propagate_cowell_heyoka_general(
            pert,
            R0_A,
            V0_A,
            t_final,
            m0=500.0,
            tol=1e-15,
            compact_mode=True,
        )

        assert np.linalg.norm(r_geqoe - r_cow) < 5.0e-4
        assert np.linalg.norm(v_geqoe - v_cow) < 1.0e-6
        assert m_geqoe == pytest.approx(m_cow, rel=0.0, abs=1e-11)
