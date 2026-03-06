"""Tests for the Phase 8c GEqOE Taylor multiple-shooting layer."""

from __future__ import annotations

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor.constants import MU
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe
from astrodyn_core.geqoe_taylor.integrator import build_thrust_state_integrator
from astrodyn_core.geqoe_taylor.perturbations.composite import CompositePerturbation
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.perturbations.thrust import ContinuousThrustPerturbation
from astrodyn_core.geqoe_taylor.shooting import (
    MultiArcShootingProblem,
    ShootingArc,
    ShootingSolveSpec,
    SmoothnessPenaltySpec,
    TerminalConstraintSpec,
)
from astrodyn_core.geqoe_taylor.thrust import ConstantRTNThrustLaw

R0_A = np.array([7178.1366, 0.0, 0.0])
V0_A = np.array([0.0, 5.269240572916780, 5.269240572916780])


def _build_mass_state(perturbation, mass_kg: float) -> np.ndarray:
    geqoe = cart2geqoe(R0_A, V0_A, MU, perturbation)
    return np.concatenate([geqoe, [mass_kg]])


def _make_perturbation(thrust_t_newtons: float, isp_s: float = 2100.0):
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[
            ContinuousThrustPerturbation(
                ConstantRTNThrustLaw(thrust_t_newtons=thrust_t_newtons, isp_s=isp_s)
            )
        ],
    )


def _build_two_arc_problem() -> MultiArcShootingProblem:
    duration0 = 600.0
    duration1 = 450.0
    pert0 = _make_perturbation(0.18)
    pert1 = _make_perturbation(0.22)

    x0 = _build_mass_state(pert0, 500.0)
    ta0, _ = build_thrust_state_integrator(pert0, x0, tol=1e-15, compact_mode=True)
    ta0.propagate_until(duration0)
    x1 = ta0.state.copy()

    return MultiArcShootingProblem(
        [
            ShootingArc(
                perturbation=pert0,
                initial_state=x0,
                duration_s=duration0,
                parameter_names=("thrust.t_newtons",),
                name="arc0",
            ),
            ShootingArc(
                perturbation=pert1,
                initial_state=x1,
                duration_s=duration1,
                parameter_names=("thrust.t_newtons",),
                name="arc1",
            ),
        ],
        tol=1e-15,
        compact_mode=True,
    )


def _build_one_arc_problem() -> MultiArcShootingProblem:
    duration = 600.0
    pert = _make_perturbation(0.18)
    x0 = _build_mass_state(pert, 500.0)
    return MultiArcShootingProblem(
        [
            ShootingArc(
                perturbation=pert,
                initial_state=x0,
                duration_s=duration,
                parameter_names=("thrust.t_newtons",),
                name="arc0",
            )
        ],
        tol=1e-15,
        compact_mode=True,
    )


class TestMultiArcShooting:
    def test_continuity_residual_closes_on_split_nominal_trajectory(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()

        result = problem.evaluate(x)

        assert problem.decision_names[0] == "arc0.nu"
        assert problem.decision_names[7] == "arc0.thrust.t_newtons"
        assert problem.continuity_constraint_names[0] == "arc0->arc1.nu"
        assert np.linalg.norm(result.continuity_residual) < 1.0e-11
        assert result.continuity_jacobian.shape == (7, problem.decision_size)

    def test_continuity_jacobian_matches_finite_difference(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        result = problem.evaluate(x)
        jac = result.continuity_jacobian.toarray()

        cases = [
            ("arc0.K", 1.0e-7),
            ("arc0.m", 1.0e-6),
            ("arc0.thrust.t_newtons", 1.0e-5),
            ("arc1.q1", 1.0e-7),
            ("arc1.m", 1.0e-6),
        ]

        for name, step in cases:
            idx = problem.decision_index(name)
            x_p = x.copy()
            x_m = x.copy()
            x_p[idx] += step
            x_m[idx] -= step
            res_p, _ = problem.continuity_constraints(x_p)
            res_m, _ = problem.continuity_constraints(x_m)
            fd_col = (res_p - res_m) / (2.0 * step)
            np.testing.assert_allclose(
                jac[:, idx],
                fd_col,
                rtol=4.0e-4,
                atol=2.0e-8,
            )

    def test_terminal_constraints_and_minimum_propellant_gradient(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        result = problem.evaluate(x)

        target = result.arc_results[-1].final_state.copy()
        residual, jac = problem.terminal_constraints(
            x,
            target_state=target,
            output_indices=[1, 3, 6],
            evaluation=result,
        )

        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)
        assert jac.shape == (3, problem.decision_size)

        _, gradient = problem.minimum_propellant_objective(x, evaluation=result)
        cases = [
            ("arc0.m", 1.0e-6),
            ("arc1.m", 1.0e-6),
            ("arc1.thrust.t_newtons", 1.0e-5),
        ]

        for name, step in cases:
            idx = problem.decision_index(name)
            x_p = x.copy()
            x_m = x.copy()
            x_p[idx] += step
            x_m[idx] -= step
            obj_p, _ = problem.minimum_propellant_objective(x_p)
            obj_m, _ = problem.minimum_propellant_objective(x_m)
            fd_grad = (obj_p - obj_m) / (2.0 * step)
            np.testing.assert_allclose(
                gradient[idx],
                fd_grad,
                rtol=4.0e-4,
                atol=2.0e-8,
            )

    def test_named_bounds_support_suffix_and_exact_selectors(self):
        problem = _build_two_arc_problem()
        bounds = problem.build_named_bounds(
            lower={"m": 450.0, "thrust.t_newtons": 0.1},
            upper={"m": 600.0, "arc1.thrust.t_newtons": 0.3},
        )

        idx_m0 = problem.decision_index("arc0.m")
        idx_m1 = problem.decision_index("arc1.m")
        idx_t0 = problem.decision_index("arc0.thrust.t_newtons")
        idx_t1 = problem.decision_index("arc1.thrust.t_newtons")

        assert bounds.lb[idx_m0] == pytest.approx(450.0)
        assert bounds.lb[idx_m1] == pytest.approx(450.0)
        assert bounds.ub[idx_m0] == pytest.approx(600.0)
        assert bounds.ub[idx_m1] == pytest.approx(600.0)
        assert bounds.lb[idx_t0] == pytest.approx(0.1)
        assert bounds.lb[idx_t1] == pytest.approx(0.1)
        assert np.isinf(bounds.ub[idx_t0])
        assert bounds.ub[idx_t1] == pytest.approx(0.3)

    def test_control_smoothness_gradient_matches_finite_difference(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()

        selector_weights = {"thrust.t_newtons": 2.5}
        value, gradient, hessian = problem.control_smoothness_objective(
            x, selector_weights
        )

        idx0 = problem.decision_index("arc0.thrust.t_newtons")
        idx1 = problem.decision_index("arc1.thrust.t_newtons")
        step = 1.0e-6

        for idx in (idx0, idx1):
            x_p = x.copy()
            x_m = x.copy()
            x_p[idx] += step
            x_m[idx] -= step
            value_p, _, _ = problem.control_smoothness_objective(x_p, selector_weights)
            value_m, _, _ = problem.control_smoothness_objective(x_m, selector_weights)
            fd_grad = (value_p - value_m) / (2.0 * step)
            np.testing.assert_allclose(gradient[idx], fd_grad, rtol=1.0e-8, atol=1.0e-10)

        expected_hessian = np.array([[2.5, -2.5], [-2.5, 2.5]])
        np.testing.assert_allclose(
            hessian[np.ix_([idx0, idx1], [idx0, idx1])].toarray(),
            expected_hessian,
            atol=1.0e-12,
        )
        assert value > 0.0

    @pytest.mark.filterwarnings("ignore:delta_grad == 0.0")
    def test_trust_constr_adapter_solves_one_arc_terminal_target(self):
        problem = _build_one_arc_problem()
        x_nom = problem.initial_guess()
        nominal_eval = problem.evaluate(x_nom)
        target = nominal_eval.arc_results[-1].final_state.copy()
        param_idx = problem.decision_index("arc0.thrust.t_newtons")

        fixed_state_bounds = {
            state_name: x_nom[problem.decision_index(f"arc0.{state_name}")]
            for state_name in problem.state_names
        }
        bounds = problem.build_named_bounds(
            lower={**fixed_state_bounds, "thrust.t_newtons": 0.05},
            upper={**fixed_state_bounds, "thrust.t_newtons": 0.35},
        )

        x_init = x_nom.copy()
        x_init[param_idx] = 0.24
        nominal_obj, _ = problem.minimum_propellant_objective(x_nom)
        initial_obj, _ = problem.minimum_propellant_objective(x_init)
        solve = problem.solve_minimum_propellant(
            target_state=target,
            output_indices=[3],
            decision_vector0=x_init,
            bounds=bounds,
            options={"maxiter": 100},
        )

        assert solve.scipy_result.success
        assert solve.terminal_residual is not None
        assert abs(solve.terminal_residual[0]) < 1.0e-10
        assert 0.05 <= solve.x[param_idx] <= 0.35
        assert solve.objective <= nominal_obj + 1.0e-12
        assert solve.objective <= initial_obj + 1.0e-12

    def test_generic_solve_spec_supports_terminal_bounds(self):
        problem = _build_one_arc_problem()
        x_nom = problem.initial_guess()
        nominal_eval = problem.evaluate(x_nom)
        outputs, _ = problem.terminal_outputs(x_nom, output_indices=[3])

        fixed_state_bounds = {
            state_name: x_nom[problem.decision_index(f"arc0.{state_name}")]
            for state_name in problem.state_names
        }
        bounds = problem.build_named_bounds(
            lower={**fixed_state_bounds, "thrust.t_newtons": 0.05},
            upper={**fixed_state_bounds, "thrust.t_newtons": 0.35},
        )
        spec = ShootingSolveSpec(
            bounds=bounds,
            terminal_constraint=TerminalConstraintSpec(
                lower=outputs - 1.0e-12,
                upper=outputs + 1.0e-12,
                output_indices=[3],
            ),
        )

        x_init = x_nom.copy()
        x_init[problem.decision_index("arc0.thrust.t_newtons")] = 0.24
        solve = problem.solve(spec, decision_vector0=x_init)

        assert solve.scipy_result.success
        assert solve.terminal_outputs is not None
        assert solve.terminal_violation is not None
        assert np.max(solve.terminal_violation) < 1.0e-9
        assert solve.continuity_residual.size == nominal_eval.continuity_residual.size

    def test_convenience_solver_accepts_smoothness_penalty(self):
        problem = _build_two_arc_problem()
        x0 = problem.initial_guess()
        target = problem.evaluate(x0).arc_results[-1].final_state.copy()

        fixed_state_bounds = {}
        for arc_name in ("arc0", "arc1"):
            for state_name in problem.state_names:
                fixed_state_bounds[f"{arc_name}.{state_name}"] = x0[
                    problem.decision_index(f"{arc_name}.{state_name}")
                ]
        bounds = problem.build_named_bounds(
            lower={**fixed_state_bounds, "thrust.t_newtons": 0.05},
            upper={**fixed_state_bounds, "thrust.t_newtons": 0.35},
        )
        x_init = x0.copy()
        x_init[problem.decision_index("arc0.thrust.t_newtons")] = 0.30
        x_init[problem.decision_index("arc1.thrust.t_newtons")] = 0.10

        solve = problem.solve_minimum_propellant(
            target_state=target,
            output_indices=[3, 6],
            decision_vector0=x_init,
            bounds=bounds,
            smoothness_penalty=SmoothnessPenaltySpec({"thrust.t_newtons": 5.0}),
            options={"maxiter": 100},
        )

        assert solve.scipy_result.success
        assert solve.terminal_residual is not None
        assert np.max(np.abs(solve.terminal_residual)) < 1.0e-10
        delta_final = abs(
            solve.x[problem.decision_index("arc0.thrust.t_newtons")]
            - solve.x[problem.decision_index("arc1.thrust.t_newtons")]
        )
        delta_initial = abs(
            x_init[problem.decision_index("arc0.thrust.t_newtons")]
            - x_init[problem.decision_index("arc1.thrust.t_newtons")]
        )
        assert delta_final <= delta_initial + 1.0e-12
