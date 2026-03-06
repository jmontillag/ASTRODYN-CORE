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
from astrodyn_core.geqoe_taylor import geqoe2cart
from astrodyn_core.geqoe_taylor.shooting import (
    DecisionTrackingPenaltySpec,
    DecisionTrackingTerm,
    InertialPositionMeasurementModel,
    InertialRangeMeasurementModel,
    MeasurementResidualEvaluation,
    MeasurementObjectiveSpec,
    MultiArcShootingProblem,
    SampledMeasurement,
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


def _build_position_measurements(
    problem: MultiArcShootingProblem,
    samples: list[tuple[str, float, str]],
    sigma_km: float | np.ndarray = 0.05,
) -> tuple[list[SampledMeasurement], MeasurementResidualEvaluation]:
    model = InertialPositionMeasurementModel()
    placeholders = [
        SampledMeasurement(
            model=model,
            value=np.zeros(3, dtype=float),
            arc_name=arc_name,
            sample_time_s=sample_time_s,
            name=name,
        )
        for arc_name, sample_time_s, name in samples
    ]
    prediction_eval = problem.evaluate_measurements(problem.initial_guess(), placeholders)
    measurements = [
        SampledMeasurement.from_standard_deviation(
            model=model,
            value=sample.predicted,
            sigma=sigma_km,
            arc_name=sample.arc_name,
            sample_time_s=sample.sample_time_s,
            name=sample.name,
        )
        for sample in prediction_eval.sample_results
    ]
    return measurements, prediction_eval


class TestMultiArcShooting:
    def test_inertial_position_measurement_model_matches_cartesian_and_fd(self):
        perturbation = _make_perturbation(0.18)
        state = _build_mass_state(perturbation, 500.0)
        model = InertialPositionMeasurementModel()

        predicted = model.evaluate(state, time_s=0.0, perturbation=perturbation)
        cart_position, _ = geqoe2cart(state, MU, perturbation)
        np.testing.assert_allclose(predicted, cart_position, atol=1.0e-12)

        jacobian = model.state_jacobian(state, time_s=0.0, perturbation=perturbation)
        cases = [
            (0, 1.0e-9),
            (1, 1.0e-8),
            (3, 1.0e-7),
            (4, 1.0e-8),
            (6, 1.0e-4),
        ]
        for index, step in cases:
            state_p = state.copy()
            state_m = state.copy()
            state_p[index] += step
            state_m[index] -= step
            pred_p = model.evaluate(state_p, time_s=0.0, perturbation=perturbation)
            pred_m = model.evaluate(state_m, time_s=0.0, perturbation=perturbation)
            fd_column = (pred_p - pred_m) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian[:, index],
                fd_column,
                rtol=1.0e-5,
                atol=1.0e-8,
            )

    def test_inertial_range_measurement_model_matches_position_model_and_fd(self):
        perturbation = _make_perturbation(0.18)
        state = _build_mass_state(perturbation, 500.0)
        reference = np.array([7050.0, -40.0, 25.0])
        position_model = InertialPositionMeasurementModel()
        range_model = InertialRangeMeasurementModel(reference)

        position = position_model.evaluate(state, time_s=0.0, perturbation=perturbation)
        predicted_range = range_model.evaluate(state, time_s=0.0, perturbation=perturbation)
        assert predicted_range[0] == pytest.approx(np.linalg.norm(position - reference))

        jacobian = range_model.state_jacobian(state, time_s=0.0, perturbation=perturbation)
        for index, step in ((0, 1.0e-9), (3, 1.0e-7), (6, 1.0e-4)):
            state_p = state.copy()
            state_m = state.copy()
            state_p[index] += step
            state_m[index] -= step
            pred_p = range_model.evaluate(state_p, time_s=0.0, perturbation=perturbation)
            pred_m = range_model.evaluate(state_m, time_s=0.0, perturbation=perturbation)
            fd_column = (pred_p - pred_m) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian[:, index],
                fd_column,
                rtol=1.0e-5,
                atol=1.0e-8,
            )

    def test_continuity_residual_closes_on_split_nominal_trajectory(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()

        result = problem.evaluate(x)

        assert problem.decision_names[0] == "arc0.nu"
        assert problem.decision_names[7] == "arc0.thrust.t_newtons"
        assert problem.continuity_constraint_names[0] == "arc0->arc1.nu"
        assert np.linalg.norm(result.continuity_residual) < 1.0e-11
        assert result.continuity_jacobian.shape == (7, problem.decision_size)

    def test_sampled_measurement_residual_closes_on_nominal_trajectory(self):
        problem = _build_two_arc_problem()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc1", 0.5 * 450.0, "arc1_mid"),
                ("arc1", 450.0, "arc1_end"),
            ],
        )

        evaluation = problem.evaluate_measurements(problem.initial_guess(), measurements)

        assert np.linalg.norm(evaluation.residual) < 1.0e-11
        assert np.linalg.norm(evaluation.shooting_evaluation.continuity_residual) < 1.0e-11
        assert evaluation.jacobian.shape == (9, problem.decision_size)

    def test_measurement_weights_scale_residual_and_jacobian(self):
        problem = _build_one_arc_problem()
        samples = [("arc0", 0.5 * 600.0, "arc0_mid")]
        base_measurements, _ = _build_position_measurements(
            problem,
            samples,
            sigma_km=1.0,
        )

        base_eval = problem.evaluate_measurements(problem.initial_guess(), base_measurements)
        predicted = base_eval.sample_results[0].predicted
        sigma_vec = np.array([0.05, 0.10, 0.20], dtype=float)
        offset = np.array([0.03, -0.04, 0.02], dtype=float)
        weighted_measurement = SampledMeasurement.from_standard_deviation(
            model=InertialPositionMeasurementModel(),
            value=predicted - offset,
            sigma=sigma_vec,
            arc_name="arc0",
            sample_time_s=0.5 * 600.0,
            name="arc0_mid",
        )

        weighted_eval = problem.evaluate_measurements(
            problem.initial_guess(),
            [weighted_measurement],
        )

        np.testing.assert_allclose(
            weighted_eval.residual,
            offset / sigma_vec,
            atol=1.0e-12,
        )
        expected_jacobian = np.diag(1.0 / sigma_vec) @ base_eval.jacobian.toarray()
        np.testing.assert_allclose(
            weighted_eval.jacobian.toarray(),
            expected_jacobian,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_mixed_position_and_range_batch_closes_and_matches_fd(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        position_model = InertialPositionMeasurementModel()
        range_model = InertialRangeMeasurementModel(np.array([7050.0, -40.0, 25.0]))
        placeholders = [
            SampledMeasurement(
                model=position_model,
                value=np.zeros(3, dtype=float),
                arc_name="arc0",
                sample_time_s=0.5 * 600.0,
                name="arc0_pos",
            ),
            SampledMeasurement(
                model=range_model,
                value=np.zeros(1, dtype=float),
                arc_name="arc1",
                sample_time_s=0.5 * 450.0,
                name="arc1_range",
            ),
        ]
        prediction_eval = problem.evaluate_measurements(x, placeholders)
        measurements = [
            SampledMeasurement.from_standard_deviation(
                model=position_model,
                value=prediction_eval.sample_results[0].predicted,
                sigma=0.05,
                arc_name="arc0",
                sample_time_s=0.5 * 600.0,
                name="arc0_pos",
            ),
            SampledMeasurement.from_standard_deviation(
                model=range_model,
                value=prediction_eval.sample_results[1].predicted,
                sigma=0.02,
                arc_name="arc1",
                sample_time_s=0.5 * 450.0,
                name="arc1_range",
            ),
        ]

        evaluation = problem.evaluate_measurements(x, measurements)
        assert np.linalg.norm(evaluation.residual) < 1.0e-11
        assert evaluation.jacobian.shape == (4, problem.decision_size)
        assert evaluation.residual_names[-1] == "arc1_range.range_km"

        idx = problem.decision_index("arc1.thrust.t_newtons")
        step = 1.0e-5
        x_p = x.copy()
        x_m = x.copy()
        x_p[idx] += step
        x_m[idx] -= step
        res_p, _ = problem.measurement_residuals(x_p, measurements)
        res_m, _ = problem.measurement_residuals(x_m, measurements)
        fd_value = (res_p[-1] - res_m[-1]) / (2.0 * step)
        np.testing.assert_allclose(
            evaluation.jacobian.toarray()[-1, idx],
            fd_value,
            rtol=5.0e-4,
            atol=2.0e-8,
        )

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

    def test_measurement_jacobian_matches_finite_difference(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc1", 0.5 * 450.0, "arc1_mid"),
            ],
            sigma_km=0.05,
        )
        evaluation = problem.evaluate_measurements(x, measurements)
        jacobian = evaluation.jacobian.toarray()

        cases = [
            ("arc0.K", 1.0e-7),
            ("arc0.thrust.t_newtons", 1.0e-5),
            ("arc1.q1", 1.0e-7),
            ("arc1.thrust.t_newtons", 1.0e-5),
        ]
        for name, step in cases:
            idx = problem.decision_index(name)
            x_p = x.copy()
            x_m = x.copy()
            x_p[idx] += step
            x_m[idx] -= step
            res_p, _ = problem.measurement_residuals(x_p, measurements)
            res_m, _ = problem.measurement_residuals(x_m, measurements)
            fd_column = (res_p - res_m) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian[:, idx],
                fd_column,
                rtol=5.0e-4,
                atol=2.0e-8,
            )

    def test_measurement_objective_matches_residual_norm_and_gradient(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc1", 0.5 * 450.0, "arc1_mid"),
            ],
        )
        measurement_eval = problem.evaluate_measurements(x, measurements)
        value, gradient, hessian = problem.measurement_objective(
            x,
            measurements,
            evaluation=measurement_eval,
            weight=1.5,
        )

        residual = measurement_eval.residual
        jacobian = measurement_eval.jacobian
        assert value == pytest.approx(0.75 * float(residual @ residual))
        np.testing.assert_allclose(
            gradient,
            np.asarray(1.5 * (jacobian.T @ residual), dtype=float).ravel(),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            hessian.toarray(),
            (1.5 * (jacobian.T @ jacobian)).toarray(),
            atol=1.0e-12,
        )

    def test_measurement_objective_spec_rejects_unknown_hessian_mode(self):
        measurement = SampledMeasurement(
            model=InertialPositionMeasurementModel(),
            value=np.zeros(3, dtype=float),
            arc_name="arc0",
            sample_time_s=0.0,
            name="arc0_initial",
        )
        with pytest.raises(ValueError, match="hessian_mode"):
            MeasurementObjectiveSpec(
                [measurement],
                hessian_mode="not-a-real-mode",  # type: ignore[arg-type]
            )

    def test_estimate_covariance_matches_scalar_inverse_with_fixed_bounds(self):
        problem = _build_one_arc_problem()
        x_nom = problem.initial_guess()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc0", 600.0, "arc0_end"),
            ],
        )
        fixed_state_bounds = {
            state_name: x_nom[problem.decision_index(f"arc0.{state_name}")]
            for state_name in problem.state_names
        }
        bounds = problem.build_named_bounds(
            lower={**fixed_state_bounds, "thrust.t_newtons": 0.05},
            upper={**fixed_state_bounds, "thrust.t_newtons": 0.35},
        )
        tracking_penalty = DecisionTrackingPenaltySpec(
            [DecisionTrackingTerm("thrust.t_newtons", target=0.0, sigma=10.0)]
        )
        spec = ShootingSolveSpec(
            bounds=bounds,
            measurement_objective=MeasurementObjectiveSpec(measurements),
            decision_tracking_penalty=tracking_penalty,
        )

        covariance = problem.estimate_covariance(x_nom, spec)
        thrust_idx = problem.decision_index("arc0.thrust.t_newtons")
        _, _, measurement_hessian = problem.measurement_objective(x_nom, measurements)
        _, _, tracking_hessian = problem.decision_tracking_objective(
            x_nom,
            tracking_penalty,
        )
        expected_precision = (
            measurement_hessian[thrust_idx, thrust_idx]
            + tracking_hessian[thrust_idx, thrust_idx]
        )

        assert covariance.effective_dimension == 1
        assert covariance.constraint_rank == len(problem.state_names)
        assert covariance.standard_deviations[thrust_idx] == pytest.approx(
            1.0 / np.sqrt(expected_precision),
            rel=1.0e-10,
        )
        for state_name in problem.state_names:
            idx = problem.decision_index(f"arc0.{state_name}")
            assert covariance.standard_deviations[idx] == pytest.approx(0.0, abs=1.0e-14)

    def test_estimate_covariance_handles_continuity_and_terminal_equalities(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc1", 0.5 * 450.0, "arc1_mid"),
            ],
        )
        target_state = problem.evaluate(x).arc_results[-1].final_state
        spec = ShootingSolveSpec(
            measurement_objective=MeasurementObjectiveSpec(measurements),
            terminal_constraint=TerminalConstraintSpec.equality(
                target_state,
                output_indices=(3, 6),
            ),
        )

        covariance = problem.estimate_covariance(x, spec)

        assert len(covariance.equality_constraint_names) == problem.continuity_size + 2
        assert covariance.constraint_rank >= problem.continuity_size
        assert covariance.effective_dimension == (
            problem.decision_size - covariance.constraint_rank
        )
        assert np.all(np.isfinite(covariance.standard_deviations))
        assert np.all(covariance.standard_deviations >= 0.0)

    def test_decision_tracking_objective_matches_finite_difference(self):
        problem = _build_two_arc_problem()
        x = problem.initial_guess()
        penalty = DecisionTrackingPenaltySpec(
            [
                DecisionTrackingTerm("thrust.t_newtons", target=0.15, sigma=0.25),
                DecisionTrackingTerm("arc0.m", target=490.0, sigma=2.0),
            ]
        )

        value, gradient, hessian = problem.decision_tracking_objective(x, penalty)
        assert value > 0.0

        for name, step in (
            ("arc0.thrust.t_newtons", 1.0e-6),
            ("arc1.thrust.t_newtons", 1.0e-6),
            ("arc0.m", 1.0e-5),
        ):
            idx = problem.decision_index(name)
            x_p = x.copy()
            x_m = x.copy()
            x_p[idx] += step
            x_m[idx] -= step
            value_p, _, _ = problem.decision_tracking_objective(x_p, penalty)
            value_m, _, _ = problem.decision_tracking_objective(x_m, penalty)
            fd_grad = (value_p - value_m) / (2.0 * step)
            np.testing.assert_allclose(
                gradient[idx],
                fd_grad,
                rtol=1.0e-7,
                atol=1.0e-9,
            )

        thrust_weight = 1.0 / (0.25**2)
        mass_weight = 1.0 / (2.0**2)
        idx_t0 = problem.decision_index("arc0.thrust.t_newtons")
        idx_t1 = problem.decision_index("arc1.thrust.t_newtons")
        idx_m0 = problem.decision_index("arc0.m")
        assert hessian[idx_t0, idx_t0] == pytest.approx(thrust_weight)
        assert hessian[idx_t1, idx_t1] == pytest.approx(thrust_weight)
        assert hessian[idx_m0, idx_m0] == pytest.approx(mass_weight)

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
    def test_solve_spec_supports_measurement_objective(self):
        problem = _build_one_arc_problem()
        x_nom = problem.initial_guess()
        measurements, _ = _build_position_measurements(
            problem,
            [
                ("arc0", 0.5 * 600.0, "arc0_mid"),
                ("arc0", 600.0, "arc0_end"),
            ],
        )

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
            measurement_objective=MeasurementObjectiveSpec(measurements),
            decision_tracking_penalty=DecisionTrackingPenaltySpec(
                [DecisionTrackingTerm("thrust.t_newtons", target=0.0, sigma=10.0)]
            ),
            options={"maxiter": 100},
        )

        x_init = x_nom.copy()
        param_idx = problem.decision_index("arc0.thrust.t_newtons")
        x_init[param_idx] = 0.30
        solve = problem.solve(spec, decision_vector0=x_init)
        fit_eval = problem.evaluate_measurements(solve.x, measurements)

        assert solve.scipy_result.success
        assert abs(solve.x[param_idx] - x_nom[param_idx]) < 1.0e-4
        assert np.linalg.norm(fit_eval.residual) < 5.0e-4

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
