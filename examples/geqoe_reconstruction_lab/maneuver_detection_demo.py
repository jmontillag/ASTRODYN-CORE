#!/usr/bin/env python
"""Constrained maneuver detection with local thrust uncertainty estimates.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_reconstruction_lab/maneuver_detection_demo.py

This example uses the GEqOE measurement and shooting stack to solve a simple
maneuver-detection problem:
  1. represent the trajectory as four candidate thrust arcs,
  2. fit mixed inertial position and inertial range samples,
  3. enforce continuity, fixed initial mass, and terminal equality constraints,
  4. regularize toward zero thrust on each candidate arc, and
  5. estimate a constrained local covariance for the recovered thrusts.

Detection is reported via per-arc thrust significance |T| / sigma_T.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrodyn_core.geqoe_taylor import (
    DecisionTrackingPenaltySpec,
    DecisionTrackingTerm,
    InertialPositionMeasurementModel,
    InertialRangeMeasurementModel,
    MeasurementObjectiveSpec,
    MultiArcShootingProblem,
    SampledMeasurement,
    ShootingArc,
    ShootingSolveSpec,
    SmoothnessPenaltySpec,
    TerminalConstraintSpec,
    build_thrust_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.geqoe_reconstruction_lab.position_fit_demo import (  # noqa: E402
    MU,
    R0,
    TRUTH_MASS_KG,
    V0,
    _header,
    _make_constant_perturbation,
    _make_output_dir,
)

warnings.filterwarnings("ignore", message="delta_grad == 0.0")
np.set_printoptions(precision=10, suppress=True)

ARC_DURATIONS_S = (330.0, 330.0, 330.0, 330.0)
QUIET_THRUST_N = 1.0e-4
TRUTH_THRUSTS_N = (QUIET_THRUST_N, 1.8, 0.9, QUIET_THRUST_N)
TRUTH_THRUSTS_DISPLAY = (0.0, 1.8, 0.9, 0.0)
POSITION_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 1.00)
RANGE_FRACTIONS = (0.15, 0.35, 0.55, 0.75, 0.95)

SIGMA_POS_KM = 0.06
SIGMA_RANGE_KM = 0.03
SIGMA_THRUST_PRIOR_N = 0.75
SMOOTHNESS_WEIGHT = 0.15
DETECTION_Z_THRESHOLD = 3.0

RANGE_REFERENCE_POSITION_KM = np.array([8878.1366, -900.0, 2100.0])

POSITION_MODEL = InertialPositionMeasurementModel()
RANGE_MODEL = InertialRangeMeasurementModel(RANGE_REFERENCE_POSITION_KM)


def _propagate_to_time(state0: np.ndarray, thrust_t: float, duration_s: float) -> np.ndarray:
    perturbation = _make_constant_perturbation(thrust_t)
    ta, _ = build_thrust_state_integrator(
        perturbation,
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )
    ta.propagate_until(duration_s)
    return np.array(ta.state, dtype=float, copy=True)


def _sample_arc(state0: np.ndarray, thrust_t: float, duration_s: float, n_samples: int = 180):
    perturbation = _make_constant_perturbation(thrust_t)
    ta, _ = build_thrust_state_integrator(
        perturbation,
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )
    t_grid = np.linspace(0.0, duration_s, n_samples)
    states = propagate_grid(ta, t_grid)
    positions = np.zeros((n_samples, 3), dtype=float)
    for i, state in enumerate(states):
        positions[i], _ = geqoe2cart(state, MU, perturbation)
    return t_grid, states, positions


def _build_arc_initial_states(initial_state: np.ndarray, thrusts_n: np.ndarray) -> list[np.ndarray]:
    states = [np.array(initial_state, dtype=float, copy=True)]
    current = states[0]
    for thrust_t, duration_s in zip(thrusts_n[:-1], ARC_DURATIONS_S[:-1], strict=True):
        current = _propagate_to_time(current, float(thrust_t), float(duration_s))
        states.append(current)
    return states


def _sample_full_trajectory(initial_state: np.ndarray, thrusts_n: np.ndarray):
    time_segments: list[np.ndarray] = []
    position_segments: list[np.ndarray] = []
    state_segments: list[np.ndarray] = []

    current = np.array(initial_state, dtype=float, copy=True)
    offset_s = 0.0
    for i, (thrust_t, duration_s) in enumerate(zip(thrusts_n, ARC_DURATIONS_S, strict=True)):
        times, states, positions = _sample_arc(current, float(thrust_t), float(duration_s))
        if i == 0:
            time_segments.append(offset_s + times)
            state_segments.append(states)
            position_segments.append(positions)
        else:
            time_segments.append(offset_s + times[1:])
            state_segments.append(states[1:, :])
            position_segments.append(positions[1:, :])
        current = states[-1].copy()
        offset_s += duration_s

    return (
        np.concatenate(time_segments),
        np.vstack(state_segments),
        np.vstack(position_segments),
    )


def _build_problem(
    arc_initial_states: list[np.ndarray],
    thrusts_n: np.ndarray,
) -> MultiArcShootingProblem:
    arcs = []
    for i, (state0, thrust_t, duration_s) in enumerate(
        zip(arc_initial_states, thrusts_n, ARC_DURATIONS_S, strict=True)
    ):
        arcs.append(
            ShootingArc(
                perturbation=_make_constant_perturbation(float(thrust_t)),
                initial_state=state0,
                duration_s=float(duration_s),
                parameter_names=("thrust.t_newtons",),
                name=f"arc{i}",
            )
        )
    return MultiArcShootingProblem(
        arcs,
        tol=1.0e-15,
        compact_mode=True,
    )


def _build_truth():
    x0_truth = np.concatenate(
        [
            cart2geqoe(
                R0,
                V0,
                MU,
                _make_constant_perturbation(TRUTH_THRUSTS_N[0]),
            ),
            [TRUTH_MASS_KG],
        ]
    )
    thrusts = np.array(TRUTH_THRUSTS_N, dtype=float)
    truth_problem = _build_problem(_build_arc_initial_states(x0_truth, thrusts), thrusts)
    return truth_problem, truth_problem.initial_guess(), x0_truth


def _build_estimation_problem(initial_state: np.ndarray):
    thrust_guess = np.full(len(ARC_DURATIONS_S), 0.05, dtype=float)
    problem = _build_problem(
        _build_arc_initial_states(initial_state, thrust_guess),
        thrust_guess,
    )
    return problem, problem.initial_guess()


def _build_measurements(
    truth_problem: MultiArcShootingProblem,
    truth_decision_vector: np.ndarray,
    rng: np.random.Generator,
) -> list[SampledMeasurement]:
    placeholders: list[SampledMeasurement] = [
        SampledMeasurement(
            model=POSITION_MODEL,
            value=np.zeros(3, dtype=float),
            arc_name="arc0",
            sample_time_s=0.0,
            name="arc0_initial_position",
        )
    ]

    for i, duration_s in enumerate(ARC_DURATIONS_S):
        arc_name = f"arc{i}"
        for j, fraction in enumerate(POSITION_FRACTIONS):
            placeholders.append(
                SampledMeasurement(
                    model=POSITION_MODEL,
                    value=np.zeros(3, dtype=float),
                    arc_name=arc_name,
                    sample_time_s=float(fraction * duration_s),
                    name=f"{arc_name}_position_{j}",
                )
            )
        for j, fraction in enumerate(RANGE_FRACTIONS):
            placeholders.append(
                SampledMeasurement(
                    model=RANGE_MODEL,
                    value=np.zeros(1, dtype=float),
                    arc_name=arc_name,
                    sample_time_s=float(fraction * duration_s),
                    name=f"{arc_name}_range_{j}",
                )
            )

    truth_eval = truth_problem.evaluate_measurements(
        truth_decision_vector,
        placeholders,
    )
    measurements: list[SampledMeasurement] = []
    for placeholder, sample in zip(placeholders, truth_eval.sample_results, strict=True):
        if placeholder.model.output_dimension == 3:
            sigma = SIGMA_POS_KM
            noise = rng.normal(0.0, sigma, size=3)
        else:
            sigma = SIGMA_RANGE_KM
            noise = rng.normal(0.0, sigma, size=1)
        measurements.append(
            SampledMeasurement.from_standard_deviation(
                model=placeholder.model,
                value=sample.predicted + noise,
                sigma=sigma,
                arc_name=sample.arc_name,
                sample_time_s=sample.sample_time_s,
                name=sample.name,
            )
        )
    return measurements


def _thrust_indices(problem: MultiArcShootingProblem) -> np.ndarray:
    return np.array(
        [
            problem.decision_index(f"arc{i}.thrust.t_newtons")
            for i in range(len(ARC_DURATIONS_S))
        ],
        dtype=int,
    )


def _plot_results(
    out_path: Path,
    truth_positions: np.ndarray,
    fit_positions: np.ndarray,
    measurements: list[SampledMeasurement],
    truth_thrusts: np.ndarray,
    fit_thrusts: np.ndarray,
    thrust_sigmas: np.ndarray,
    thrust_z_scores: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    fig.suptitle("GEqOE maneuver detection with local thrust uncertainty", fontsize=13)

    ax = axes[0]
    ax.plot(truth_positions[:, 0], truth_positions[:, 1], label="truth trajectory")
    ax.plot(fit_positions[:, 0], fit_positions[:, 1], linestyle="--", label="fitted trajectory")
    position_measurements = np.array(
        [
            measurement.value
            for measurement in measurements
            if measurement.model.output_dimension == 3
        ],
        dtype=float,
    )
    ax.scatter(
        position_measurements[:, 0],
        position_measurements[:, 1],
        marker="x",
        s=55,
        color="black",
        label="position samples",
    )
    ax.scatter(
        RANGE_REFERENCE_POSITION_KM[0],
        RANGE_REFERENCE_POSITION_KM[1],
        marker="o",
        s=70,
        color="tab:red",
        label="range reference",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Mixed measurement geometry")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    centers = np.arange(len(truth_thrusts), dtype=float)
    ax.step(centers, truth_thrusts, where="mid", linewidth=2.5, label="truth")
    ax.errorbar(
        centers,
        fit_thrusts,
        yerr=3.0 * thrust_sigmas,
        fmt="o",
        capsize=5,
        linewidth=1.6,
        label="fit +/- 3sigma",
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([f"arc{i}" for i in range(len(truth_thrusts))])
    ax.set_ylabel("Tangential thrust (N)")
    ax.set_title("Recovered arc thrusts with local uncertainty")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.bar(centers, thrust_z_scores, color="tab:blue")
    ax.axhline(DETECTION_Z_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"arc{i}" for i in range(len(truth_thrusts))])
    ax.set_ylabel("|T| / sigma_T")
    ax.set_title("Maneuver significance per candidate arc")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(21)
    out_dir = _make_output_dir()

    _header("1. Build A Sparse Maneuver Truth")
    truth_problem, x_truth, x0_truth = _build_truth()
    truth_eval = truth_problem.evaluate(x_truth)
    truth_final_state = truth_eval.arc_results[-1].final_state
    measurements = _build_measurements(truth_problem, x_truth, rng)
    print(f"Truth thrusts (N):        {TRUTH_THRUSTS_DISPLAY}")
    print(f"Candidate arcs:           {len(ARC_DURATIONS_S)}")
    print(f"Position sigma:           {SIGMA_POS_KM * 1.0e3:.1f} m")
    print(f"Range sigma:              {SIGMA_RANGE_KM * 1.0e3:.1f} m")
    print(
        "Observation batch:        "
        f"{sum(m.model.output_dimension == 3 for m in measurements)} positions + "
        f"{sum(m.model.output_dimension == 1 for m in measurements)} ranges"
    )

    _header("2. Build The Constrained Detection Problem")
    problem, x0 = _build_estimation_problem(x0_truth)
    thrust_indices = _thrust_indices(problem)
    fixed_initial_state = {
        f"arc0.{state_name}": x0_truth[i]
        for i, state_name in enumerate(problem.state_names)
    }
    bounds = problem.build_named_bounds(
        lower={**fixed_initial_state, "thrust.t_newtons": QUIET_THRUST_N, "m": 100.0},
        upper={**fixed_initial_state, "thrust.t_newtons": 2.5, "m": 1000.0},
    )
    tracking_penalty = DecisionTrackingPenaltySpec(
        [
            DecisionTrackingTerm(
                selector="thrust.t_newtons",
                target=0.0,
                sigma=SIGMA_THRUST_PRIOR_N,
            )
        ]
    )
    smoothness_penalty = SmoothnessPenaltySpec({"thrust.t_newtons": SMOOTHNESS_WEIGHT})
    terminal_constraint = TerminalConstraintSpec.equality(
        truth_final_state,
        output_indices=(0, 2, 3, 6),
    )
    spec = ShootingSolveSpec(
        bounds=bounds,
        measurement_objective=MeasurementObjectiveSpec(
            measurements=measurements,
            hessian_mode="quasi-newton",
        ),
        decision_tracking_penalty=tracking_penalty,
        terminal_constraint=terminal_constraint,
        smoothness_penalty=smoothness_penalty,
        options={"maxiter": 2000},
    )
    print("Constraints:")
    print(f"  continuity equations:   {problem.continuity_size}")
    print("  fixed initial state:    yes")
    print("  terminal equalities:    final nu, p2, K, and mass")
    print("Regularization:")
    print(f"  zero-thrust prior sigma:{SIGMA_THRUST_PRIOR_N:.2f} N")
    print(f"  smoothness weight:      {SMOOTHNESS_WEIGHT:.3f}")
    print(f"Initial thrust guess (N): {x0[thrust_indices]}")

    _header("3. Solve And Estimate Local Detection Uncertainty")
    t0 = time.perf_counter()
    solve = problem.solve(spec, decision_vector0=x0)
    covariance = problem.estimate_covariance(solve.x, spec)
    solve_s = time.perf_counter() - t0

    fit_eval = problem.evaluate_measurements(
        solve.x,
        spec.measurement_objective.measurements,
    )
    fit_initial_state = solve.x[:7]
    fit_thrusts = solve.x[thrust_indices]
    thrust_sigmas = covariance.standard_deviations[thrust_indices]
    thrust_z_scores = np.abs(fit_thrusts) / np.maximum(thrust_sigmas, 1.0e-12)
    detected = thrust_z_scores >= DETECTION_Z_THRESHOLD

    truth_times, _, truth_positions = _sample_full_trajectory(
        x0_truth,
        np.array(TRUTH_THRUSTS_N, dtype=float),
    )
    fit_times, _, fit_positions = _sample_full_trajectory(fit_initial_state, fit_thrusts)

    continuity_norm = np.linalg.norm(fit_eval.shooting_evaluation.continuity_residual)
    position_residual_norms = [
        np.linalg.norm(sample.raw_residual)
        for measurement, sample in zip(measurements, fit_eval.sample_results, strict=True)
        if measurement.model.output_dimension == 3
    ]
    range_residuals_km = np.array(
        [
            sample.raw_residual[0]
            for measurement, sample in zip(measurements, fit_eval.sample_results, strict=True)
            if measurement.model.output_dimension == 1
        ],
        dtype=float,
    )
    position_rms_m = np.sqrt(np.mean(np.square(position_residual_norms))) * 1.0e3
    range_rms_m = np.sqrt(np.mean(np.square(range_residuals_km))) * 1.0e3

    print(f"Solve success:            {solve.scipy_result.success}")
    print(f"Solver status:            {solve.scipy_result.status}")
    print(f"Solver message:           {solve.scipy_result.message}")
    print(f"Wall time:                {solve_s:.2f} s")
    print(f"Position-fit RMS:         {position_rms_m:.3f} m")
    print(f"Range-fit RMS:            {range_rms_m:.3f} m")
    print(f"Continuity residual norm: {continuity_norm:.3e}")
    print(f"Covariance dimension:     {covariance.effective_dimension}")
    print(f"Constraint rank:          {covariance.constraint_rank}")
    print(f"Reduced condition number: {covariance.reduced_condition_number:.3e}")
    print("")
    print("Arc thrust estimates:")
    for i, (truth_t, fit_t, sigma_t, z_score, is_detected) in enumerate(
        zip(
            TRUTH_THRUSTS_DISPLAY,
            fit_thrusts,
            thrust_sigmas,
            thrust_z_scores,
            detected,
            strict=True,
        )
    ):
        print(
            f"  arc{i}: truth={truth_t:.3f} N, "
            f"fit={fit_t:.3f} +/- {sigma_t:.3f} N, "
            f"|T|/sigma={z_score:.2f}, "
            f"detected={bool(is_detected)}"
        )

    out_path = out_dir / "maneuver_detection_demo.png"
    _plot_results(
        out_path,
        truth_positions,
        fit_positions,
        measurements,
        np.array(TRUTH_THRUSTS_DISPLAY, dtype=float),
        fit_thrusts,
        thrust_sigmas,
        thrust_z_scores,
    )
    print(f"Saved figure:             {out_path}")
    print(f"Detected arcs:            {[f'arc{i}' for i, flag in enumerate(detected) if flag]}")
    print(f"Fit trajectory span:      {fit_times[-1] / 60.0:.2f} min")
    print(f"Truth trajectory span:    {truth_times[-1] / 60.0:.2f} min")


if __name__ == "__main__":
    main()
