#!/usr/bin/env python
"""Synthetic maneuver reconstruction from inertial-position samples.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_reconstruction_lab/position_fit_demo.py

This example uses the GEqOE multiple-shooting measurement layer directly:
  1. generate a two-arc truth trajectory with piecewise-constant thrust,
  2. sample noisy inertial positions at intermediate arc times,
  3. build a two-arc GEqOE shooting transcription,
  4. minimize weighted position residuals plus maneuver effort, and
  5. enforce inter-arc continuity exactly through the shooting constraints.

The implemented toy measurement model is inertial position only. To keep the
example well posed, the initial mass is held fixed while the initial orbital
state and arc thrust levels are estimated.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrodyn_core.geqoe_taylor import (
    MU,
    CompositePerturbation,
    ConstantRTNThrustLaw,
    ContinuousThrustPerturbation,
    DecisionTrackingPenaltySpec,
    DecisionTrackingTerm,
    InertialPositionMeasurementModel,
    J2Perturbation,
    MultiArcShootingProblem,
    SampledMeasurement,
    ShootingArc,
    build_thrust_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

np.set_printoptions(precision=10, suppress=True)
warnings.filterwarnings("ignore", message="delta_grad == 0.0")

R0 = np.array([7178.1366, 0.0, 0.0])
V0 = np.array([0.0, 5.269240572916780, 5.269240572916780])

TRUTH_MASS_KG = 480.0
ARC_DURATIONS_S = (600.0, 720.0)
TRUTH_THRUSTS_N = (2.0, 0.8)
MEASUREMENT_FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)

SIGMA_POS_KM = 0.05
SIGMA_THRUST_N = 10.0

POSITION_MODEL = InertialPositionMeasurementModel()


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _make_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "generated" / "geqoe_reconstruction_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _make_constant_perturbation(thrust_t_newtons: float, isp_s: float = 2100.0):
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[
            ContinuousThrustPerturbation(
                ConstantRTNThrustLaw(thrust_t_newtons=thrust_t_newtons, isp_s=isp_s)
            )
        ],
    )


def _propagate_to_time(state0: np.ndarray, thrust_t: float, t_final_s: float) -> np.ndarray:
    perturbation = _make_constant_perturbation(thrust_t)
    ta, _ = build_thrust_state_integrator(
        perturbation,
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )
    ta.propagate_until(t_final_s)
    return np.array(ta.state, dtype=float, copy=True)


def _sample_arc(state0: np.ndarray, thrust_t: float, duration_s: float, n_samples: int = 280):
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


def _sample_full_trajectory(initial_state: np.ndarray, thrusts_n: np.ndarray):
    thrust0, thrust1 = thrusts_n
    t0, states0, pos0 = _sample_arc(initial_state, thrust0, ARC_DURATIONS_S[0])
    x1 = states0[-1].copy()
    t1, states1, pos1 = _sample_arc(x1, thrust1, ARC_DURATIONS_S[1])
    return (
        np.concatenate([t0, ARC_DURATIONS_S[0] + t1[1:]]),
        np.vstack([states0, states1[1:, :]]),
        np.vstack([pos0, pos1[1:, :]]),
    )


def _build_problem(
    initial_state_arc0: np.ndarray,
    initial_state_arc1: np.ndarray,
    thrusts_n: np.ndarray,
) -> MultiArcShootingProblem:
    return MultiArcShootingProblem(
        [
            ShootingArc(
                perturbation=_make_constant_perturbation(float(thrusts_n[0])),
                initial_state=initial_state_arc0,
                duration_s=ARC_DURATIONS_S[0],
                parameter_names=("thrust.t_newtons",),
                name="arc0",
            ),
            ShootingArc(
                perturbation=_make_constant_perturbation(float(thrusts_n[1])),
                initial_state=initial_state_arc1,
                duration_s=ARC_DURATIONS_S[1],
                parameter_names=("thrust.t_newtons",),
                name="arc1",
            ),
        ],
        tol=1.0e-15,
        compact_mode=True,
    )


def _measurement_schedule() -> list[tuple[str, float, str]]:
    schedule: list[tuple[str, float, str]] = [("arc0", 0.0, "arc0_initial")]
    for arc_name, duration_s in zip(("arc0", "arc1"), ARC_DURATIONS_S, strict=True):
        for i, fraction in enumerate(MEASUREMENT_FRACTIONS):
            schedule.append(
                (
                    arc_name,
                    fraction * duration_s,
                    f"{arc_name}_sample{i}",
                )
            )
    return schedule


def _measurement_placeholders() -> list[SampledMeasurement]:
    return [
        SampledMeasurement(
            model=POSITION_MODEL,
            value=np.zeros(3, dtype=float),
            arc_name=arc_name,
            sample_time_s=sample_time_s,
            name=name,
        )
        for arc_name, sample_time_s, name in _measurement_schedule()
    ]


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
    x1_truth = _propagate_to_time(x0_truth, TRUTH_THRUSTS_N[0], ARC_DURATIONS_S[0])
    truth_problem = _build_problem(
        x0_truth,
        x1_truth,
        np.asarray(TRUTH_THRUSTS_N, dtype=float),
    )
    return truth_problem, truth_problem.initial_guess(), x0_truth


def _build_measurements(
    truth_problem: MultiArcShootingProblem,
    truth_decision_vector: np.ndarray,
    rng: np.random.Generator,
) -> list[SampledMeasurement]:
    truth_eval = truth_problem.evaluate_measurements(
        truth_decision_vector,
        _measurement_placeholders(),
    )
    measurements: list[SampledMeasurement] = []
    for sample in truth_eval.sample_results:
        measurements.append(
            SampledMeasurement.from_standard_deviation(
                model=POSITION_MODEL,
                value=sample.predicted + rng.normal(0.0, SIGMA_POS_KM, size=3),
                sigma=SIGMA_POS_KM,
                arc_name=sample.arc_name,
                sample_time_s=sample.sample_time_s,
                name=sample.name,
            )
        )
    return measurements


def _build_initial_guess():
    r0_prior = R0 + np.array([0.12, -0.08, 0.05])
    v0_prior = V0 + np.array([1.8e-4, -1.2e-4, 0.9e-4])
    x0_prior = np.concatenate(
        [
            cart2geqoe(r0_prior, v0_prior, MU, _make_constant_perturbation(0.0)),
            [TRUTH_MASS_KG],
        ]
    )
    thrust_guess = np.array([0.20, 0.20], dtype=float)
    return x0_prior, thrust_guess


def _build_estimation_problem() -> tuple[MultiArcShootingProblem, np.ndarray]:
    x0_prior, thrust_guess = _build_initial_guess()
    x1_prior = _propagate_to_time(x0_prior, thrust_guess[0], ARC_DURATIONS_S[0])
    problem = _build_problem(x0_prior, x1_prior, thrust_guess)
    return problem, problem.initial_guess()


def _thrust_indices(problem: MultiArcShootingProblem) -> np.ndarray:
    return np.array(
        [
            problem.decision_index("arc0.thrust.t_newtons"),
            problem.decision_index("arc1.thrust.t_newtons"),
        ],
        dtype=int,
    )


def _plot_results(
    out_path: Path,
    truth_times: np.ndarray,
    truth_states: np.ndarray,
    truth_positions: np.ndarray,
    fit_times: np.ndarray,
    fit_states: np.ndarray,
    fit_positions: np.ndarray,
    measurements: list[SampledMeasurement],
    fit_thrusts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("GEqOE shooting-based position-fit experiment", fontsize=13)

    ax = axes[0]
    schedule_t = (
        np.array([0.0, ARC_DURATIONS_S[0], ARC_DURATIONS_S[0], sum(ARC_DURATIONS_S)])
        / 60.0
    )
    truth_schedule_u = np.array(
        [TRUTH_THRUSTS_N[0], TRUTH_THRUSTS_N[0], TRUTH_THRUSTS_N[1], TRUTH_THRUSTS_N[1]]
    )
    fit_schedule_u = np.array([fit_thrusts[0], fit_thrusts[0], fit_thrusts[1], fit_thrusts[1]])
    ax.plot(schedule_t, truth_schedule_u, label="truth", linewidth=2.5)
    ax.plot(schedule_t, fit_schedule_u, label="fit", linestyle="--", linewidth=2.5)
    ax.set_ylabel("Tangential thrust (N)")
    ax.set_title("Recovered maneuver coefficients")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(truth_positions[:, 0], truth_positions[:, 1], label="truth trajectory")
    ax.plot(fit_positions[:, 0], fit_positions[:, 1], linestyle="--", label="fitted trajectory")

    arc0_measurements = np.array(
        [measurement.value for measurement in measurements if measurement.arc_name == "arc0"],
        dtype=float,
    )
    arc1_measurements = np.array(
        [measurement.value for measurement in measurements if measurement.arc_name == "arc1"],
        dtype=float,
    )
    if len(arc0_measurements) > 0:
        ax.scatter(
            arc0_measurements[:, 0],
            arc0_measurements[:, 1],
            marker="x",
            s=70,
            color="black",
            label="arc 0 position samples",
        )
    if len(arc1_measurements) > 0:
        ax.scatter(
            arc1_measurements[:, 0],
            arc1_measurements[:, 1],
            marker="x",
            s=70,
            color="tab:red",
            label="arc 1 position samples",
        )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Intermediate inertial-position measurements")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(truth_times / 60.0, truth_states[:, 6], label="truth mass")
    ax.plot(fit_times / 60.0, fit_states[:, 6], linestyle="--", label="fitted mass")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass depletion implied by the recovered maneuver")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(42)
    out_dir = _make_output_dir()

    _header("1. Generate Synthetic Truth And Position Measurements")
    truth_problem, x_truth, x0_truth = _build_truth()
    measurements = _build_measurements(truth_problem, x_truth, rng)
    print(f"Truth thrusts (N):        {TRUTH_THRUSTS_N}")
    print(f"Measurement sigma:        {SIGMA_POS_KM * 1.0e3:.1f} m")
    print(f"Position samples:         {len(measurements)}")
    print("Observation model:        inertial Cartesian position")

    _header("2. Build The GEqOE Shooting Estimate")
    problem, x0 = _build_estimation_problem()
    bounds = problem.build_named_bounds(
        lower={"arc0.m": TRUTH_MASS_KG, "arc1.m": 100.0, "thrust.t_newtons": 0.0},
        upper={"arc0.m": TRUTH_MASS_KG, "arc1.m": 1000.0, "thrust.t_newtons": 2.5},
    )
    thrust_indices = _thrust_indices(problem)
    print("Decision vector:          two 7-state arc nodes + two thrust coefficients")
    print(f"Continuity constraints:   {problem.continuity_size}")
    print(f"Initial thrust guess (N): {x0[thrust_indices]}")
    print("Initial mass handling:    fixed at the known truth mass for this toy model")

    tracking_penalty = DecisionTrackingPenaltySpec(
        [
            DecisionTrackingTerm(
                selector="thrust.t_newtons",
                target=0.0,
                sigma=SIGMA_THRUST_N,
            )
        ]
    )

    _header("3. Solve The Position-Fit Reconstruction")
    t0 = time.perf_counter()
    solve = problem.solve_measurement_fit(
        measurements,
        decision_vector0=x0,
        bounds=bounds,
        measurement_hessian_mode="quasi-newton",
        decision_tracking_penalty=tracking_penalty,
        options={"maxiter": 500},
    )
    solve_s = time.perf_counter() - t0

    fit_eval = problem.evaluate_measurements(solve.x, measurements)
    fit_initial_state = solve.x[:7]
    fit_thrusts = solve.x[thrust_indices]
    fit_times, fit_states, fit_positions = _sample_full_trajectory(
        fit_initial_state,
        fit_thrusts,
    )
    truth_times, truth_states, truth_positions = _sample_full_trajectory(
        x0_truth,
        np.array(TRUTH_THRUSTS_N, dtype=float),
    )

    residual_norms = [
        np.linalg.norm(sample.raw_residual) for sample in fit_eval.sample_results
    ]
    rms_m = np.sqrt(np.mean(np.square(residual_norms))) * 1.0e3
    continuity_norm = np.linalg.norm(
        fit_eval.shooting_evaluation.continuity_residual
    )

    print(f"Solve success:            {solve.scipy_result.success}")
    print(f"Solver status:            {solve.scipy_result.status}")
    print(f"Solver message:           {solve.scipy_result.message}")
    print(f"Wall time:                {solve_s:.2f} s")
    print(f"Recovered thrusts (N):    {fit_thrusts}")
    print(f"True thrusts (N):         {TRUTH_THRUSTS_N}")
    print(f"Position-fit RMS:         {rms_m:.3f} m")
    print(f"Continuity residual norm: {continuity_norm:.3e}")
    print(f"Objective value:          {solve.objective:.6e}")
    print(f"Recovered mass use (kg):  {fit_states[0, 6] - fit_states[-1, 6]:.6f}")

    out_path = out_dir / "continuous_thrust_position_fit.png"
    _plot_results(
        out_path,
        truth_times,
        truth_states,
        truth_positions,
        fit_times,
        fit_states,
        fit_positions,
        measurements,
        fit_thrusts,
    )
    print(f"Saved figure:             {out_path}")


if __name__ == "__main__":
    main()
