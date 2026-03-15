#!/usr/bin/env python
"""Synthetic maneuver reconstruction from mixed position and range samples.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_reconstruction_lab/mixed_position_range_fit_demo.py

This example reuses the GEqOE shooting measurement layer to mix two toy
observation types in one transcription:
  1. inertial Cartesian position samples, and
  2. inertial one-way range samples to a fixed inertial reference point.

Both measurement families are assembled, weighted, and differentiated through
the same 7-state GEqOE + mass sensitivity machinery used elsewhere in the
multiple-shooting prototype.
"""

from __future__ import annotations

import time
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrodyn_core.geqoe_taylor import (
    DecisionTrackingPenaltySpec,
    DecisionTrackingTerm,
    InertialRangeMeasurementModel,
    SampledMeasurement,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.geqoe_reconstruction_lab.position_fit_demo import (
    ARC_DURATIONS_S,
    MEASUREMENT_FRACTIONS,
    POSITION_MODEL,
    SIGMA_POS_KM,
    SIGMA_THRUST_N,
    TRUTH_MASS_KG,
    TRUTH_THRUSTS_N,
    _build_estimation_problem,
    _build_truth,
    _header,
    _make_output_dir,
    _sample_full_trajectory,
    _thrust_indices,
)

warnings.filterwarnings("ignore", message="delta_grad == 0.0")
np.set_printoptions(precision=10, suppress=True)

RANGE_REFERENCE_POSITION_KM = np.array([8878.1366, -1200.0, 1800.0])
RANGE_FRACTIONS = (0.15, 0.40, 0.65, 0.90)
SIGMA_RANGE_KM = 0.02

RANGE_MODEL = InertialRangeMeasurementModel(RANGE_REFERENCE_POSITION_KM)


def _build_mixed_measurements(
    truth_problem,
    truth_decision_vector: np.ndarray,
    rng: np.random.Generator,
):
    placeholders: list[SampledMeasurement] = [
        SampledMeasurement(
            model=POSITION_MODEL,
            value=np.zeros(3, dtype=float),
            arc_name="arc0",
            sample_time_s=0.0,
            name="arc0_initial_position",
        )
    ]

    for arc_name, duration_s in zip(("arc0", "arc1"), ARC_DURATIONS_S, strict=True):
        for i, fraction in enumerate(MEASUREMENT_FRACTIONS):
            placeholders.append(
                SampledMeasurement(
                    model=POSITION_MODEL,
                    value=np.zeros(3, dtype=float),
                    arc_name=arc_name,
                    sample_time_s=fraction * duration_s,
                    name=f"{arc_name}_position_{i}",
                )
            )
        for i, fraction in enumerate(RANGE_FRACTIONS):
            placeholders.append(
                SampledMeasurement(
                    model=RANGE_MODEL,
                    value=np.zeros(1, dtype=float),
                    arc_name=arc_name,
                    sample_time_s=fraction * duration_s,
                    name=f"{arc_name}_range_{i}",
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


def _plot_results(
    out_path: Path,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    fit_times: np.ndarray,
    fit_positions: np.ndarray,
    fit_thrusts: np.ndarray,
    measurements: list[SampledMeasurement],
    fit_range_times_min: np.ndarray,
    fit_range_residuals_m: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("GEqOE mixed position + range reconstruction", fontsize=13)

    ax = axes[0]
    schedule_t = (
        np.array([0.0, ARC_DURATIONS_S[0], ARC_DURATIONS_S[0], sum(ARC_DURATIONS_S)])
        / 60.0
    )
    truth_schedule_u = np.array(
        [TRUTH_THRUSTS_N[0], TRUTH_THRUSTS_N[0], TRUTH_THRUSTS_N[1], TRUTH_THRUSTS_N[1]]
    )
    fit_schedule_u = np.array(
        [fit_thrusts[0], fit_thrusts[0], fit_thrusts[1], fit_thrusts[1]]
    )
    ax.plot(schedule_t, truth_schedule_u, label="truth", linewidth=2.5)
    ax.plot(schedule_t, fit_schedule_u, label="fit", linestyle="--", linewidth=2.5)
    ax.set_ylabel("Tangential thrust (N)")
    ax.set_title("Recovered maneuver coefficients")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
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
    if len(position_measurements) > 0:
        ax.scatter(
            position_measurements[:, 0],
            position_measurements[:, 1],
            marker="x",
            s=60,
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
    ax.set_title("Mixed inertial position and range geometry")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.plot(
        fit_range_times_min,
        fit_range_residuals_m,
        marker="o",
        linewidth=1.5,
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Range residual (m)")
    ax.set_title("Post-fit one-way range residuals")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(7)
    out_dir = _make_output_dir()

    truth_problem, x_truth, x0_truth = _build_truth()
    measurements = _build_mixed_measurements(truth_problem, x_truth, rng)
    num_position = sum(
        measurement.model.output_dimension == 3 for measurement in measurements
    )
    num_range = sum(
        measurement.model.output_dimension == 1 for measurement in measurements
    )

    _header("1. Generate Synthetic Mixed Measurements")
    print(f"Truth thrusts (N):        {TRUTH_THRUSTS_N}")
    print(f"Position sigma:           {SIGMA_POS_KM * 1.0e3:.1f} m")
    print(f"Range sigma:              {SIGMA_RANGE_KM * 1.0e3:.1f} m")
    print(f"Position samples:         {num_position}")
    print(f"Range samples:            {num_range}")
    print("Observation batch:        inertial position + inertial range")

    _header("2. Build The GEqOE Shooting Estimate")
    problem, x0 = _build_estimation_problem()
    bounds = problem.build_named_bounds(
        lower={"arc0.m": TRUTH_MASS_KG, "arc1.m": 100.0, "thrust.t_newtons": 0.0},
        upper={"arc0.m": TRUTH_MASS_KG, "arc1.m": 1000.0, "thrust.t_newtons": 2.5},
    )
    thrust_indices = _thrust_indices(problem)
    tracking_penalty = DecisionTrackingPenaltySpec(
        [
            DecisionTrackingTerm(
                selector="thrust.t_newtons",
                target=0.0,
                sigma=SIGMA_THRUST_N,
            )
        ]
    )
    print("Decision vector:          two 7-state arc nodes + two thrust coefficients")
    print(f"Continuity constraints:   {problem.continuity_size}")
    print(f"Initial thrust guess (N): {x0[thrust_indices]}")
    print("Initial mass handling:    fixed at the known truth mass for this toy model")

    _header("3. Solve The Mixed-Measurement Reconstruction")
    t0 = time.perf_counter()
    solve = problem.solve_measurement_fit(
        measurements,
        decision_vector0=x0,
        bounds=bounds,
        measurement_hessian_mode="quasi-newton",
        decision_tracking_penalty=tracking_penalty,
        options={"maxiter": 1500},
    )
    solve_s = time.perf_counter() - t0

    fit_eval = problem.evaluate_measurements(solve.x, measurements)
    fit_initial_state = solve.x[:7]
    fit_thrusts = solve.x[thrust_indices]
    fit_times, _, fit_positions = _sample_full_trajectory(fit_initial_state, fit_thrusts)
    truth_times, _, truth_positions = _sample_full_trajectory(
        x0_truth,
        np.array(TRUTH_THRUSTS_N, dtype=float),
    )

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
    range_times_min = np.array(
        [
            sample.absolute_time_s / 60.0
            for measurement, sample in zip(measurements, fit_eval.sample_results, strict=True)
            if measurement.model.output_dimension == 1
        ],
        dtype=float,
    )
    position_rms_m = np.sqrt(np.mean(np.square(position_residual_norms))) * 1.0e3
    range_rms_m = np.sqrt(np.mean(np.square(range_residuals_km))) * 1.0e3
    continuity_norm = np.linalg.norm(
        fit_eval.shooting_evaluation.continuity_residual
    )

    print(f"Solve success:            {solve.scipy_result.success}")
    print(f"Solver status:            {solve.scipy_result.status}")
    print(f"Solver message:           {solve.scipy_result.message}")
    print(f"Wall time:                {solve_s:.2f} s")
    print(f"Recovered thrusts (N):    {fit_thrusts}")
    print(f"True thrusts (N):         {TRUTH_THRUSTS_N}")
    print(f"Position-fit RMS:         {position_rms_m:.3f} m")
    print(f"Range-fit RMS:            {range_rms_m:.3f} m")
    print(f"Continuity residual norm: {continuity_norm:.3e}")
    print(f"Objective value:          {solve.objective:.6e}")

    out_path = out_dir / "mixed_position_range_fit.png"
    _plot_results(
        out_path,
        truth_times,
        truth_positions,
        fit_times,
        fit_positions,
        fit_thrusts,
        measurements,
        range_times_min,
        range_residuals_km * 1.0e3,
    )
    print(f"Saved figure:             {out_path}")


if __name__ == "__main__":
    main()
