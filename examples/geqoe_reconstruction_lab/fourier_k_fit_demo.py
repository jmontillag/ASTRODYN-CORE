#!/usr/bin/env python
"""Fit a direct Fourier-in-K thrust law from inertial-position samples.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_reconstruction_lab/fourier_k_fit_demo.py

This example intentionally uses the full GEqOE dynamics, not any orbit-averaged
or mean-element TFC reduction. A low-order tangential Fourier law in the
propagated generalized eccentric longitude K is estimated through the existing
measurement-aware shooting transcription.
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
    ContinuousThrustPerturbation,
    FourierKRTNThrustLaw,
    InertialPositionMeasurementModel,
    J2Perturbation,
    MeasurementObjectiveSpec,
    MultiArcShootingProblem,
    SampledMeasurement,
    ShootingArc,
    ShootingSolveSpec,
    build_thrust_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

np.set_printoptions(precision=10, suppress=True)
warnings.filterwarnings("ignore", message="delta_grad == 0.0")

R0 = np.array([7178.1366, 0.0, 0.0])
V0 = np.array([0.0, 5.269240572916780, 5.269240572916780])

DURATION_S = 5400.0
MASS_KG = 480.0
SIGMA_POS_KM = 0.005
MEASUREMENT_FRACTIONS = tuple(np.linspace(0.05, 1.0, 24))

TRUTH_COEFFS = {
    "bias_t_newtons": 0.18,
    "cosine_t_newtons": (0.04,),
    "sine_t_newtons": (0.0,),
}
INITIAL_GUESS = {
    "bias_t_newtons": 0.12,
    "cosine_t_newtons": (0.0,),
    "sine_t_newtons": (0.0,),
}

POSITION_MODEL = InertialPositionMeasurementModel()


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _make_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "generated" / "geqoe_reconstruction_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _make_perturbation(
    *,
    bias_t_newtons: float,
    cosine_t_newtons: tuple[float, ...],
    sine_t_newtons: tuple[float, ...],
    isp_s: float = 2100.0,
):
    law = FourierKRTNThrustLaw(
        order=1,
        bias_t_newtons=bias_t_newtons,
        cosine_t_newtons=cosine_t_newtons,
        sine_t_newtons=sine_t_newtons,
        isp_s=isp_s,
    )
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[ContinuousThrustPerturbation(law)],
    )


def _build_problem(
    initial_state: np.ndarray,
    *,
    bias_t_newtons: float,
    cosine_t_newtons: tuple[float, ...],
    sine_t_newtons: tuple[float, ...],
) -> MultiArcShootingProblem:
    return MultiArcShootingProblem(
        [
            ShootingArc(
                perturbation=_make_perturbation(
                    bias_t_newtons=bias_t_newtons,
                    cosine_t_newtons=cosine_t_newtons,
                    sine_t_newtons=sine_t_newtons,
                ),
                initial_state=initial_state,
                duration_s=DURATION_S,
                parameter_names=(
                    "thrust.t_bias_newtons",
                    "thrust.t_cos1_newtons",
                    "thrust.t_sin1_newtons",
                ),
                name="arc0",
            )
        ],
        tol=1.0e-15,
        compact_mode=True,
    )


def _measurement_placeholders() -> list[SampledMeasurement]:
    return [
        SampledMeasurement(
            model=POSITION_MODEL,
            value=np.zeros(3, dtype=float),
            arc_name="arc0",
            sample_time_s=fraction * DURATION_S,
            name=f"sample{i}",
        )
        for i, fraction in enumerate(MEASUREMENT_FRACTIONS)
    ]


def _build_truth() -> tuple[MultiArcShootingProblem, np.ndarray, np.ndarray]:
    perturbation = _make_perturbation(**TRUTH_COEFFS)
    initial_state = np.concatenate(
        [
            cart2geqoe(R0, V0, MU, perturbation),
            [MASS_KG],
        ]
    )
    problem = _build_problem(initial_state, **TRUTH_COEFFS)
    return problem, problem.initial_guess(), initial_state


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


def _set_initial_guess(problem: MultiArcShootingProblem) -> np.ndarray:
    x0 = problem.initial_guess()
    x0[problem.decision_index("arc0.thrust.t_bias_newtons")] = INITIAL_GUESS[
        "bias_t_newtons"
    ]
    x0[problem.decision_index("arc0.thrust.t_cos1_newtons")] = INITIAL_GUESS[
        "cosine_t_newtons"
    ][0]
    x0[problem.decision_index("arc0.thrust.t_sin1_newtons")] = INITIAL_GUESS[
        "sine_t_newtons"
    ][0]
    return x0


def _fixed_state_bounds(initial_state: np.ndarray) -> dict[str, float]:
    return {
        state_name: float(value)
        for state_name, value in zip(MultiArcShootingProblem.state_names, initial_state, strict=True)
    }


def _propagate_profile(
    initial_state: np.ndarray,
    *,
    bias_t_newtons: float,
    cosine_t_newtons: tuple[float, ...],
    sine_t_newtons: tuple[float, ...],
    n_samples: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perturbation = _make_perturbation(
        bias_t_newtons=bias_t_newtons,
        cosine_t_newtons=cosine_t_newtons,
        sine_t_newtons=sine_t_newtons,
    )
    ta, _ = build_thrust_state_integrator(
        perturbation,
        initial_state,
        tol=1.0e-15,
        compact_mode=True,
    )
    time_grid = np.linspace(0.0, DURATION_S, n_samples)
    states = propagate_grid(ta, time_grid)
    positions = np.zeros((n_samples, 3), dtype=float)
    thrust_t = np.zeros(n_samples, dtype=float)
    for i, state in enumerate(states):
        positions[i], _ = geqoe2cart(state, MU, perturbation)
        K = float(state[3])
        thrust_t[i] = (
            bias_t_newtons
            + cosine_t_newtons[0] * np.cos(K)
            + sine_t_newtons[0] * np.sin(K)
        )
    return time_grid, positions, thrust_t


def _plot_results(
    out_path: Path,
    measurements: list[SampledMeasurement],
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    fit_positions: np.ndarray,
    truth_thrust_t: np.ndarray,
    fit_thrust_t: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    sample_times = np.array([m.sample_time_s for m in measurements], dtype=float)
    observed_positions = np.vstack([m.value for m in measurements])

    axes[0].plot(truth_times, truth_positions[:, 0], label="truth x")
    axes[0].plot(truth_times, fit_positions[:, 0], "--", label="fit x")
    axes[0].scatter(sample_times, observed_positions[:, 0], s=18, label="samples x")
    axes[0].set_ylabel("x [km]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(truth_times, truth_thrust_t, label="truth T_t")
    axes[1].plot(truth_times, fit_thrust_t, "--", label="fit T_t")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("tangential thrust [N]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    _header("GEqOE Fourier-in-K Full-Dynamics Fit Demo")
    rng = np.random.default_rng(12)
    out_dir = _make_output_dir()

    truth_problem, truth_x, initial_state = _build_truth()
    measurements = _build_measurements(truth_problem, truth_x, rng)

    estimation_problem = _build_problem(initial_state, **TRUTH_COEFFS)
    x0 = _set_initial_guess(estimation_problem)

    fixed_bounds = _fixed_state_bounds(initial_state)
    bounds = estimation_problem.build_named_bounds(
        lower={
            **fixed_bounds,
            "thrust.t_bias_newtons": 0.05,
            "thrust.t_cos1_newtons": -0.10,
            "thrust.t_sin1_newtons": -0.10,
        },
        upper={
            **fixed_bounds,
            "thrust.t_bias_newtons": 0.30,
            "thrust.t_cos1_newtons": 0.10,
            "thrust.t_sin1_newtons": 0.10,
        },
    )

    t0 = time.perf_counter()
    solve = estimation_problem.solve_measurement_fit(
        measurements,
        decision_vector0=x0,
        bounds=bounds,
        measurement_hessian_mode="gauss-newton",
        options={"maxiter": 300},
    )
    elapsed = time.perf_counter() - t0

    spec = ShootingSolveSpec(
        bounds=bounds,
        measurement_objective=MeasurementObjectiveSpec(
            measurements=measurements,
            hessian_mode="gauss-newton",
        ),
    )
    covariance = estimation_problem.estimate_covariance(solve.x, spec)

    measurement_eval = estimation_problem.evaluate_measurements(solve.x, measurements)
    rms_km = float(np.sqrt(np.mean(measurement_eval.residual**2)) * SIGMA_POS_KM)

    idx_bias = estimation_problem.decision_index("arc0.thrust.t_bias_newtons")
    idx_cos1 = estimation_problem.decision_index("arc0.thrust.t_cos1_newtons")
    idx_sin1 = estimation_problem.decision_index("arc0.thrust.t_sin1_newtons")

    truth_times, truth_positions, truth_thrust_t = _propagate_profile(
        initial_state,
        **TRUTH_COEFFS,
    )
    _, fit_positions, fit_thrust_t = _propagate_profile(
        initial_state,
        bias_t_newtons=float(solve.x[idx_bias]),
        cosine_t_newtons=(float(solve.x[idx_cos1]),),
        sine_t_newtons=(float(solve.x[idx_sin1]),),
    )

    plot_path = out_dir / "fourier_k_fit_demo.png"
    _plot_results(
        plot_path,
        measurements,
        truth_times,
        truth_positions,
        fit_positions,
        truth_thrust_t,
        fit_thrust_t,
    )

    print(f"Solve success: {solve.scipy_result.success}")
    print(f"Solve status: {solve.scipy_result.status}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Weighted residual RMS: {np.sqrt(np.mean(measurement_eval.residual**2)):.3f}")
    print(f"Position RMS: {rms_km * 1000.0:.3f} m")
    print(
        "Recovered tangential coefficients [N]: "
        f"bias={solve.x[idx_bias]:.6f}, cos1={solve.x[idx_cos1]:.6f}, sin1={solve.x[idx_sin1]:.6f}"
    )
    print(
        "Coefficient 1-sigma [N]: "
        f"bias={covariance.standard_deviations[idx_bias]:.6f}, "
        f"cos1={covariance.standard_deviations[idx_cos1]:.6f}, "
        f"sin1={covariance.standard_deviations[idx_sin1]:.6f}"
    )
    print(
        "Truth tangential coefficients [N]: "
        f"bias={TRUTH_COEFFS['bias_t_newtons']:.6f}, "
        f"cos1={TRUTH_COEFFS['cosine_t_newtons'][0]:.6f}, "
        f"sin1={TRUTH_COEFFS['sine_t_newtons'][0]:.6f}"
    )
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
