#!/usr/bin/env python
"""Synthetic maneuver reconstruction from intermediate inertial positions.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_reconstruction_lab/position_fit_demo.py

This is a direct-shooting continuous-thrust analogue of the maneuver-effort
estimation idea in the planning notes:
  1. generate a maneuvered truth trajectory,
  2. sample noisy inertial positions along the trajectory,
  3. keep the initial state and the control coefficients free,
  4. penalize both the initial-state offset and the maneuver effort, and
  5. solve for the smallest maneuver explanation consistent with the data.

The control is piecewise-constant tangential thrust over two arcs. This keeps
the example compact and runnable while still exposing the essential pieces of
the new GEqOE heyoka maneuvering stack: propagated mass, smooth parameterized
control, and uncertainty-weighted fitting.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrodyn_core.geqoe_taylor import (
    MU,
    CompositePerturbation,
    ConstantRTNThrustLaw,
    ContinuousThrustPerturbation,
    J2Perturbation,
    build_thrust_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

np.set_printoptions(precision=10, suppress=True)

R0 = np.array([7178.1366, 0.0, 0.0])
V0 = np.array([0.0, 5.269240572916780, 5.269240572916780])

TRUTH_MASS_KG = 480.0
ARC_DURATIONS_S = (600.0, 720.0)
TRUTH_THRUSTS_N = (2.0, 0.8)

SIGMA_POS_KM = 0.05
SIGMA_PRIOR_POS_KM = 0.20
SIGMA_PRIOR_VEL_KMPS = 2.0e-4
SIGMA_PRIOR_MASS_KG = 0.25
SIGMA_THRUST_N = 1.0

MEASUREMENT_ORDER = ("arc0_mid", "arc0_end", "arc1_mid", "arc1_end")


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


def _state_to_rv(state: np.ndarray) -> np.ndarray:
    perturbation = _make_constant_perturbation(0.0)
    r_vec, v_vec = geqoe2cart(state, MU, perturbation)
    return np.concatenate([r_vec, v_vec])


def _predict_measurements(
    initial_state: np.ndarray,
    thrusts_n: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    thrust0, thrust1 = thrusts_n

    x0_mid = _propagate_to_time(initial_state, thrust0, 0.5 * ARC_DURATIONS_S[0])
    x1 = _propagate_to_time(initial_state, thrust0, ARC_DURATIONS_S[0])
    x1_mid = _propagate_to_time(x1, thrust1, 0.5 * ARC_DURATIONS_S[1])
    x2 = _propagate_to_time(x1, thrust1, ARC_DURATIONS_S[1])

    predictions = {
        "arc0_mid": geqoe2cart(x0_mid, MU, _make_constant_perturbation(thrust0))[0],
        "arc0_end": geqoe2cart(x1, MU, _make_constant_perturbation(thrust0))[0],
        "arc1_mid": geqoe2cart(x1_mid, MU, _make_constant_perturbation(thrust1))[0],
        "arc1_end": geqoe2cart(x2, MU, _make_constant_perturbation(thrust1))[0],
    }
    return predictions, x2


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


def _build_truth():
    x0_truth = np.concatenate(
        [cart2geqoe(R0, V0, MU, _make_constant_perturbation(TRUTH_THRUSTS_N[0])), [TRUTH_MASS_KG]]
    )
    truth_measurements, x2_truth = _predict_measurements(
        x0_truth,
        np.array(TRUTH_THRUSTS_N, dtype=float),
    )
    return x0_truth, truth_measurements, x2_truth


def _build_initial_guess():
    r0_prior = R0 + np.array([0.12, -0.08, 0.05])
    v0_prior = V0 + np.array([1.8e-4, -1.2e-4, 0.9e-4])
    x0_prior = np.concatenate(
        [
            cart2geqoe(r0_prior, v0_prior, MU, _make_constant_perturbation(0.0)),
            [TRUTH_MASS_KG + 0.35],
        ]
    )
    thrust_guess = np.array([0.2, 0.2], dtype=float)
    prior_rv_mass = np.concatenate([r0_prior, v0_prior, [TRUTH_MASS_KG + 0.35]])
    return x0_prior, thrust_guess, prior_rv_mass


def _residual_vector(
    z: np.ndarray,
    measurements: dict[str, np.ndarray],
    prior_rv_mass: np.ndarray,
) -> np.ndarray:
    initial_state = z[:7]
    thrusts = z[7:]
    predictions, _ = _predict_measurements(initial_state, thrusts)

    residuals = []
    for name in MEASUREMENT_ORDER:
        residuals.append((predictions[name] - measurements[name]) / SIGMA_POS_KM)

    rv0 = _state_to_rv(initial_state)
    residuals.append((rv0[:3] - prior_rv_mass[:3]) / SIGMA_PRIOR_POS_KM)
    residuals.append((rv0[3:] - prior_rv_mass[3:6]) / SIGMA_PRIOR_VEL_KMPS)
    residuals.append(np.array([(initial_state[6] - prior_rv_mass[6]) / SIGMA_PRIOR_MASS_KG]))
    residuals.append(thrusts / SIGMA_THRUST_N)
    return np.concatenate(residuals)


def _plot_results(
    out_path: Path,
    truth_times: np.ndarray,
    truth_states: np.ndarray,
    truth_positions: np.ndarray,
    fit_times: np.ndarray,
    fit_states: np.ndarray,
    fit_positions: np.ndarray,
    measurements: dict[str, np.ndarray],
    fit_thrusts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("GEqOE direct-shooting position-fit experiment", fontsize=13)

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
    colors = {
        "arc0_mid": "black",
        "arc0_end": "tab:purple",
        "arc1_mid": "tab:green",
        "arc1_end": "tab:red",
    }
    labels = {
        "arc0_mid": "measurement @ arc 0 mid",
        "arc0_end": "measurement @ arc 0 end",
        "arc1_mid": "measurement @ arc 1 mid",
        "arc1_end": "measurement @ final time",
    }
    for name in MEASUREMENT_ORDER:
        ax.scatter(
            measurements[name][0],
            measurements[name][1],
            marker="x",
            s=80,
            color=colors[name],
            label=labels[name],
        )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Inertial position fit")
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

    _header("1. Generate Synthetic Truth And Measurements")
    x0_truth, truth_measurements, _ = _build_truth()
    measurements = {
        name: truth_measurements[name] + rng.normal(0.0, SIGMA_POS_KM, size=3)
        for name in MEASUREMENT_ORDER
    }
    print(f"Truth thrusts (N): {TRUTH_THRUSTS_N}")
    print(f"Measurement sigma: {SIGMA_POS_KM * 1.0e3:.1f} m")
    print("Measured positions: arc0 mid/end and arc1 mid/end.")

    _header("2. Build The Direct-Shooting Estimate")
    x0_prior, thrust_guess, prior_rv_mass = _build_initial_guess()
    z0 = np.concatenate([x0_prior, thrust_guess])
    print("Decision vector: 7-state initial condition + 2 thrust coefficients")
    print(f"Initial thrust guess (N): {thrust_guess}")

    lower = np.full_like(z0, -np.inf)
    upper = np.full_like(z0, np.inf)
    lower[6] = 100.0
    upper[6] = 1000.0
    lower[7:] = 0.0
    upper[7:] = 2.5

    _header("3. Solve The Position-Fit Reconstruction")
    t0 = time.perf_counter()
    result = least_squares(
        _residual_vector,
        z0,
        bounds=(lower, upper),
        args=(measurements, prior_rv_mass),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=200,
        verbose=0,
    )
    solve_s = time.perf_counter() - t0

    fit_initial_state = result.x[:7]
    fit_thrusts = result.x[7:]
    fit_measurements, _ = _predict_measurements(fit_initial_state, fit_thrusts)
    fit_times, fit_states, fit_positions = _sample_full_trajectory(fit_initial_state, fit_thrusts)
    truth_times, truth_states, truth_positions = _sample_full_trajectory(
        x0_truth,
        np.array(TRUTH_THRUSTS_N, dtype=float),
    )

    residual_norms = [
        np.linalg.norm(fit_measurements[name] - measurements[name]) for name in MEASUREMENT_ORDER
    ]
    rms_m = np.sqrt(np.mean(np.square(residual_norms))) * 1.0e3

    print(f"Solve success:            {result.success}")
    print(f"Solver status:            {result.status}")
    print(f"Solver message:           {result.message}")
    print(f"Function evaluations:     {result.nfev}")
    print(f"Wall time:                {solve_s:.2f} s")
    print(f"Recovered thrusts (N):    {fit_thrusts}")
    print(f"True thrusts (N):         {TRUTH_THRUSTS_N}")
    print(f"Position-fit RMS:         {rms_m:.3f} m")
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
