#!/usr/bin/env python
"""Visual gallery for the GEqOE heyoka maneuvering stack.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_heyoka/maneuver_profile_gallery.py

This script generates three figures under ``examples/generated/geqoe_heyoka/``:
  1. constant tangential low-thrust orbit raising
  2. a smooth cubic-Hermite RTN thrust arc
  3. the current two-arc multiple-shooting prototype solve, plotted as a
     maneuver profile rather than only a terminal constraint report
"""

from __future__ import annotations

import math
import time
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
    CubicHermiteRTNThrustLaw,
    J2Perturbation,
    MultiArcShootingProblem,
    ShootingArc,
    ShootingSolveSpec,
    SmoothnessPenaltySpec,
    TerminalConstraintSpec,
    build_thrust_state_integrator,
    cart2geqoe,
    geqoe2cart,
)
from astrodyn_core.geqoe_taylor.integrator import propagate_grid

np.set_printoptions(precision=10, suppress=True)

R0 = np.array([7178.1366, 0.0, 0.0])
V0 = np.array([0.0, 5.269240572916780, 5.269240572916780])
M0 = 450.0


def _header(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def _make_output_dir() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "generated" / "geqoe_heyoka"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _make_constant_perturbation(
    thrust_r: float = 0.0,
    thrust_t: float = 0.0,
    thrust_n: float = 0.0,
    isp_s: float = 2200.0,
):
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[
            ContinuousThrustPerturbation(
                ConstantRTNThrustLaw(
                    thrust_r_newtons=thrust_r,
                    thrust_t_newtons=thrust_t,
                    thrust_n_newtons=thrust_n,
                    isp_s=isp_s,
                )
            )
        ],
    )


def _make_spline_perturbation(duration_s: float):
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[
            ContinuousThrustPerturbation(
                CubicHermiteRTNThrustLaw(
                    duration_s=duration_s,
                    thrust_r_newtons=(0.0, 0.0),
                    thrust_t_newtons=(0.04, 0.04),
                    thrust_n_newtons=(0.0, 0.0),
                    slope_r_newtons=(0.03, -0.03),
                    slope_t_newtons=(0.24, -0.24),
                    slope_n_newtons=(0.06, -0.06),
                    isp_s=2100.0,
                )
            )
        ],
    )


def _osculating_sma_km(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    r_mag = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    return 1.0 / ((2.0 / r_mag) - v2 / MU)


def _orbital_period_s() -> float:
    a = _osculating_sma_km(R0, V0)
    return 2.0 * math.pi * math.sqrt(a**3 / MU)


def _numeric_cubic_hermite(
    y0: float, y1: float, m0: float, m1: float, tau: np.ndarray
) -> np.ndarray:
    tau2 = tau * tau
    tau3 = tau2 * tau
    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


def _sample_law_rtn(law, times_s: np.ndarray) -> np.ndarray:
    if isinstance(law, ConstantRTNThrustLaw):
        profile = np.column_stack(
            [
                np.full_like(times_s, law.thrust_r_newtons),
                np.full_like(times_s, law.thrust_t_newtons),
                np.full_like(times_s, law.thrust_n_newtons),
            ]
        )
    elif isinstance(law, CubicHermiteRTNThrustLaw):
        tau = np.clip(times_s / law.duration_s, 0.0, 1.0)
        profile = np.column_stack(
            [
                _numeric_cubic_hermite(
                    law.thrust_r_newtons[0],
                    law.thrust_r_newtons[1],
                    law.slope_r_newtons[0],
                    law.slope_r_newtons[1],
                    tau,
                ),
                _numeric_cubic_hermite(
                    law.thrust_t_newtons[0],
                    law.thrust_t_newtons[1],
                    law.slope_t_newtons[0],
                    law.slope_t_newtons[1],
                    tau,
                ),
                _numeric_cubic_hermite(
                    law.thrust_n_newtons[0],
                    law.thrust_n_newtons[1],
                    law.slope_n_newtons[0],
                    law.slope_n_newtons[1],
                    tau,
                ),
            ]
        )
    else:
        raise TypeError(f"Unsupported thrust law type: {type(law).__name__}")

    return np.column_stack([profile, np.linalg.norm(profile, axis=1)])


def _sample_arc(perturbation, state0: np.ndarray, duration_s: float, n_samples: int = 500):
    ta, _ = build_thrust_state_integrator(
        perturbation,
        state0,
        tol=1.0e-15,
        compact_mode=True,
    )
    t_grid = np.linspace(0.0, duration_s, n_samples)
    states = propagate_grid(ta, t_grid)

    positions = np.zeros((n_samples, 3), dtype=float)
    velocities = np.zeros((n_samples, 3), dtype=float)
    sma = np.zeros(n_samples, dtype=float)
    for i, state in enumerate(states):
        r_vec, v_vec = geqoe2cart(state, MU, perturbation)
        positions[i] = r_vec
        velocities[i] = v_vec
        sma[i] = _osculating_sma_km(r_vec, v_vec)

    law = perturbation.non_conservative[0].law
    thrust_profile = _sample_law_rtn(law, t_grid)
    return t_grid, states, positions, velocities, sma, thrust_profile


def _plot_profile_case(
    title: str,
    out_path: Path,
    times_s: np.ndarray,
    thrust_profile: np.ndarray,
    states: np.ndarray,
    positions: np.ndarray,
    sma_km: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(times_s / 60.0, thrust_profile[:, 0], label="T_r")
    ax.plot(times_s / 60.0, thrust_profile[:, 1], label="T_t")
    ax.plot(times_s / 60.0, thrust_profile[:, 2], label="T_n")
    ax.plot(times_s / 60.0, thrust_profile[:, 3], label="|T|", linewidth=2.0, color="black")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Thrust (N)")
    ax.set_title("RTN thrust profile")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(times_s / 60.0, states[:, 6], color="tab:green")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Propagated mass")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(times_s / 60.0, sma_km, color="tab:red")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Osculating a (km)")
    ax.set_title("Energy / semimajor-axis response")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(positions[:, 0], positions[:, 1], color="tab:blue")
    ax.scatter(positions[0, 0], positions[0, 1], color="black", s=35, label="start")
    ax.scatter(positions[-1, 0], positions[-1, 1], color="tab:orange", s=35, label="end")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Inertial trajectory projection")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_constant_raise_case(out_dir: Path) -> None:
    _header("1. Constant Tangential Thrust Orbit Raise")
    period_s = _orbital_period_s()
    duration_s = 2.5 * period_s
    perturbation = _make_constant_perturbation(thrust_t=0.25, isp_s=2200.0)
    state0 = np.concatenate([cart2geqoe(R0, V0, MU, perturbation), [M0]])

    times_s, states, positions, _, sma_km, thrust_profile = _sample_arc(
        perturbation,
        state0,
        duration_s,
        n_samples=700,
    )
    out_path = out_dir / "constant_tangential_raise.png"
    _plot_profile_case(
        "GEqOE heyoka: constant tangential thrust",
        out_path,
        times_s,
        thrust_profile,
        states,
        positions,
        sma_km,
    )

    print(f"Duration: {duration_s / 3600.0:.2f} h")
    print(f"Mass used: {states[0, 6] - states[-1, 6]:.6f} kg")
    print(f"Delta a:   {sma_km[-1] - sma_km[0]:.6f} km")
    print(f"Saved:     {out_path}")


def _run_spline_case(out_dir: Path) -> None:
    _header("2. Smooth Cubic-Hermite RTN Arc")
    duration_s = 1800.0
    perturbation = _make_spline_perturbation(duration_s)
    state0 = np.concatenate([cart2geqoe(R0, V0, MU, perturbation), [M0]])

    times_s, states, positions, _, sma_km, thrust_profile = _sample_arc(
        perturbation,
        state0,
        duration_s,
        n_samples=500,
    )
    out_path = out_dir / "smooth_cubic_hermite_arc.png"
    _plot_profile_case(
        "GEqOE heyoka: smooth cubic-Hermite RTN thrust",
        out_path,
        times_s,
        thrust_profile,
        states,
        positions,
        sma_km,
    )

    print("Control law:")
    print("  thrust_t_newtons = (0.04, 0.04)")
    print("  slope_t_newtons  = (0.24, -0.24)")
    print("  slope_r_newtons  = (0.03, -0.03)")
    print("  slope_n_newtons  = (0.06, -0.06)")
    print(f"Peak |T|: {np.max(thrust_profile[:, 3]):.6f} N")
    print(f"Mass used: {states[0, 6] - states[-1, 6]:.6f} kg")
    print(f"Saved:     {out_path}")


def _sample_two_arc_solution(
    problem: MultiArcShootingProblem, solve_result
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arc_times = []
    arc_states = []
    arc_positions = []
    arc_sma = []
    time_offset = 0.0

    for arc_idx, arc_result in enumerate(solve_result.evaluation.arc_results):
        thrust_value = solve_result.x[
            problem.decision_index(f"arc{arc_idx}.thrust.t_newtons")
        ]
        perturbation = _make_constant_perturbation(thrust_t=thrust_value, isp_s=2100.0)
        local_t, states, positions, _, sma_km, thrust_profile = _sample_arc(
            perturbation,
            arc_result.initial_state,
            arc_result.duration_s,
            n_samples=240,
        )
        if arc_idx > 0:
            local_t = local_t[1:]
            states = states[1:, :]
            positions = positions[1:, :]
            sma_km = sma_km[1:]
            thrust_profile = thrust_profile[1:, :]

        arc_times.append(local_t + time_offset)
        arc_states.append(np.column_stack([states, thrust_profile]))
        arc_positions.append(positions)
        arc_sma.append(sma_km)
        time_offset += arc_result.duration_s

    times = np.concatenate(arc_times)
    states_and_thrust = np.vstack(arc_states)
    positions = np.vstack(arc_positions)
    sma_km = np.concatenate(arc_sma)
    return times, states_and_thrust, positions, sma_km


def _run_two_arc_shooting_case(out_dir: Path) -> None:
    _header("3. Two-Arc Multiple-Shooting Profile")
    duration0 = 600.0
    duration1 = 450.0
    pert0 = _make_constant_perturbation(thrust_t=0.18, isp_s=2100.0)
    pert1 = _make_constant_perturbation(thrust_t=0.22, isp_s=2100.0)
    x0 = np.concatenate([cart2geqoe(R0, V0, MU, pert0), [500.0]])

    print("Preparing nominal split trajectory...", flush=True)
    t0 = time.perf_counter()
    ta0, _ = build_thrust_state_integrator(pert0, x0, tol=1.0e-15, compact_mode=True)
    ta0.propagate_until(duration0)
    x1_nominal = ta0.state.copy()
    print(f"  nominal split ready in {time.perf_counter() - t0:.2f} s")

    print("Building and solving the shooting prototype...", flush=True)
    t0 = time.perf_counter()
    problem = MultiArcShootingProblem(
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
                initial_state=x1_nominal,
                duration_s=duration1,
                parameter_names=("thrust.t_newtons",),
                name="arc1",
            ),
        ],
        tol=1.0e-15,
        compact_mode=True,
    )
    x_nominal = problem.initial_guess()
    nominal_eval = problem.evaluate(x_nominal)
    target_final = nominal_eval.arc_results[-1].final_state.copy()

    lower_bounds = {
        f"arc0.{state_name}": x_nominal[problem.decision_index(f"arc0.{state_name}")]
        for state_name in problem.state_names
    }
    upper_bounds = dict(lower_bounds)
    lower_bounds["thrust.t_newtons"] = 0.05
    upper_bounds["thrust.t_newtons"] = 0.35
    bounds = problem.build_named_bounds(lower=lower_bounds, upper=upper_bounds)

    spec = ShootingSolveSpec(
        bounds=bounds,
        terminal_constraint=TerminalConstraintSpec.equality(
            target_final,
            output_indices=[3, 6],
        ),
        smoothness_penalty=SmoothnessPenaltySpec({"thrust.t_newtons": 5.0}),
        options={"maxiter": 100},
    )

    x_init = x_nominal.copy()
    x_init[problem.decision_index("arc0.thrust.t_newtons")] = 0.30
    x_init[problem.decision_index("arc1.thrust.t_newtons")] = 0.10
    solve = problem.solve(spec, decision_vector0=x_init)
    print(f"  prototype solve completed in {time.perf_counter() - t0:.2f} s")

    times_s, states_and_thrust, positions, sma_km = _sample_two_arc_solution(problem, solve)
    states = states_and_thrust[:, :7]
    thrust_profile = states_and_thrust[:, 7:]
    out_path = out_dir / "two_arc_shooting_profile.png"
    _plot_profile_case(
        "GEqOE heyoka: two-arc shooting prototype",
        out_path,
        times_s,
        thrust_profile,
        states,
        positions,
        sma_km,
    )

    print(f"Solve success: {solve.scipy_result.success}")
    print(f"Continuity residual norm: {np.linalg.norm(solve.continuity_residual):.3e}")
    print(
        "Solved tangential thrusts (N): "
        f"{solve.x[problem.decision_index('arc0.thrust.t_newtons')]:.6f}, "
        f"{solve.x[problem.decision_index('arc1.thrust.t_newtons')]:.6f}"
    )
    print(f"Saved: {out_path}")


def main() -> None:
    out_dir = _make_output_dir()
    print(f"Writing figures under: {out_dir}")
    _run_constant_raise_case(out_dir)
    _run_spline_case(out_dir)
    _run_two_arc_shooting_case(out_dir)


if __name__ == "__main__":
    main()
