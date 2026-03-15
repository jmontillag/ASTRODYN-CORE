"""GEqOE Taylor continuous-thrust multiple-shooting prototype demo.

Run:
  conda run -n astrodyn-core-env python examples/geqoe_heyoka/geqoe_taylor_shooting_demo.py

This example builds a two-arc continuous-thrust problem, transcribes it as a
GEqOE Taylor multiple-shooting solve, and runs the current SciPy-based
minimum-propellant prototype with a cross-arc smoothness penalty.

The first multiple-shooting build compiles two mass-augmented variational
systems, so it can take a few seconds before the solve begins.
"""

from __future__ import annotations

import time

import numpy as np

from astrodyn_core.geqoe_taylor import (
    MU,
    CompositePerturbation,
    ConstantRTNThrustLaw,
    ContinuousThrustPerturbation,
    J2Perturbation,
    MultiArcShootingProblem,
    ShootingArc,
    ShootingSolveSpec,
    SmoothnessPenaltySpec,
    TerminalConstraintSpec,
    build_thrust_state_integrator,
    cart2geqoe,
)

np.set_printoptions(precision=10, suppress=True)

R0 = np.array([7178.1366, 0.0, 0.0])
V0 = np.array([0.0, 5.269240572916780, 5.269240572916780])


def make_perturbation(thrust_t_newtons: float) -> CompositePerturbation:
    return CompositePerturbation(
        conservative=[J2Perturbation()],
        non_conservative=[
            ContinuousThrustPerturbation(
                ConstantRTNThrustLaw(thrust_t_newtons=thrust_t_newtons, isp_s=2100.0)
            )
        ],
    )


def main() -> None:
    print("=" * 72)
    print("GEqOE Taylor Multiple-Shooting Prototype")
    print("=" * 72)

    duration0 = 600.0
    duration1 = 450.0
    pert0 = make_perturbation(0.18)
    pert1 = make_perturbation(0.22)

    print("\nPreparing nominal split trajectory...", flush=True)
    x0 = np.concatenate([cart2geqoe(R0, V0, MU, pert0), [500.0]])
    t0 = time.perf_counter()
    ta0, _ = build_thrust_state_integrator(pert0, x0, tol=1e-15, compact_mode=True)
    ta0.propagate_until(duration0)
    x1_nominal = ta0.state.copy()
    print(f"  nominal arc prepared in {time.perf_counter() - t0:.2f} s", flush=True)

    print(
        "\nBuilding multiple-shooting problem "
        "(compiles two sensitivity systems)...",
        flush=True,
    )
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
        tol=1e-15,
        compact_mode=True,
    )
    print(f"  shooting problem built in {time.perf_counter() - t0:.2f} s", flush=True)

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

    obj_init, _ = problem.minimum_propellant_objective(x_init)
    smooth_init, _, _ = problem.control_smoothness_objective(
        x_init,
        spec.smoothness_penalty,
    )

    print("\nNominal arc thrusts (N):")
    print(f"  arc0: {x_nominal[problem.decision_index('arc0.thrust.t_newtons')]:.6f}")
    print(f"  arc1: {x_nominal[problem.decision_index('arc1.thrust.t_newtons')]:.6f}")

    print("\nInitial guess thrusts (N):")
    print(f"  arc0: {x_init[problem.decision_index('arc0.thrust.t_newtons')]:.6f}")
    print(f"  arc1: {x_init[problem.decision_index('arc1.thrust.t_newtons')]:.6f}")
    print(f"  propellant objective only: {obj_init:.12f} kg")
    print(f"  smoothness penalty only:   {smooth_init:.12f}")

    print("\nSolving prototype NLP...", flush=True)
    t0 = time.perf_counter()
    solve = problem.solve(spec, decision_vector0=x_init)
    print(f"  solve completed in {time.perf_counter() - t0:.2f} s", flush=True)

    idx_t0 = problem.decision_index("arc0.thrust.t_newtons")
    idx_t1 = problem.decision_index("arc1.thrust.t_newtons")
    obj_final, _ = problem.minimum_propellant_objective(
        solve.x,
        evaluation=solve.evaluation,
    )
    smooth_final, _, _ = problem.control_smoothness_objective(
        solve.x,
        spec.smoothness_penalty,
    )

    print("\nSolve status:")
    print(f"  success: {solve.scipy_result.success}")
    print(f"  message: {solve.scipy_result.message}")
    print(f"  iterations: {solve.scipy_result.niter}")

    print("\nSolved arc thrusts (N):")
    print(f"  arc0: {solve.x[idx_t0]:.9f}")
    print(f"  arc1: {solve.x[idx_t1]:.9f}")
    print(f"  |delta|: {abs(solve.x[idx_t0] - solve.x[idx_t1]):.9f}")

    print("\nConstraint summary:")
    print(f"  continuity residual norm: {np.linalg.norm(solve.continuity_residual):.3e}")
    if solve.terminal_outputs is not None:
        print(f"  terminal outputs [K, m]: {solve.terminal_outputs}")
    if solve.terminal_violation is not None:
        print(f"  terminal violation max: {np.max(solve.terminal_violation):.3e}")

    print("\nObjective summary:")
    print(f"  propellant objective only: {obj_final:.12f} kg")
    print(f"  smoothness penalty only:   {smooth_final:.12f}")
    print(f"  combined objective:        {solve.objective:.12f}")

    print("\nFinal arc-end state [nu, p1, p2, K, q1, q2, m]:")
    print(solve.evaluation.arc_results[-1].final_state)


if __name__ == "__main__":
    main()
