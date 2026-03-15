"""Heyoka Taylor integrator wrappers for the GEqOE system.

Provides builders for state-only and state+STM integrators, plus
propagation and dense output helpers.
"""

from __future__ import annotations

import heyoka as hy
import numpy as np

from astrodyn_core.geqoe_taylor.constants import MU
from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel
from astrodyn_core.geqoe_taylor.rhs import build_geqoe_mass_system, build_geqoe_system


def _parameter_defaults(perturbation) -> dict[str, float]:
    defaults = getattr(perturbation, "parameter_defaults", None)
    if defaults is None:
        return {}
    return dict(defaults())


def _build_par_values(perturbation, par_map):
    """Build parameter values array from par_map."""
    if not par_map:
        return []
    n_pars = max(par_map.values()) + 1
    par_values = [0.0] * n_pars
    if "mu" in par_map:
        par_values[par_map["mu"]] = getattr(perturbation, "mu", MU)
    if "A_J2" in par_map:
        if not hasattr(perturbation, "A"):
            raise AttributeError(
                "J2 fast-path perturbations must define an 'A' coefficient."
            )
        par_values[par_map["A_J2"]] = perturbation.A
    for name, value in _parameter_defaults(perturbation).items():
        if name in par_map:
            par_values[par_map[name]] = float(value)
    return par_values


def _build_taylor_adaptive(sys, ic, perturbation, par_map, t0, tol, compact_mode):
    return hy.taylor_adaptive(
        sys,
        state=list(ic),
        time=t0,
        pars=_build_par_values(perturbation, par_map),
        tol=tol,
        compact_mode=compact_mode,
    )


def _build_variational_integrator(
    sys,
    ic,
    state_dim,
    perturbation,
    par_map,
    t0,
    tol,
    compact_mode,
    with_params=False,
):
    vargs = hy.var_args.vars
    if with_params and par_map:
        vargs = vargs | hy.var_args.params

    vsys = hy.var_ode_sys(sys, vargs, order=1)
    ic_list = list(ic)
    jac_cols = state_dim + (len(par_map) if with_params else 0)
    jacobian_flat = []
    for i in range(state_dim):
        for j in range(jac_cols):
            jacobian_flat.append(1.0 if j < state_dim and i == j else 0.0)
    ic_aug = ic_list + jacobian_flat
    return hy.taylor_adaptive(
        vsys,
        state=ic_aug,
        time=t0,
        pars=_build_par_values(perturbation, par_map),
        tol=tol,
        compact_mode=compact_mode,
    )


def build_state_integrator(
    perturbation: PerturbationModel,
    ic: np.ndarray | list,
    t0: float = 0.0,
    tol: float = 1e-15,
    compact_mode: bool = False,
) -> tuple[hy.taylor_adaptive, dict]:
    """Build a state-only GEqOE Taylor integrator (6 DOF).

    Args:
        perturbation: PerturbationModel instance.
        ic: initial conditions [nu, p1, p2, K, q1, q2].
        t0: initial time in seconds. Time-dependent perturbations interpret
            this as the absolute integrator time corresponding to the epoch
            of the initial state.
        tol: integrator tolerance.
        compact_mode: use compact mode for expression compilation.

    Returns:
        (ta, par_map): integrator and parameter index mapping.
    """
    if len(ic) != 6:
        raise ValueError("build_state_integrator() expects a 6-element GEqOE state.")

    sys, _, par_map = build_geqoe_system(
        perturbation, use_par=True, time_origin=t0
    )

    ta = _build_taylor_adaptive(
        sys, ic, perturbation, par_map, t0, tol, compact_mode
    )
    return ta, par_map


def build_stm_integrator(
    perturbation: PerturbationModel,
    ic: np.ndarray | list,
    t0: float = 0.0,
    tol: float = 1e-15,
    compact_mode: bool = True,
) -> tuple[hy.taylor_adaptive, dict]:
    """Build a GEqOE integrator with automatic 1st-order variational equations (STM).

    The augmented system has 6 + 36 = 42 state variables.

    Args:
        perturbation: PerturbationModel instance.
        ic: initial conditions [nu, p1, p2, K, q1, q2] (6 elements only).
        t0: initial time in seconds. Time-dependent perturbations interpret
            this as the absolute integrator time corresponding to the epoch
            of the initial state.
        tol: integrator tolerance.
        compact_mode: use compact mode (recommended for 42-DOF).

    Returns:
        (ta, par_map): integrator and parameter index mapping.
    """
    if len(ic) != 6:
        raise ValueError("build_stm_integrator() expects a 6-element GEqOE state.")

    sys, _, par_map = build_geqoe_system(
        perturbation, use_par=True, time_origin=t0
    )

    ta = _build_variational_integrator(
        sys,
        ic,
        6,
        perturbation,
        par_map,
        t0,
        tol,
        compact_mode,
        with_params=False,
    )
    return ta, par_map


def build_thrust_state_integrator(
    perturbation: PerturbationModel,
    ic: np.ndarray | list,
    t0: float = 0.0,
    tol: float = 1e-15,
    compact_mode: bool = True,
) -> tuple[hy.taylor_adaptive, dict]:
    """Build a 7-state GEqOE + mass integrator for continuous-thrust propagation."""
    if len(ic) != 7:
        raise ValueError(
            "build_thrust_state_integrator() expects a 7-element "
            "GEqOE state [nu, p1, p2, K, q1, q2, m]."
        )

    sys, _, par_map = build_geqoe_mass_system(
        perturbation, use_par=True, time_origin=t0
    )
    ta = _build_taylor_adaptive(
        sys, ic, perturbation, par_map, t0, tol, compact_mode
    )
    return ta, par_map


def build_thrust_stm_integrator(
    perturbation: PerturbationModel,
    ic: np.ndarray | list,
    t0: float = 0.0,
    tol: float = 1e-15,
    compact_mode: bool = True,
) -> tuple[hy.taylor_adaptive, dict]:
    """Build a 7-state GEqOE + mass integrator with first-order variational equations."""
    if len(ic) != 7:
        raise ValueError(
            "build_thrust_stm_integrator() expects a 7-element "
            "GEqOE state [nu, p1, p2, K, q1, q2, m]."
        )

    sys, _, par_map = build_geqoe_mass_system(
        perturbation, use_par=True, time_origin=t0
    )
    ta = _build_variational_integrator(
        sys,
        ic,
        7,
        perturbation,
        par_map,
        t0,
        tol,
        compact_mode,
        with_params=False,
    )
    return ta, par_map


def build_thrust_sensitivity_integrator(
    perturbation: PerturbationModel,
    ic: np.ndarray | list,
    t0: float = 0.0,
    tol: float = 1e-15,
    compact_mode: bool = True,
) -> tuple[hy.taylor_adaptive, dict]:
    """Build a 7-state thrust integrator with sensitivities wrt state and params."""
    if len(ic) != 7:
        raise ValueError(
            "build_thrust_sensitivity_integrator() expects a 7-element "
            "GEqOE state [nu, p1, p2, K, q1, q2, m]."
        )

    sys, _, par_map = build_geqoe_mass_system(
        perturbation, use_par=True, time_origin=t0
    )
    ta = _build_variational_integrator(
        sys,
        ic,
        7,
        perturbation,
        par_map,
        t0,
        tol,
        compact_mode,
        with_params=True,
    )
    return ta, par_map


def propagate(
    ta: hy.taylor_adaptive,
    t_final: float,
    max_delta_t: float | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    """Step-by-step propagation to t_final.

    Args:
        ta: heyoka integrator (modified in place).
        t_final: target time in seconds.
        max_delta_t: maximum step size (None = adaptive only, clamped to t_final).

    Returns:
        (times, states): lists of step boundary times and states.
    """
    times = [ta.time]
    states = [ta.state.copy()]

    while ta.time < t_final:
        remaining = t_final - ta.time
        if max_delta_t is not None:
            step_limit = min(max_delta_t, remaining)
        else:
            step_limit = remaining

        ta.step(max_delta_t=step_limit)
        times.append(ta.time)
        states.append(ta.state.copy())

    return times, states


def propagate_grid(
    ta: hy.taylor_adaptive,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Propagate and return states at specified time grid points.

    Uses heyoka's built-in propagate_grid with dense output.

    Args:
        ta: heyoka integrator (modified in place).
        t_grid: array of output times (must be sorted, first > ta.time or == ta.time).

    Returns:
        states: (len(t_grid), n_vars) array of states at grid points.
    """
    result = ta.propagate_grid(t_grid)
    # result is a tuple: (outcome, min_h, max_h, n_steps, cb_result, state_array)
    return result[-1]


def extract_stm(
    state_aug: np.ndarray,
    state_dim: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the state and square STM from an augmented variational state.

    Args:
        state_aug: augmented state from a variational integrator.
        state_dim: physical state dimension (6 for the legacy GEqOE state,
            7 for the mass-augmented thrust state).

    Returns:
        (y, phi): state vector and ``state_dim x state_dim`` STM matrix.
    """
    y = state_aug[:state_dim]
    phi = state_aug[state_dim:].reshape(state_dim, state_dim)
    return y, phi


def parameter_names_from_map(par_map: dict[str, int]) -> list[str]:
    """Return runtime parameter names ordered by heyoka parameter index."""
    return [name for name, _ in sorted(par_map.items(), key=lambda item: item[1])]


def extract_variational_matrices(
    state_aug: np.ndarray,
    state_dim: int,
    par_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract the propagated state, STM, and parameter sensitivities.

    The augmented Jacobian is returned in the same column order used by heyoka:
    first the state variables, then the runtime parameters sorted by `par[i]`.
    """
    param_names = parameter_names_from_map(par_map or {})
    n_param = len(param_names)
    expected_size = state_dim + state_dim * (state_dim + n_param)
    if len(state_aug) != expected_size:
        raise ValueError(
            f"Expected augmented state of length {expected_size}, got {len(state_aug)}."
        )

    y = state_aug[:state_dim]
    jac = state_aug[state_dim:].reshape(state_dim, state_dim + n_param)
    phi_x = jac[:, :state_dim]
    phi_p = jac[:, state_dim:]
    return y, phi_x, phi_p, param_names


def extract_endpoint_jacobian(
    state_aug: np.ndarray,
    state_dim: int,
    par_map: dict[str, int] | None = None,
    output_indices: list[int] | np.ndarray | None = None,
    parameter_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract endpoint Jacobian blocks for selected outputs and parameters.

    Returns the Jacobian of the selected propagated outputs with respect to the
    initial augmented state and the requested runtime parameters.
    """
    _, phi_x, phi_p, all_param_names = extract_variational_matrices(
        state_aug, state_dim=state_dim, par_map=par_map
    )

    if output_indices is None:
        row_idx = np.arange(state_dim)
    else:
        row_idx = np.asarray(output_indices, dtype=int)

    if parameter_names is None:
        col_idx = np.arange(len(all_param_names))
        selected_names = list(all_param_names)
    else:
        index_map = {name: i for i, name in enumerate(all_param_names)}
        try:
            col_idx = np.asarray([index_map[name] for name in parameter_names], dtype=int)
        except KeyError as exc:
            raise KeyError(f"Unknown runtime parameter name: {exc.args[0]!r}") from exc
        selected_names = list(parameter_names)

    jac_x = phi_x[row_idx, :]
    jac_p = phi_p[np.ix_(row_idx, col_idx)]
    return jac_x, jac_p, selected_names
