"""Heyoka Taylor integrator wrappers for the GEqOE system.

Provides builders for state-only and state+STM integrators, plus
propagation and dense output helpers.
"""

from __future__ import annotations

import numpy as np
import heyoka as hy

from astrodyn_core.geqoe_taylor.rhs import build_geqoe_system
from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel
from astrodyn_core.geqoe_taylor.constants import MU


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
    return par_values


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
    sys, _, par_map = build_geqoe_system(
        perturbation, use_par=True, time_origin=t0
    )

    ta = hy.taylor_adaptive(
        sys,
        state=list(ic),
        time=t0,
        pars=_build_par_values(perturbation, par_map),
        tol=tol,
        compact_mode=compact_mode,
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
    sys, _, par_map = build_geqoe_system(
        perturbation, use_par=True, time_origin=t0
    )

    vsys = hy.var_ode_sys(sys, hy.var_args.vars, order=1)

    # Augmented IC: state + flattened 6x6 identity matrix
    ic_list = list(ic)
    identity_flat = [1.0 if i == j else 0.0 for i in range(6) for j in range(6)]
    ic_aug = ic_list + identity_flat

    ta = hy.taylor_adaptive(
        vsys,
        state=ic_aug,
        time=t0,
        pars=_build_par_values(perturbation, par_map),
        tol=tol,
        compact_mode=compact_mode,
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


def extract_stm(state_aug: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the 6x6 STM from the 42-element augmented state.

    Args:
        state_aug: 42-element array from a variational integrator.

    Returns:
        (y, phi): 6-element state and 6x6 STM matrix.
    """
    y = state_aug[:6]
    phi = state_aug[6:].reshape(6, 6)
    return y, phi
