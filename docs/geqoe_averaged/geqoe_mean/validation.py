"""Shared validation utilities for GEqOE averaged theory scripts."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU, RE
from .short_period import (
    evaluate_truncated_mean_rhs_pqm,
    isolated_short_period_expressions_for,
)

# Stage C: heyoka cfunc fast path
try:
    from .heyoka_compiled import (
        rk4_integrate_mean_compiled as _rk4_cfunc,
        adaptive_integrate_mean_compiled as _adaptive_cfunc,
    )
    _USE_CFUNC = True
except ImportError:
    _USE_CFUNC = False


def ensure_symbolic_cache(j_coeffs: dict[int, float]) -> None:
    """Pre-build the symbolic short-period cache for all degrees and variables."""
    for n in sorted(j_coeffs):
        for var in ("g", "Q", "Psi", "Omega", "M"):
            isolated_short_period_expressions_for(var, n)


def integrate_mean(
    state0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
    method: str = "auto",
    substeps: int = 8,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Integrate the mean GEqOE slow flow.

    Parameters
    ----------
    state0 : [nu, p1, p2, M, q1, q2] mean state
    t_eval : time grid [s]
    j_coeffs : {degree: Jn_value}
    method : "auto" (default), "adaptive", or "rk4"
        "auto" uses adaptive DOP853 with cfunc when available, else rk4 cfunc,
        else Python RK4 fallback.
    substeps : RK4 substeps per interval (only for method="rk4")
    rtol, atol : tolerances (only for method="adaptive")

    Returns
    -------
    states : (N, 6) array of mean states at each time
    """
    has_cfunc = _USE_CFUNC and set(j_coeffs.keys()) == {2, 3, 4, 5}

    if method == "auto":
        if has_cfunc:
            return _adaptive_cfunc(state0, t_eval, j_coeffs,
                                   re_val=re_val, mu_val=mu_val,
                                   rtol=rtol, atol=atol)
        # fall through to Python RK4

    elif method == "adaptive":
        if not has_cfunc:
            raise RuntimeError("Adaptive integrator requires heyoka cfunc "
                               "and j_coeffs keys {2,3,4,5}")
        return _adaptive_cfunc(state0, t_eval, j_coeffs,
                               re_val=re_val, mu_val=mu_val,
                               rtol=rtol, atol=atol)

    elif method == "rk4":
        if has_cfunc:
            return _rk4_cfunc(state0, t_eval, j_coeffs,
                              re_val=re_val, mu_val=mu_val, substeps=substeps)
        # fall through to Python RK4

    # Python RK4 fallback
    out = np.empty((len(t_eval), 6))
    out[0] = state0
    y = state0.copy()
    for i in range(len(t_eval) - 1):
        dt = (t_eval[i + 1] - t_eval[i]) / substeps
        for _ in range(substeps):
            k1 = evaluate_truncated_mean_rhs_pqm(y, j_coeffs, re_val=re_val, mu_val=mu_val)
            k2 = evaluate_truncated_mean_rhs_pqm(y + 0.5 * dt * k1, j_coeffs, re_val=re_val, mu_val=mu_val)
            k3 = evaluate_truncated_mean_rhs_pqm(y + 0.5 * dt * k2, j_coeffs, re_val=re_val, mu_val=mu_val)
            k4 = evaluate_truncated_mean_rhs_pqm(y + dt * k3, j_coeffs, re_val=re_val, mu_val=mu_val)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y[0] = state0[0]  # nu is constant at first order
        out[i + 1] = y
    return out


# Backwards-compatible alias
rk4_integrate_mean = integrate_mean


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Relative RMS error between two arrays."""
    resid = a - b
    denom = max(float(np.max(np.abs(b))), 1.0e-30)
    return float(np.sqrt(np.mean(resid * resid)) / denom)


def phase_error(a: np.ndarray, b: np.ndarray) -> float:
    """RMS phase error between two angle arrays [rad]."""
    diff = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(np.sqrt(np.mean(diff * diff)))


def compute_position_errors(
    truth: np.ndarray,
    test: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compute position error metrics between truth and test trajectories.

    Parameters
    ----------
    truth, test : (N, 3) position arrays [km]
    label : optional label

    Returns
    -------
    dict with pos_rms_km, pos_max_km, rad_rms_km, rad_max_km
    """
    diff = test - truth
    dist = np.linalg.norm(diff, axis=1)
    r_hat = truth / np.linalg.norm(truth, axis=1, keepdims=True)
    radial_err = np.sum(diff * r_hat, axis=1)
    valid = ~np.isnan(dist)
    if valid.sum() == 0:
        return {"label": label, "pos_rms_km": np.nan, "pos_max_km": np.nan,
                "rad_rms_km": np.nan, "rad_max_km": np.nan}
    d = dist[valid]
    re = radial_err[valid]
    return {
        "label": label,
        "pos_rms_km": float(np.sqrt(np.mean(d**2))),
        "pos_max_km": float(np.max(d)),
        "rad_rms_km": float(np.sqrt(np.mean(re**2))),
        "rad_max_km": float(np.max(np.abs(re))),
    }
