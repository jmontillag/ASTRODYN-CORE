"""Shared validation utilities for GEqOE averaged theory scripts."""

from __future__ import annotations

import numpy as np

from .constants import MU, RE
from .short_period import (
    evaluate_truncated_mean_rhs_pqm,
    isolated_short_period_expressions_for,
)


def ensure_symbolic_cache(j_coeffs: dict[int, float]) -> None:
    """Pre-build the symbolic short-period cache for all degrees and variables."""
    for n in sorted(j_coeffs):
        for var in ("g", "Q", "Psi", "Omega", "M"):
            isolated_short_period_expressions_for(var, n)


def rk4_integrate_mean(
    state0: np.ndarray,
    t_eval: np.ndarray,
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
    substeps: int = 8,
) -> np.ndarray:
    """RK4 integration of the mean GEqOE slow flow.

    Parameters
    ----------
    state0 : [nu, p1, p2, M, q1, q2] mean state
    t_eval : time grid [s]
    j_coeffs : {degree: Jn_value}
    substeps : RK4 substeps per interval

    Returns
    -------
    states : (N, 6) array of mean states at each time
    """
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
