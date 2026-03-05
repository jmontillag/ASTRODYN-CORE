"""Helper utilities for the GEqOE Taylor propagator."""

from __future__ import annotations

import numpy as np


def solve_kepler_gen(
    L: float | np.ndarray,
    p1: float | np.ndarray,
    p2: float | np.ndarray,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> float | np.ndarray:
    """Solve the generalized Kepler equation L = K + p1*cos(K) - p2*sin(K) for K.

    Newton-Raphson iteration.  Eq. 30 of Baù et al. (2021).

    Args:
        L: generalized mean longitude(s).
        p1, p2: eccentricity-like parameters.
        tol: convergence tolerance.
        max_iter: maximum iterations.

    Returns:
        K: generalized eccentric longitude(s).
    """
    K = np.asarray(L, dtype=float).copy()
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    for _ in range(max_iter):
        sinK = np.sin(K)
        cosK = np.cos(K)
        f = L - K - p1 * cosK + p2 * sinK
        fp = -1.0 + p1 * sinK + p2 * cosK
        fp = np.where(np.abs(fp) < 1e-15, 1e-15, fp)
        dK = -f / fp
        K += dK
        if np.all(np.abs(dK) < tol):
            break
    return K


def K_to_L(K: float, p1: float, p2: float) -> float:
    """Compute L from K via the forward Kepler equation (Eq. 30)."""
    return K + p1 * np.cos(K) - p2 * np.sin(K)
