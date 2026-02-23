"""Generalized Equinoctial Orbital Elements (GEqOE) utilities.

Provides the generalized Kepler equation solver used by the GEqOE
conversion and propagation routines.
"""

from typing import Optional

import numpy as np


def _get_f(Lr: np.ndarray, p1: np.ndarray, p2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Evaluate the generalized Kepler equation residual.

    ``f = Lr - K - p1*cos(K) + p2*sin(K)``
    """
    return Lr - K - p1 * np.cos(K) + p2 * np.sin(K)


def _get_fp(p1: np.ndarray, p2: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Evaluate the derivative of the generalized Kepler equation w.r.t. *K*.

    ``f' = -1 + p1*sin(K) + p2*cos(K)``
    """
    return -1 + p1 * np.sin(K) + p2 * np.cos(K)


def solve_kep_gen(
    Lr: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    k_initial: Optional[np.ndarray] = None,
    tol: float = 1e-14,
    max_iter: int = 1000,
) -> np.ndarray:
    """Solve the generalized Kepler equation via Newton-Raphson.

    Finds *K* such that ``Lr - K - p1*cos(K) + p2*sin(K) = 0``.

    Parameters
    ----------
    Lr : array
        True longitude values.
    p1, p2 : array
        Eccentricity-like parameters.
    k_initial : array, optional
        Initial guess for *K*.  Defaults to *Lr*.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton-Raphson iterations.

    Returns
    -------
    K : array
        Solved eccentric longitude values.
    """
    K = Lr.copy() if k_initial is None else k_initial.copy()

    for _ in range(max_iter):
        f = _get_f(Lr, p1, p2, K)
        fp = _get_fp(p1, p2, K)

        fp[np.abs(fp) < 1e-15] = 1e-15

        delta_K = -f / fp
        K += delta_K

        if np.all(np.abs(delta_K) < tol):
            return K

    raise RuntimeError(
        f"Newton-Raphson solver failed to converge within {max_iter} iterations "
        "in solve_kep_gen."
    )


__all__ = ["solve_kep_gen"]
