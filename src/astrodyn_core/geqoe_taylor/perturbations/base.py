"""Base protocol for perturbation models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PerturbationModel(Protocol):
    """Interface for conservative perturbation models.

    A model must provide:
    - U_expr: heyoka symbolic expression for the disturbing potential U
    - U_numeric: scalar evaluation of U for coordinate conversions
    """

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        """Return U as a heyoka symbolic expression.

        Args:
            x, y, z: heyoka expressions for Cartesian position components.
            r_mag: heyoka expression for |r|.
            t: heyoka time expression.
            pars: dict mapping parameter names to heyoka par[] expressions.

        Returns:
            heyoka expression for U.
        """
        ...

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        """Evaluate U numerically (for coordinate conversions).

        Args:
            r_vec: position vector (3,) in km.
            t: time in seconds.

        Returns:
            U in km^2/s^2.
        """
        ...
