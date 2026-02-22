"""Factory and method-entrypoint helpers for uncertainty propagation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from astrodyn_core.uncertainty.spec import UncertaintySpec
from astrodyn_core.uncertainty.stm import STMCovariancePropagator


def setup_stm_propagator(
    propagator: Any,
    spec: UncertaintySpec | None = None,
    *,
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    default_mass_kg: float = 1000.0,
) -> STMCovariancePropagator:
    """Configure a propagator for STM-only extraction (no covariance required)."""
    return STMCovariancePropagator(
        propagator,
        None,
        spec,
        frame=frame,
        mu_m3_s2=mu_m3_s2,
        default_mass_kg=default_mass_kg,
    )


def create_covariance_propagator(
    propagator: Any,
    initial_covariance: np.ndarray | Sequence[Sequence[float]],
    spec: UncertaintySpec,
    *,
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    default_mass_kg: float = 1000.0,
) -> STMCovariancePropagator:
    """Factory that creates the appropriate covariance propagator from a spec.

    Currently supports ``method='stm'`` only.  Additional methods (e.g.
    unscented transform) may be added here by future contributors --
    see ``docs/extending-propagators.md`` for the extension pattern.
    """
    if spec.method == "stm":
        return STMCovariancePropagator(
            propagator,
            initial_covariance,
            spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
        )
    raise ValueError(f"Unknown uncertainty method: {spec.method!r}. Supported: 'stm'.")
