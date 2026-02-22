"""High-level uncertainty API for covariance propagation and I/O workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from astrodyn_core.states.models import OutputEpochSpec, StateSeries
from astrodyn_core.uncertainty.io import (
    load_covariance_series as _load_covariance_series,
    save_covariance_series as _save_covariance_series,
)
from astrodyn_core.uncertainty.models import CovarianceSeries
from astrodyn_core.uncertainty.factory import (
    create_covariance_propagator as _create_covariance_propagator,
)
from astrodyn_core.uncertainty.stm import STMCovariancePropagator
from astrodyn_core.uncertainty.spec import UncertaintySpec


@dataclass(slots=True)
class UncertaintyClient:
    """Single entrypoint for covariance propagation and covariance-series I/O."""

    default_mass_kg: float = 1000.0

    def create_covariance_propagator(
        self,
        propagator: Any,
        initial_covariance: np.ndarray | Sequence[Sequence[float]],
        *,
        spec: UncertaintySpec | None = None,
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        default_mass_kg: float | None = None,
    ) -> STMCovariancePropagator:
        resolved_spec = spec if spec is not None else UncertaintySpec()
        return _create_covariance_propagator(
            propagator,
            initial_covariance,
            resolved_spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    def propagate_with_covariance(
        self,
        propagator: Any,
        initial_covariance: np.ndarray | Sequence[Sequence[float]],
        epoch_spec: OutputEpochSpec,
        *,
        spec: UncertaintySpec | None = None,
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        series_name: str = "trajectory",
        covariance_name: str = "covariance",
        default_mass_kg: float | None = None,
    ) -> tuple[StateSeries, CovarianceSeries]:
        cov_propagator = self.create_covariance_propagator(
            propagator,
            initial_covariance,
            spec=spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
        )
        return cov_propagator.propagate_series(
            epoch_spec,
            series_name=series_name,
            covariance_name=covariance_name,
        )

    def save_covariance_series(
        self,
        path: str | Path,
        series: CovarianceSeries,
        **kwargs: Any,
    ) -> Path:
        return _save_covariance_series(path, series, **kwargs)

    def load_covariance_series(self, path: str | Path) -> CovarianceSeries:
        return _load_covariance_series(path)

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)
