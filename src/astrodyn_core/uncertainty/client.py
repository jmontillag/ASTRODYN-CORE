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
    """Facade for covariance propagation and covariance-series I/O workflows.

    Args:
        default_mass_kg: Fallback spacecraft mass used when propagated states do
            not expose a mass value.
    """

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
        """Create a covariance propagator configured for the requested method.

        Args:
            propagator: Orekit propagator instance (numerical/DSST) supporting
                STM extraction.
            initial_covariance: Initial covariance matrix (6x6 or 7x7 depending
                on ``spec.include_mass``).
            spec: Uncertainty propagation configuration. Defaults to
                ``UncertaintySpec()``.
            frame: Output frame name for generated covariance/state records.
            mu_m3_s2: Gravitational parameter used in state record serialization
                (or a symbolic resolver value such as ``"WGS84"``).
            default_mass_kg: Optional per-call override for fallback mass.

        Returns:
            Configured ``STMCovariancePropagator``.
        """
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
        """Propagate a trajectory and covariance series on a shared epoch grid.

        Args:
            propagator: Orekit propagator instance supporting STM extraction.
            initial_covariance: Initial covariance matrix (6x6 or 7x7).
            epoch_spec: Output epoch grid specification.
            spec: Uncertainty propagation configuration. Defaults to STM.
            frame: Output frame name for state/covariance records.
            mu_m3_s2: Gravitational parameter for serialized state records.
            series_name: Output state-series name.
            covariance_name: Output covariance-series name.
            default_mass_kg: Optional per-call override for fallback mass.

        Returns:
            Tuple ``(StateSeries, CovarianceSeries)`` evaluated on the same
            epochs.
        """
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
        """Persist a covariance series to YAML or HDF5 based on file extension.

        Args:
            path: Destination file path.
            series: Covariance series to persist.
            **kwargs: Format-specific options forwarded to the underlying save
                function (for example HDF5 compression settings).

        Returns:
            Resolved output path.
        """
        return _save_covariance_series(path, series, **kwargs)

    def load_covariance_series(self, path: str | Path) -> CovarianceSeries:
        """Load a covariance series from YAML or HDF5 based on file extension.

        Args:
            path: Source file path.

        Returns:
            Loaded covariance series.
        """
        return _load_covariance_series(path)

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        """Resolve a per-call mass override against the client default."""
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)
