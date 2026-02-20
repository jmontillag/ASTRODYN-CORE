"""High-level state-file API for loading, saving, and Orekit conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from astrodyn_core.states.io import (
    load_initial_state as _load_initial_state,
    load_state_file as _load_state_file,
    load_state_series_hdf5 as _load_state_series_hdf5,
    save_initial_state as _save_initial_state,
    save_state_file as _save_state_file,
    save_state_series_compact_with_style as _save_state_series_compact_with_style,
    save_state_series_hdf5 as _save_state_series_hdf5,
)
from astrodyn_core.states.models import OrbitStateRecord, OutputEpochSpec, ScenarioStateFile, StateSeries
from astrodyn_core.states.orekit import (
    export_trajectory_from_propagator as _export_trajectory_from_propagator,
    from_orekit_date as _from_orekit_date,
    scenario_to_ephemeris as _scenario_to_ephemeris,
    state_series_to_ephemeris as _state_series_to_ephemeris,
    to_orekit_date as _to_orekit_date,
    to_orekit_orbit as _to_orekit_orbit,
)

if TYPE_CHECKING:
    from astrodyn_core.mission import CompiledManeuver, MissionExecutionReport
    from astrodyn_core.uncertainty.models import CovarianceSeries
    from astrodyn_core.uncertainty.propagator import STMCovariancePropagator
    from astrodyn_core.uncertainty.spec import UncertaintySpec


@dataclass(slots=True)
class StateFileClient:
    """Single entrypoint for state-file workflows and Orekit conversion."""

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None

    def load_state_file(self, path: str | Path) -> ScenarioStateFile:
        return _load_state_file(path)

    def load_initial_state(self, path: str | Path) -> OrbitStateRecord:
        return _load_initial_state(path)

    def save_state_file(self, path: str | Path, scenario: ScenarioStateFile) -> Path:
        return _save_state_file(path, scenario)

    def save_initial_state(self, path: str | Path, state: OrbitStateRecord) -> Path:
        return _save_initial_state(path, state)

    def save_state_series(
        self,
        path: str | Path,
        series: StateSeries,
        *,
        dense_yaml: bool = True,
        compression: str = "gzip",
        compression_level: int = 4,
        shuffle: bool = True,
    ) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() in {".h5", ".hdf5"}:
            return _save_state_series_hdf5(
                output_path,
                series,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
            )
        return _save_state_series_compact_with_style(output_path, series, dense_rows=dense_yaml)

    def load_state_series(self, path: str | Path, *, series_name: str | None = None) -> StateSeries:
        input_path = Path(path)
        if input_path.suffix.lower() in {".h5", ".hdf5"}:
            return _load_state_series_hdf5(input_path)

        scenario = self.load_state_file(input_path)
        if not scenario.state_series:
            raise ValueError(f"No state_series found in '{input_path}'.")

        if series_name is None:
            return scenario.state_series[0]

        for series in scenario.state_series:
            if series.name == series_name:
                return series
        raise ValueError(f"State series '{series_name}' was not found in '{input_path}'.")

    def to_orekit_date(self, epoch: str):
        return _to_orekit_date(epoch)

    def from_orekit_date(self, date: Any) -> str:
        return _from_orekit_date(date)

    def to_orekit_orbit(
        self,
        record: OrbitStateRecord,
        *,
        universe: Mapping[str, Any] | None = None,
    ):
        return _to_orekit_orbit(record, universe=self._resolve_universe(universe))

    def state_series_to_ephemeris(
        self,
        series: StateSeries,
        *,
        universe: Mapping[str, Any] | None = None,
        interpolation_samples: int | None = None,
        default_mass_kg: float | None = None,
    ):
        return _state_series_to_ephemeris(
            series,
            universe=self._resolve_universe(universe),
            interpolation_samples=self._resolve_optional_interpolation_samples(interpolation_samples),
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    def scenario_to_ephemeris(
        self,
        scenario: ScenarioStateFile,
        *,
        series_name: str | None = None,
        interpolation_samples: int | None = None,
        default_mass_kg: float | None = None,
    ):
        return _scenario_to_ephemeris(
            scenario,
            series_name=series_name,
            interpolation_samples=self._resolve_optional_interpolation_samples(interpolation_samples),
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    def export_trajectory_from_propagator(
        self,
        propagator: Any,
        epoch_spec: OutputEpochSpec,
        output_path: str | Path,
        *,
        series_name: str = "trajectory",
        representation: str = "cartesian",
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        interpolation_samples: int | None = None,
        dense_yaml: bool = True,
        universe: Mapping[str, Any] | None = None,
        default_mass_kg: float | None = None,
    ) -> Path:
        return _export_trajectory_from_propagator(
            propagator,
            epoch_spec,
            output_path,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            interpolation_samples=self._resolve_required_interpolation_samples(interpolation_samples),
            dense_yaml=dense_yaml,
            universe=self._resolve_universe(universe),
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    def compile_scenario_maneuvers(
        self,
        scenario: ScenarioStateFile,
        initial_state: Any,
    ) -> tuple[CompiledManeuver, ...]:
        return self._mission_client().compile_scenario_maneuvers(scenario, initial_state)

    def export_trajectory_from_scenario(
        self,
        propagator: Any,
        scenario: ScenarioStateFile,
        epoch_spec: OutputEpochSpec,
        output_path: str | Path,
        *,
        series_name: str = "trajectory",
        representation: str = "cartesian",
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        interpolation_samples: int | None = None,
        dense_yaml: bool = True,
        universe: Mapping[str, Any] | None = None,
        default_mass_kg: float | None = None,
    ) -> tuple[Path, tuple[CompiledManeuver, ...]]:
        return self._mission_client().export_trajectory_from_scenario(
            propagator,
            scenario,
            epoch_spec,
            output_path,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            interpolation_samples=interpolation_samples,
            dense_yaml=dense_yaml,
            universe=universe,
            default_mass_kg=default_mass_kg,
        )

    def plot_orbital_elements(
        self,
        series_or_path: StateSeries | str | Path,
        output_png: str | Path,
        *,
        series_name: str | None = None,
        universe: Mapping[str, Any] | None = None,
        title: str | None = None,
    ) -> Path:
        if isinstance(series_or_path, StateSeries):
            series = series_or_path
        else:
            series = self.load_state_series(series_or_path, series_name=series_name)
        return self._mission_client().plot_orbital_elements_series(
            series,
            output_png,
            universe=universe,
            title=title,
        )

    # ------------------------------------------------------------------
    # Covariance / Uncertainty propagation
    # ------------------------------------------------------------------

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
        """Create a covariance propagator from a numerical Orekit propagator.

        Parameters
        ----------
        propagator:
            Orekit NumericalPropagator (or DSST) instance. Must not have been
            used for propagation yet (``setupMatricesComputation`` must be
            called before the first ``propagate()`` call).
        initial_covariance:
            Initial covariance matrix, shape (6, 6) or (7, 7) if
            ``spec.include_mass=True``.
        spec:
            :class:`~astrodyn_core.uncertainty.spec.UncertaintySpec`.
            Defaults to ``UncertaintySpec(method='stm')``.
        frame:
            Output frame name for state records (default ``"GCRF"``).
        mu_m3_s2:
            Gravitational parameter for output state records.
        default_mass_kg:
            Fallback mass. Uses ``self.default_mass_kg`` when not provided.
        """
        from astrodyn_core.uncertainty.propagator import create_covariance_propagator
        from astrodyn_core.uncertainty.spec import UncertaintySpec as _UncertaintySpec

        resolved_spec = spec if spec is not None else _UncertaintySpec()
        return create_covariance_propagator(
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
        state_output_path: str | Path | None = None,
        covariance_output_path: str | Path | None = None,
        default_mass_kg: float | None = None,
    ) -> tuple[StateSeries, CovarianceSeries]:
        """Propagate state + covariance over a set of epochs.

        Configures the propagator with STM computation, propagates over all
        epochs in ``epoch_spec``, and returns both the state trajectory and the
        propagated covariance series.

        Parameters
        ----------
        propagator:
            Orekit NumericalPropagator (or DSST) instance.
        initial_covariance:
            Initial covariance matrix, shape (6, 6).
        epoch_spec:
            Output epoch specification.
        spec:
            Uncertainty method configuration. Defaults to STM, Cartesian.
        frame:
            Output frame name.
        mu_m3_s2:
            Gravitational parameter for output state records.
        series_name:
            Name for the state :class:`~astrodyn_core.states.models.StateSeries`.
        covariance_name:
            Name for the :class:`~astrodyn_core.uncertainty.models.CovarianceSeries`.
        state_output_path:
            Optional path to save the state series (auto-detects YAML/HDF5).
        covariance_output_path:
            Optional path to save the covariance series (auto-detects YAML/HDF5).
        default_mass_kg:
            Fallback spacecraft mass.

        Returns
        -------
        tuple[StateSeries, CovarianceSeries]
        """
        cov_propagator = self.create_covariance_propagator(
            propagator,
            initial_covariance,
            spec=spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
        )
        state_series, cov_series = cov_propagator.propagate_series(
            epoch_spec,
            series_name=series_name,
            covariance_name=covariance_name,
        )

        if state_output_path is not None:
            self.save_state_series(state_output_path, state_series)
        if covariance_output_path is not None:
            self.save_covariance_series(covariance_output_path, cov_series)

        return state_series, cov_series

    def save_covariance_series(
        self,
        path: str | Path,
        series: CovarianceSeries,
        **kwargs: Any,
    ) -> Path:
        """Save a covariance series to YAML or HDF5 (auto-detected from extension)."""
        from astrodyn_core.uncertainty.io import save_covariance_series as _save_cov

        return _save_cov(path, series, **kwargs)

    def load_covariance_series(self, path: str | Path) -> CovarianceSeries:
        """Load a covariance series from YAML or HDF5 (auto-detected from extension)."""
        from astrodyn_core.uncertainty.io import load_covariance_series as _load_cov

        return _load_cov(path)

    # ------------------------------------------------------------------
    # Detector-driven scenario execution
    # ------------------------------------------------------------------

    def run_scenario_detector_mode(
        self,
        propagator: Any,
        scenario: ScenarioStateFile,
        epoch_spec: OutputEpochSpec,
        *,
        series_name: str = "trajectory",
        representation: str = "cartesian",
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        output_path: str | Path | None = None,
        dense_yaml: bool = True,
        universe: Mapping[str, Any] | None = None,
        default_mass_kg: float | None = None,
    ) -> tuple[StateSeries, MissionExecutionReport]:
        """Execute scenario maneuvers using Orekit event detectors (closed-loop mode).

        Unlike ``export_trajectory_from_scenario`` (which uses Keplerian-approximation
        timing + propagation replay), this method binds each maneuver trigger to a
        real Orekit ``EventDetector`` attached to the numerical propagator. Maneuvers
        fire precisely when the detector condition is met during integration.

        Parameters
        ----------
        propagator:
            Orekit NumericalPropagator instance. Event detectors will be added
            to it before propagation.
        scenario:
            Scenario file with timeline, maneuvers, and optional guard/occurrence
            fields in each maneuver's trigger dict.
        epoch_spec:
            Output epoch specification.
        series_name:
            Name for the returned :class:`~astrodyn_core.states.models.StateSeries`.
        representation:
            Output orbit representation (``"cartesian"``, ``"keplerian"``, or
            ``"equinoctial"``).
        frame:
            Output frame name.
        mu_m3_s2:
            Gravitational parameter for output records.
        output_path:
            Optional path to save the state series.
        dense_yaml:
            Whether to use dense YAML row format.
        universe:
            Optional universe configuration override.
        default_mass_kg:
            Fallback spacecraft mass.

        Returns
        -------
        tuple[StateSeries, MissionExecutionReport]
            The sampled trajectory and a report of which maneuvers fired.
        """
        state_series, report = self._mission_client().run_scenario_detector_mode(
            propagator,
            scenario,
            epoch_spec,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            universe=universe,
            default_mass_kg=default_mass_kg,
        )

        if output_path is not None:
            self.save_state_series(output_path, state_series)

        return state_series, report

    def _resolve_universe(self, universe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if universe is not None:
            return universe
        return self.universe

    def _mission_client(self):
        from astrodyn_core.mission import MissionClient

        return MissionClient(
            universe=self.universe,
            default_mass_kg=self.default_mass_kg,
            interpolation_samples=self.interpolation_samples,
        )

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)

    def _resolve_optional_interpolation_samples(self, interpolation_samples: int | None) -> int | None:
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is None:
            return None
        return int(self.interpolation_samples)

    def _resolve_required_interpolation_samples(self, interpolation_samples: int | None) -> int:
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is not None:
            return int(self.interpolation_samples)
        return 8
