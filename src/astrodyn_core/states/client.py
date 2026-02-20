"""High-level state-file API for loading, saving, and Orekit conversion."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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


def _cross_domain_deprecation(method_name: str, target_client: str, stacklevel: int = 3) -> None:
    """Emit a deprecation warning for cross-domain delegation methods."""
    warnings.warn(
        f"StateFileClient.{method_name}() is deprecated. "
        f"Use {target_client} directly, or access it via AstrodynClient. "
        "This method will be removed in a future release.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


@dataclass(slots=True)
class StateFileClient:
    """Single entrypoint for state-file workflows and Orekit conversion.

    Core responsibilities (stable):
        - load/save state files, initial states, state series
        - Orekit date/orbit conversion
        - ephemeris conversion from state series
        - trajectory export from propagator

    Cross-domain methods (deprecated, use AstrodynClient or domain clients):
        - compile_scenario_maneuvers -> MissionClient
        - export_trajectory_from_scenario -> MissionClient
        - plot_orbital_elements -> MissionClient
        - create_covariance_propagator -> UncertaintyClient
        - propagate_with_covariance -> UncertaintyClient
        - save_covariance_series -> UncertaintyClient
        - load_covariance_series -> UncertaintyClient
        - run_scenario_detector_mode -> MissionClient
    """

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None
    _cached_mission_client: Any | None = field(init=False, default=None, repr=False)
    _cached_uncertainty_client: Any | None = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------
    # Core state-file operations (stable)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Orekit conversion helpers (stable)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cross-domain: Mission delegation (deprecated)
    # ------------------------------------------------------------------

    def compile_scenario_maneuvers(
        self,
        scenario: ScenarioStateFile,
        initial_state: Any,
    ) -> tuple[CompiledManeuver, ...]:
        """.. deprecated:: Use ``MissionClient.compile_scenario_maneuvers()`` instead."""
        _cross_domain_deprecation("compile_scenario_maneuvers", "MissionClient")
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
        """.. deprecated:: Use ``MissionClient.export_trajectory_from_scenario()`` instead."""
        _cross_domain_deprecation("export_trajectory_from_scenario", "MissionClient")
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
        """.. deprecated:: Use ``MissionClient.plot_orbital_elements_series()`` instead."""
        _cross_domain_deprecation("plot_orbital_elements", "MissionClient")
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
    # Cross-domain: Uncertainty delegation (deprecated)
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
        """.. deprecated:: Use ``UncertaintyClient.create_covariance_propagator()`` instead."""
        _cross_domain_deprecation("create_covariance_propagator", "UncertaintyClient")
        return self._uncertainty_client().create_covariance_propagator(
            propagator,
            initial_covariance,
            spec=spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
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
        """.. deprecated:: Use ``UncertaintyClient.propagate_with_covariance()`` instead."""
        _cross_domain_deprecation("propagate_with_covariance", "UncertaintyClient")
        state_series, cov_series = self._uncertainty_client().propagate_with_covariance(
            propagator,
            initial_covariance,
            epoch_spec,
            spec=spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            series_name=series_name,
            covariance_name=covariance_name,
            default_mass_kg=default_mass_kg,
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
        """.. deprecated:: Use ``UncertaintyClient.save_covariance_series()`` instead."""
        _cross_domain_deprecation("save_covariance_series", "UncertaintyClient")
        return self._uncertainty_client().save_covariance_series(path, series, **kwargs)

    def load_covariance_series(self, path: str | Path) -> CovarianceSeries:
        """.. deprecated:: Use ``UncertaintyClient.load_covariance_series()`` instead."""
        _cross_domain_deprecation("load_covariance_series", "UncertaintyClient")
        return self._uncertainty_client().load_covariance_series(path)

    # ------------------------------------------------------------------
    # Cross-domain: Detector-driven scenario execution (deprecated)
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
        """.. deprecated:: Use ``MissionClient.run_scenario_detector_mode()`` instead."""
        _cross_domain_deprecation("run_scenario_detector_mode", "MissionClient")
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self, universe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if universe is not None:
            return universe
        return self.universe

    def _mission_client(self):
        if self._cached_mission_client is None:
            from astrodyn_core.mission import MissionClient

            self._cached_mission_client = MissionClient(
                universe=self.universe,
                default_mass_kg=self.default_mass_kg,
                interpolation_samples=self.interpolation_samples,
            )
        return self._cached_mission_client

    def _uncertainty_client(self):
        if self._cached_uncertainty_client is None:
            from astrodyn_core.uncertainty import UncertaintyClient

            self._cached_uncertainty_client = UncertaintyClient(
                default_mass_kg=self.default_mass_kg,
            )
        return self._cached_uncertainty_client

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
