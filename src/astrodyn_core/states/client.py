"""High-level state-file API for loading, saving, and Orekit conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.states.io import (
    load_initial_state as _load_initial_state,
    load_state_file as _load_state_file,
    load_state_series_hdf5 as _load_state_series_hdf5,
    save_initial_state as _save_initial_state,
    save_state_file as _save_state_file,
    save_state_series_compact_with_style as _save_state_series_compact_with_style,
    save_state_series_hdf5 as _save_state_series_hdf5,
)
from astrodyn_core.states.models import (
    OrbitStateRecord,
    OutputEpochSpec,
    ScenarioStateFile,
    StateSeries,
)
from astrodyn_core.states.orekit_convert import to_orekit_orbit as _to_orekit_orbit
from astrodyn_core.states.orekit_dates import (
    from_orekit_date as _from_orekit_date,
    to_orekit_date as _to_orekit_date,
)
from astrodyn_core.states.orekit_ephemeris import (
    scenario_to_ephemeris as _scenario_to_ephemeris,
    state_series_to_ephemeris as _state_series_to_ephemeris,
)
from astrodyn_core.states.orekit_export import (
    export_trajectory_from_propagator as _export_trajectory_from_propagator,
)


@dataclass(slots=True)
class StateFileClient:
    """Single entrypoint for state-file workflows and Orekit conversion.

    Core responsibilities:
        - load/save state files, initial states, state series
        - Orekit date/orbit conversion
        - ephemeris conversion from state series
        - trajectory export from propagator
    """

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None

    # ------------------------------------------------------------------
    # Core state-file operations
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
    # Orekit conversion helpers
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
            interpolation_samples=self._resolve_optional_interpolation_samples(
                interpolation_samples
            ),
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
            interpolation_samples=self._resolve_optional_interpolation_samples(
                interpolation_samples
            ),
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
            interpolation_samples=self._resolve_required_interpolation_samples(
                interpolation_samples
            ),
            dense_yaml=dense_yaml,
            universe=self._resolve_universe(universe),
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_universe(self, universe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if universe is not None:
            return universe
        return self.universe

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)

    def _resolve_optional_interpolation_samples(
        self, interpolation_samples: int | None
    ) -> int | None:
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
