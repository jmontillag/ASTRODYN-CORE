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

    Args:
        universe: Optional universe configuration mapping used by frame/mu
            resolvers during Orekit conversions.
        default_mass_kg: Fallback spacecraft mass used when states omit mass.
        interpolation_samples: Optional default interpolation sample count for
            ephemeris export helpers.
    """

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None

    # ------------------------------------------------------------------
    # Core state-file operations
    # ------------------------------------------------------------------

    def load_state_file(self, path: str | Path) -> ScenarioStateFile:
        """Load a YAML/JSON state file into a typed scenario model.

        Args:
            path: Source file path.

        Returns:
            Parsed scenario state file model.
        """
        return _load_state_file(path)

    def load_initial_state(self, path: str | Path) -> OrbitStateRecord:
        """Load only the initial state record from a state file.

        Args:
            path: Source file path.

        Returns:
            Initial orbit state record.
        """
        return _load_initial_state(path)

    def save_state_file(self, path: str | Path, scenario: ScenarioStateFile) -> Path:
        """Save a scenario state file as YAML or JSON.

        Args:
            path: Destination file path.
            scenario: Scenario model to serialize.

        Returns:
            Resolved output path.
        """
        return _save_state_file(path, scenario)

    def save_initial_state(self, path: str | Path, state: OrbitStateRecord) -> Path:
        """Save a file containing only an initial state payload.

        Args:
            path: Destination file path.
            state: Orbit state record to store.

        Returns:
            Resolved output path.
        """
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
        """Save a state series to YAML/JSON compact format or HDF5.

        Args:
            path: Destination file path. ``.h5``/``.hdf5`` selects HDF5.
            series: State series to serialize.
            dense_yaml: When writing YAML/JSON compact rows, prefer dense row
                formatting for YAML output.
            compression: HDF5 compression algorithm for HDF5 outputs.
            compression_level: HDF5 compression level.
            shuffle: Whether to enable HDF5 shuffle filter.

        Returns:
            Resolved output path.
        """
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
        """Load a state series from a scenario YAML/JSON file or HDF5 file.

        Args:
            path: Source file path.
            series_name: Optional series name selector when multiple series are
                present in a scenario file.

        Returns:
            Loaded state series.

        Raises:
            ValueError: If no state series exist, or a named series is missing.
        """
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
        """Convert an ISO-8601 UTC epoch string to Orekit ``AbsoluteDate``."""
        return _to_orekit_date(epoch)

    def from_orekit_date(self, date: Any) -> str:
        """Convert an Orekit ``AbsoluteDate`` to an ISO-8601 UTC epoch string."""
        return _from_orekit_date(date)

    def to_orekit_orbit(
        self,
        record: OrbitStateRecord,
        *,
        universe: Mapping[str, Any] | None = None,
    ) -> Any:
        """Convert a serializable orbit-state record into an Orekit orbit.

        Args:
            record: Serializable orbit-state record.
            universe: Optional per-call universe config override.

        Returns:
            Orekit orbit instance matching the record representation.
        """
        return _to_orekit_orbit(record, universe=self._resolve_universe(universe))

    def state_series_to_ephemeris(
        self,
        series: StateSeries,
        *,
        universe: Mapping[str, Any] | None = None,
        interpolation_samples: int | None = None,
        default_mass_kg: float | None = None,
    ) -> Any:
        """Convert a state series into an Orekit bounded ephemeris propagator.

        Args:
            series: Input state series.
            universe: Optional per-call universe config override.
            interpolation_samples: Optional interpolation sample count override.
            default_mass_kg: Optional fallback mass override.

        Returns:
            Orekit ephemeris / bounded propagator built from the state series.
        """
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
    ) -> Any:
        """Convert a scenario's state series into an Orekit ephemeris.

        Args:
            scenario: Scenario state file model.
            series_name: Optional state-series selector. Defaults to first.
            interpolation_samples: Optional interpolation sample count override.
            default_mass_kg: Optional fallback mass override.

        Returns:
            Orekit ephemeris / bounded propagator.
        """
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
        """Sample a propagator/ephemeris and export a serialized trajectory file.

        Args:
            propagator: Orekit propagator or precomputed ephemeris-like object.
            epoch_spec: Output epoch grid specification.
            output_path: Destination YAML/JSON/HDF5 file path.
            series_name: Name for the exported state series.
            representation: Output state representation.
            frame: Output frame name.
            mu_m3_s2: Gravitational parameter stored in exported records.
            interpolation_samples: Interpolation sample count override.
            dense_yaml: Whether YAML compact rows should use dense formatting.
            universe: Optional per-call universe config override.
            default_mass_kg: Optional fallback mass override.

        Returns:
            Resolved output path.
        """
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
        """Resolve a per-call universe override against the client default."""
        if universe is not None:
            return universe
        return self.universe

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        """Resolve a per-call default mass override against the client default."""
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)

    def _resolve_optional_interpolation_samples(
        self, interpolation_samples: int | None
    ) -> int | None:
        """Resolve an optional interpolation sample override."""
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is None:
            return None
        return int(self.interpolation_samples)

    def _resolve_required_interpolation_samples(self, interpolation_samples: int | None) -> int:
        """Resolve interpolation samples, falling back to the package default (8)."""
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is not None:
            return int(self.interpolation_samples)
        return 8
