"""High-level mission API for maneuver planning, execution, export, and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.mission.executor import MissionExecutionReport, ScenarioExecutor
from astrodyn_core.mission.maneuvers import (
    CompiledManeuver,
    compile_scenario_maneuvers as _compile_scenario_maneuvers,
    export_scenario_series as _export_scenario_series,
    simulate_scenario_series as _simulate_scenario_series,
)
from astrodyn_core.mission.plotting import plot_orbital_elements_series as _plot_orbital_elements_series
from astrodyn_core.states.models import OutputEpochSpec, ScenarioStateFile, StateSeries


@dataclass(slots=True)
class MissionClient:
    """Single entrypoint for mission-profile workflows."""

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None

    def compile_scenario_maneuvers(
        self,
        scenario: ScenarioStateFile,
        initial_state: Any,
    ) -> tuple[CompiledManeuver, ...]:
        return _compile_scenario_maneuvers(scenario, initial_state)

    def simulate_scenario_series(
        self,
        propagator: Any,
        scenario: ScenarioStateFile,
        epoch_spec: OutputEpochSpec,
        *,
        series_name: str = "trajectory",
        representation: str = "cartesian",
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        interpolation_samples: int | None = None,
        universe: Mapping[str, Any] | None = None,
        default_mass_kg: float | None = None,
    ) -> tuple[StateSeries, tuple[CompiledManeuver, ...]]:
        return _simulate_scenario_series(
            propagator,
            scenario,
            epoch_spec,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            interpolation_samples=self._resolve_required_interpolation_samples(interpolation_samples),
            universe=self._resolve_universe(universe),
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

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
        return _export_scenario_series(
            propagator,
            scenario,
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
        universe: Mapping[str, Any] | None = None,
        default_mass_kg: float | None = None,
    ) -> tuple[StateSeries, MissionExecutionReport]:
        executor = ScenarioExecutor(
            propagator,
            scenario,
            universe=self._resolve_universe(universe),
        )
        return executor.run_and_sample(
            epoch_spec,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=self._resolve_default_mass(default_mass_kg),
        )

    def plot_orbital_elements_series(
        self,
        series: StateSeries,
        output_png: str | Path,
        *,
        universe: Mapping[str, Any] | None = None,
        title: str | None = None,
    ) -> Path:
        return _plot_orbital_elements_series(
            series,
            output_png,
            universe=self._resolve_universe(universe),
            title=title,
        )

    def _resolve_universe(self, universe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if universe is not None:
            return universe
        return self.universe

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)

    def _resolve_required_interpolation_samples(self, interpolation_samples: int | None) -> int:
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is not None:
            return int(self.interpolation_samples)
        return 8
