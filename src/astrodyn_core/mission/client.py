"""High-level mission API for maneuver planning, execution, export, and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.mission.executor import MissionExecutionReport, ScenarioExecutor
from astrodyn_core.mission.models import CompiledManeuver
from astrodyn_core.mission.simulation import (
    compile_scenario_maneuvers as _compile_scenario_maneuvers,
    export_scenario_series as _export_scenario_series,
    simulate_scenario_series as _simulate_scenario_series,
)
from astrodyn_core.mission.plotting import (
    plot_orbital_elements_series as _plot_orbital_elements_series,
)
from astrodyn_core.states.models import OutputEpochSpec, ScenarioStateFile, StateSeries


@dataclass(slots=True)
class MissionClient:
    """Facade for mission planning, execution, export, and plotting workflows.

    Args:
        universe: Optional default universe configuration used by frame/mu
            resolvers in downstream helpers.
        default_mass_kg: Fallback spacecraft mass for exported state records.
        interpolation_samples: Optional default interpolation sample count for
            exported trajectories.
    """

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None

    def compile_scenario_maneuvers(
        self,
        scenario: ScenarioStateFile,
        initial_state: Any,
    ) -> tuple[CompiledManeuver, ...]:
        """Compile scenario maneuvers into absolute epochs and inertial delta-v.

        Args:
            scenario: Scenario state file model containing maneuvers/timeline.
            initial_state: Orekit ``SpacecraftState`` used as planning anchor.

        Returns:
            Tuple of compiled maneuvers in execution order.
        """
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
        """Simulate a scenario by applying maneuvers during propagation and sample output.

        Args:
            propagator: Orekit propagator exposing ``propagate`` and
                ``resetInitialState`` when maneuvers are present.
            scenario: Scenario state file model.
            epoch_spec: Output epoch specification.
            series_name: Output state-series name.
            representation: Output orbit representation.
            frame: Output frame name.
            mu_m3_s2: Gravitational parameter stored in output records.
            interpolation_samples: Optional interpolation sample override.
            universe: Optional per-call universe config override.
            default_mass_kg: Optional per-call fallback mass override.

        Returns:
            Tuple ``(StateSeries, compiled_maneuvers)``.
        """
        return _simulate_scenario_series(
            propagator,
            scenario,
            epoch_spec,
            series_name=series_name,
            representation=representation,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            interpolation_samples=self._resolve_required_interpolation_samples(
                interpolation_samples
            ),
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
        """Simulate a scenario and export the sampled trajectory to file.

        Args:
            propagator: Orekit propagator used for simulation.
            scenario: Scenario state file model.
            epoch_spec: Output epoch specification.
            output_path: Destination YAML/JSON/HDF5 path.
            series_name: Output state-series name.
            representation: Output orbit representation.
            frame: Output frame name.
            mu_m3_s2: Gravitational parameter stored in output records.
            interpolation_samples: Optional interpolation sample override.
            dense_yaml: Dense row formatting for YAML compact output.
            universe: Optional per-call universe config override.
            default_mass_kg: Optional per-call fallback mass override.

        Returns:
            Tuple ``(saved_path, compiled_maneuvers)``.
        """
        return _export_scenario_series(
            propagator,
            scenario,
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
        """Execute a scenario in detector-driven mode and sample the result.

        Args:
            propagator: Orekit numerical/DSST propagator supporting event detectors.
            scenario: Scenario state file model.
            epoch_spec: Output epoch specification.
            series_name: Output state-series name.
            representation: Output orbit representation.
            frame: Output frame name.
            mu_m3_s2: Gravitational parameter stored in output records.
            universe: Optional per-call universe config override.
            default_mass_kg: Optional per-call fallback mass override.

        Returns:
            Tuple ``(StateSeries, MissionExecutionReport)``.
        """
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
        """Plot orbital elements from a state series and save as PNG.

        Args:
            series: Input state series.
            output_png: Destination PNG path.
            universe: Optional per-call universe config override.
            title: Optional plot title override.

        Returns:
            Resolved output path.
        """
        return _plot_orbital_elements_series(
            series,
            output_png,
            universe=self._resolve_universe(universe),
            title=title,
        )

    def _resolve_universe(self, universe: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Resolve a per-call universe override against the client default."""
        if universe is not None:
            return universe
        return self.universe

    def _resolve_default_mass(self, default_mass_kg: float | None) -> float:
        """Resolve a per-call fallback mass override against the client default."""
        if default_mass_kg is not None:
            return float(default_mass_kg)
        return float(self.default_mass_kg)

    def _resolve_required_interpolation_samples(self, interpolation_samples: int | None) -> int:
        """Resolve interpolation samples, falling back to the package default (8)."""
        if interpolation_samples is not None:
            return int(interpolation_samples)
        if self.interpolation_samples is not None:
            return int(self.interpolation_samples)
        return 8
