"""Unified high-level client that groups major end-user workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.mission import MissionClient
from astrodyn_core.states import StateFileClient
from astrodyn_core.tle import TLEClient
from astrodyn_core.uncertainty import UncertaintyClient


@dataclass(slots=True)
class AstrodynClient:
    """Unified app-level façade for common state, mission, uncertainty, and TLE workflows."""

    universe: Mapping[str, Any] | None = None
    default_mass_kg: float = 1000.0
    interpolation_samples: int | None = None
    tle_base_dir: str | Path = "data/tle"
    tle_allow_download: bool = False
    space_track_client: Any | None = None

    state: StateFileClient = field(init=False)
    mission: MissionClient = field(init=False)
    uncertainty: UncertaintyClient = field(init=False)
    tle: TLEClient = field(init=False)

    def __post_init__(self) -> None:
        self.state = StateFileClient(
            universe=self.universe,
            default_mass_kg=self.default_mass_kg,
            interpolation_samples=self.interpolation_samples,
        )
        self.mission = MissionClient(
            universe=self.universe,
            default_mass_kg=self.default_mass_kg,
            interpolation_samples=self.interpolation_samples,
        )
        self.uncertainty = UncertaintyClient(
            default_mass_kg=self.default_mass_kg,
        )
        self.tle = TLEClient(
            base_dir=self.tle_base_dir,
            allow_download=self.tle_allow_download,
            space_track_client=self.space_track_client,
        )
