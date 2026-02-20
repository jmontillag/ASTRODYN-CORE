"""Ephemeris conversion helpers for state series and scenarios."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.states.models import ScenarioStateFile, StateSeries
from astrodyn_core.states.orekit_convert import to_orekit_orbit
from astrodyn_core.states.validation import parse_epoch_utc


def state_series_to_ephemeris(
    series: StateSeries,
    *,
    universe: Mapping[str, Any] | None = None,
    interpolation_samples: int | None = None,
    default_mass_kg: float = 1000.0,
):
    """Convert a StateSeries into an Orekit Ephemeris (bounded propagator)."""
    if not isinstance(series, StateSeries):
        raise TypeError("series must be a StateSeries.")

    try:
        from java.util import ArrayList
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.analytical import Ephemeris
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    sorted_records = sorted(series.states, key=lambda item: parse_epoch_utc(item.epoch))

    states = ArrayList()
    for record in sorted_records:
        orbit = to_orekit_orbit(record, universe=universe)
        mass = float(record.mass_kg) if record.mass_kg is not None else float(default_mass_kg)
        states.add(SpacecraftState(orbit, mass))

    samples = resolve_interpolation_samples(series, interpolation_samples)
    return Ephemeris(states, samples)


def scenario_to_ephemeris(
    scenario: ScenarioStateFile,
    *,
    series_name: str | None = None,
    interpolation_samples: int | None = None,
    default_mass_kg: float = 1000.0,
):
    """Convert one state series from a ScenarioStateFile into an Orekit Ephemeris."""
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile.")
    if not scenario.state_series:
        raise ValueError("Scenario has no state_series to convert.")

    selected: StateSeries | None = None
    if series_name is None:
        selected = scenario.state_series[0]
    else:
        for candidate in scenario.state_series:
            if candidate.name == series_name:
                selected = candidate
                break
        if selected is None:
            raise ValueError(f"State series '{series_name}' was not found.")

    return state_series_to_ephemeris(
        selected,
        universe=scenario.universe,
        interpolation_samples=interpolation_samples,
        default_mass_kg=default_mass_kg,
    )


def resolve_interpolation_samples(series: StateSeries, explicit: int | None) -> int:
    """Resolve interpolation sample count from explicit or series hints."""
    if explicit is not None:
        if explicit < 2:
            raise ValueError("interpolation_samples must be >= 2.")
        return int(explicit)

    interpolation_cfg = series.interpolation
    samples = interpolation_cfg.get("samples")
    if samples is not None:
        samples_int = int(samples)
        if samples_int < 2:
            raise ValueError("state_series.interpolation.samples must be >= 2.")
        return samples_int

    hint = (series.interpolation_hint or "").strip().lower()
    hint_to_samples = {
        "linear": 2,
        "coarse": 4,
        "medium": 6,
        "lagrange": 8,
        "fine": 10,
    }
    if hint in hint_to_samples:
        return hint_to_samples[hint]

    return 8
