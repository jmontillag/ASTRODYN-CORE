"""Force-model parser helpers for propagation config payloads."""

from __future__ import annotations

from typing import Any

from astrodyn_core.propagation.forces import (
    DragSpec,
    ForceSpec,
    GravitySpec,
    OceanTidesSpec,
    RelativitySpec,
    SRPSpec,
    SolidTidesSpec,
    ThirdBodySpec,
)


def parse_forces(raw: dict[str, Any] | None) -> list[ForceSpec]:
    """Parse a ``forces`` config section into declarative force specs.

    Args:
        raw: Mapping of force keys to ``true``/``false`` or parameter mappings.

    Returns:
        Parsed list of force specs in input order.

    Raises:
        ValueError: If an unknown force key is encountered.
    """
    if raw is None:
        return []

    specs: list[ForceSpec] = []

    for key, value in raw.items():
        if value is False:
            continue

        params = {} if value is True else (value if isinstance(value, dict) else {})

        parser = FORCE_PARSERS.get(key)
        if parser is None:
            raise ValueError(
                f"Unknown force key '{key}' in config. Supported: {sorted(FORCE_PARSERS.keys())}"
            )
        specs.append(parser(params))

    return specs


def parse_gravity(params: dict) -> GravitySpec:
    """Parse a gravity force config mapping into ``GravitySpec``."""
    return GravitySpec(
        degree=params.get("degree", 0),
        order=params.get("order", 0),
        normalized=params.get("normalized", True),
    )


def parse_drag(params: dict) -> DragSpec:
    """Parse a drag force config mapping into ``DragSpec``."""
    return DragSpec(
        atmosphere_model=params.get("atmosphere_model", "nrlmsise00"),
        space_weather_source=params.get("space_weather_source", "cssi"),
        solar_activity_strength=params.get("solar_activity_strength", "average"),
        space_weather_data=params.get("space_weather_data", "default"),
        ref_rho=params.get("ref_rho"),
        ref_alt=params.get("ref_alt"),
        scale_height=params.get("scale_height"),
    )


def parse_srp(params: dict) -> SRPSpec:
    """Parse an SRP force config mapping into ``SRPSpec``."""
    return SRPSpec(
        enable_moon_eclipse=params.get("enable_moon_eclipse", False),
        enable_albedo=params.get("enable_albedo", False),
    )


def parse_third_body(params: dict) -> ThirdBodySpec:
    """Parse a third-body force config mapping into ``ThirdBodySpec``."""
    bodies = params.get("bodies", ("sun", "moon"))
    if isinstance(bodies, list):
        bodies = tuple(bodies)
    return ThirdBodySpec(bodies=bodies)


def parse_relativity(params: dict) -> RelativitySpec:
    """Parse a relativity force config mapping into ``RelativitySpec``."""
    return RelativitySpec()


def parse_solid_tides(params: dict) -> SolidTidesSpec:
    """Parse a solid-tides force config mapping into ``SolidTidesSpec``."""
    return SolidTidesSpec()


def parse_ocean_tides(params: dict) -> OceanTidesSpec:
    """Parse an ocean-tides force config mapping into ``OceanTidesSpec``."""
    return OceanTidesSpec(
        degree=params.get("degree", 5),
        order=params.get("order", 5),
    )


FORCE_PARSERS: dict[str, Any] = {
    "gravity": parse_gravity,
    "drag": parse_drag,
    "srp": parse_srp,
    "third_body": parse_third_body,
    "relativity": parse_relativity,
    "solid_tides": parse_solid_tides,
    "ocean_tides": parse_ocean_tides,
}
