"""YAML configuration loader for propagation specs.

Provides two independent loaders:

- ``load_dynamics_config``   — dynamics YAML -> ``PropagatorSpec``
- ``load_spacecraft_config`` — spacecraft YAML -> ``SpacecraftSpec``

Dynamics and spacecraft are kept separate because the same dynamics model
(force fidelity, integrator, attitude law) is typically reused across many
satellites while the spacecraft physical properties change per object.

At usage time the two are combined::

    spec = load_dynamics_config("high_fidelity.yaml")
    sc   = load_spacecraft_config("my_sat.yaml")
    spec = spec.with_spacecraft(sc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from astrodyn_core.propagation.attitude import AttitudeSpec
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
from astrodyn_core.propagation.spacecraft import SpacecraftSpec
from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec


# ============================================================================
# Public API
# ============================================================================


def load_dynamics_config(
    path: str | Path,
    spacecraft: str | Path | None = None,
) -> PropagatorSpec:
    """Load a dynamics configuration YAML and return a ``PropagatorSpec``.

    Parameters
    ----------
    path:
        Path to the dynamics YAML file.
    spacecraft:
        Optional path to a spacecraft YAML file.  When provided the
        returned spec already has the ``SpacecraftSpec`` attached.

    Returns
    -------
    PropagatorSpec
        Fully constructed spec ready for factory consumption.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    spec = load_dynamics_from_dict(data)

    if spacecraft is not None:
        sc = load_spacecraft_config(spacecraft)
        spec = spec.with_spacecraft(sc)

    return spec


def load_spacecraft_config(path: str | Path) -> SpacecraftSpec:
    """Load a spacecraft configuration YAML and return a ``SpacecraftSpec``.

    The YAML can be either a flat mapping of spacecraft fields at root level,
    or contain a top-level ``spacecraft`` key.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    return load_spacecraft_from_dict(data)


def load_dynamics_from_dict(data: dict[str, Any]) -> PropagatorSpec:
    """Build a ``PropagatorSpec`` from an already-parsed dictionary.

    Expected top-level keys (all optional except ``propagator``):

    - ``propagator``  — ``{kind, position_angle_type}``
    - ``integrator``  — ``{kind, min_step, max_step, ...}``
    - ``attitude``    — ``{mode}`` or ``{mode, provider}``
    - ``forces``      — mapping of force-name -> params or bool
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    # --- propagator section (required) ---
    prop_raw = data.get("propagator", {})
    kind_str = prop_raw.get("kind", "numerical")
    kind = PropagatorKind(kind_str.strip().lower())
    position_angle_type = prop_raw.get("position_angle_type", "MEAN")
    mass_kg = prop_raw.get("mass_kg", 1000.0)

    # --- integrator section ---
    integrator = _parse_integrator(data.get("integrator"))

    # --- attitude section ---
    attitude = _parse_attitude(data.get("attitude"))

    # --- forces section ---
    force_specs = _parse_forces(data.get("forces"))

    # --- TLE section (rare in config files, but supported) ---
    tle = None
    tle_raw = data.get("tle")
    if tle_raw is not None:
        from astrodyn_core.propagation.specs import TLESpec

        tle = TLESpec(line1=tle_raw["line1"], line2=tle_raw["line2"])

    return PropagatorSpec(
        kind=kind,
        mass_kg=mass_kg,
        position_angle_type=position_angle_type,
        integrator=integrator,
        tle=tle,
        force_specs=force_specs,
        attitude=attitude,
    )


def load_spacecraft_from_dict(data: dict[str, Any]) -> SpacecraftSpec:
    """Build a ``SpacecraftSpec`` from an already-parsed dictionary.

    Accepts either a flat mapping or one with a ``spacecraft`` key.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    # Allow nesting under a 'spacecraft' key
    if "spacecraft" in data and isinstance(data["spacecraft"], dict):
        data = data["spacecraft"]

    # Filter to only fields SpacecraftSpec accepts
    from dataclasses import fields as dc_fields

    valid_keys = {f.name for f in dc_fields(SpacecraftSpec)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    # Convert solar_array_axis list -> tuple if present
    if "solar_array_axis" in filtered and isinstance(filtered["solar_array_axis"], list):
        filtered["solar_array_axis"] = tuple(filtered["solar_array_axis"])

    return SpacecraftSpec(**filtered)


# ============================================================================
# Internal parsers
# ============================================================================


def _parse_integrator(raw: dict[str, Any] | None) -> IntegratorSpec | None:
    """Parse the integrator section into an IntegratorSpec."""
    if raw is None:
        return None

    return IntegratorSpec(
        kind=raw.get("kind", raw.get("integrator_type", "dp853")),
        min_step=raw.get("min_step"),
        max_step=raw.get("max_step"),
        position_tolerance=raw.get("position_tolerance"),
        step=raw.get("step", raw.get("step_size")),
        n_steps=raw.get("n_steps"),
    )


def _parse_attitude(raw: dict[str, Any] | str | None) -> AttitudeSpec | None:
    """Parse the attitude section."""
    if raw is None:
        return None

    # Allow shorthand: attitude: "nadir"
    if isinstance(raw, str):
        return AttitudeSpec(mode=raw)

    return AttitudeSpec(mode=raw.get("mode", "inertial"))


def _parse_forces(raw: dict[str, Any] | None) -> list[ForceSpec]:
    """Parse the forces section into a list of ForceSpec objects.

    Each key in the forces dict maps to a force type.  The value can be:
    - ``true``  — enable with defaults
    - a dict    — enable with those parameters
    - ``false`` — skip (force not included)
    """
    if raw is None:
        return []

    specs: list[ForceSpec] = []

    for key, value in raw.items():
        # Skip explicitly disabled forces
        if value is False:
            continue

        # Normalise: true -> empty dict, dict stays dict
        params = {} if value is True else (value if isinstance(value, dict) else {})

        parser = _FORCE_PARSERS.get(key)
        if parser is None:
            raise ValueError(
                f"Unknown force key '{key}' in config. Supported: {sorted(_FORCE_PARSERS.keys())}"
            )
        specs.append(parser(params))

    return specs


# --- Individual force parsers ---


def _parse_gravity(params: dict) -> GravitySpec:
    return GravitySpec(
        degree=params.get("degree", 0),
        order=params.get("order", 0),
        normalized=params.get("normalized", True),
    )


def _parse_drag(params: dict) -> DragSpec:
    return DragSpec(
        atmosphere_model=params.get("atmosphere_model", "nrlmsise00"),
        space_weather_source=params.get("space_weather_source", "cssi"),
        solar_activity_strength=params.get("solar_activity_strength", "average"),
        space_weather_data=params.get("space_weather_data", "default"),
        ref_rho=params.get("ref_rho"),
        ref_alt=params.get("ref_alt"),
        scale_height=params.get("scale_height"),
    )


def _parse_srp(params: dict) -> SRPSpec:
    return SRPSpec(
        enable_moon_eclipse=params.get("enable_moon_eclipse", False),
        enable_albedo=params.get("enable_albedo", False),
    )


def _parse_third_body(params: dict) -> ThirdBodySpec:
    bodies = params.get("bodies", ("sun", "moon"))
    if isinstance(bodies, list):
        bodies = tuple(bodies)
    return ThirdBodySpec(bodies=bodies)


def _parse_relativity(params: dict) -> RelativitySpec:
    return RelativitySpec()


def _parse_solid_tides(params: dict) -> SolidTidesSpec:
    return SolidTidesSpec()


def _parse_ocean_tides(params: dict) -> OceanTidesSpec:
    return OceanTidesSpec(
        degree=params.get("degree", 5),
        order=params.get("order", 5),
    )


_FORCE_PARSERS: dict[str, Any] = {
    "gravity": _parse_gravity,
    "drag": _parse_drag,
    "srp": _parse_srp,
    "third_body": _parse_third_body,
    "relativity": _parse_relativity,
    "solid_tides": _parse_solid_tides,
    "ocean_tides": _parse_ocean_tides,
}
