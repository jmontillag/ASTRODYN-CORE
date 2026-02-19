"""YAML configuration loader for propagation specs.

Provides three independent loaders:

- ``load_dynamics_config``   — dynamics YAML -> ``PropagatorSpec``
- ``load_spacecraft_config`` — spacecraft YAML -> ``SpacecraftSpec``
- ``load_universe_config``   — universe YAML -> validated config mapping

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
from typing import Any, Mapping, Sequence

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
# Universe / Orekit configuration
# ============================================================================

_DEFAULT_UNIVERSE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "universe_model" / "basic_model.yaml"
)

_DEFAULT_UNIVERSE_CONFIG: dict[str, Any] = {
    "iers_conventions": "IERS_2010",
    "itrf_version": "ITRF_2020",
    "use_simple_eop": True,
    "earth_shape_model": "WGS84",
    "gravitational_parameter": "WGS84",
}

_UNIVERSE_CONFIG: dict[str, Any] | None = None

_SUPPORTED_IERS_CONVENTIONS = frozenset({"IERS_2010", "IERS_2003", "IERS_1996"})
_SUPPORTED_ITRF_VERSIONS = frozenset({"ITRF_2020", "ITRF_2014", "ITRF_2008"})
_SUPPORTED_PREDEFINED_ELLIPSOIDS = frozenset({"WGS84", "GRS80", "IERS96", "IERS2003", "IERS2010"})
_SUPPORTED_PREDEFINED_MU = frozenset({"WGS84", "GRS80", "EGM96", "EIGEN5C", "IERS2010"})


def load_universe_config(path: str | Path) -> dict[str, Any]:
    """Load and cache the universe configuration from a YAML file."""
    config_path = Path(path)
    with open(config_path) as fh:
        data = yaml.safe_load(fh) or {}
    universe = load_universe_from_dict(data)

    global _UNIVERSE_CONFIG
    _UNIVERSE_CONFIG = dict(universe)
    return dict(universe)


def load_default_universe_config() -> dict[str, Any]:
    """Load and cache the repository default universe configuration."""
    return load_universe_config(_DEFAULT_UNIVERSE_CONFIG_PATH)


def get_universe_config() -> dict[str, Any]:
    """Return currently active universe configuration (loads default on demand)."""
    global _UNIVERSE_CONFIG
    if _UNIVERSE_CONFIG is None:
        _UNIVERSE_CONFIG = dict(_DEFAULT_UNIVERSE_CONFIG)
        if _DEFAULT_UNIVERSE_CONFIG_PATH.exists():
            _UNIVERSE_CONFIG = load_universe_config(_DEFAULT_UNIVERSE_CONFIG_PATH)
    return dict(_UNIVERSE_CONFIG)


def load_universe_from_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an already parsed universe mapping."""
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping for universe config, got {type(data).__name__}")

    raw = data.get("universe", data)
    if not isinstance(raw, Mapping):
        raise TypeError("The 'universe' section must be a mapping.")

    cfg = dict(_DEFAULT_UNIVERSE_CONFIG)
    cfg.update(dict(raw))

    cfg["iers_conventions"] = str(cfg["iers_conventions"]).strip().upper()
    cfg["itrf_version"] = str(cfg["itrf_version"]).strip().upper()
    cfg["use_simple_eop"] = _coerce_bool(cfg.get("use_simple_eop", True))

    if cfg["iers_conventions"] not in _SUPPORTED_IERS_CONVENTIONS:
        raise ValueError(
            f"Unsupported iers_conventions '{cfg['iers_conventions']}'. "
            f"Supported: {sorted(_SUPPORTED_IERS_CONVENTIONS)}"
        )

    if cfg["itrf_version"] not in _SUPPORTED_ITRF_VERSIONS:
        raise ValueError(
            f"Unsupported itrf_version '{cfg['itrf_version']}'. "
            f"Supported: {sorted(_SUPPORTED_ITRF_VERSIONS)}"
        )

    shape = cfg["earth_shape_model"]
    if isinstance(shape, str):
        shape = shape.strip().upper()
        if shape not in _SUPPORTED_PREDEFINED_ELLIPSOIDS:
            raise ValueError(
                f"Unsupported earth_shape_model '{shape}'. "
                f"Supported: {sorted(_SUPPORTED_PREDEFINED_ELLIPSOIDS)} or a custom mapping."
            )
        cfg["earth_shape_model"] = shape
    elif isinstance(shape, Mapping):
        model_type = str(shape.get("type", "custom")).strip().lower()
        if model_type != "custom":
            raise ValueError("Only earth_shape_model.type='custom' is supported for mapping values.")
        try:
            eq_radius = float(shape["equatorial_radius"])
            flattening = float(shape["flattening"])
        except KeyError as exc:
            raise KeyError(f"Custom earth_shape_model missing required key: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid custom earth_shape_model value: {exc}") from exc
        cfg["earth_shape_model"] = {
            "type": "custom",
            "equatorial_radius": eq_radius,
            "flattening": flattening,
        }
    else:
        raise TypeError(
            "earth_shape_model must be a predefined model string or a mapping with custom values."
        )

    mu_cfg = cfg["gravitational_parameter"]
    if isinstance(mu_cfg, str):
        mu_name = mu_cfg.strip().upper()
        if mu_name not in _SUPPORTED_PREDEFINED_MU:
            raise ValueError(
                f"Unsupported gravitational_parameter '{mu_name}'. "
                f"Supported: {sorted(_SUPPORTED_PREDEFINED_MU)} or a float value."
            )
        cfg["gravitational_parameter"] = mu_name
    elif isinstance(mu_cfg, (int, float)):
        cfg["gravitational_parameter"] = float(mu_cfg)
    else:
        raise TypeError("gravitational_parameter must be a predefined model string or a numeric value.")

    return cfg


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _resolve_universe_config(universe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if universe is None:
        return get_universe_config()
    return load_universe_from_dict(universe)


def get_iers_conventions(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit IERSConventions enum."""
    from org.orekit.utils import IERSConventions

    cfg = _resolve_universe_config(universe)
    return getattr(IERSConventions, cfg["iers_conventions"])


def get_itrf_version(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit ITRFVersion enum."""
    from org.orekit.frames import ITRFVersion

    cfg = _resolve_universe_config(universe)
    return getattr(ITRFVersion, cfg["itrf_version"])


def get_itrf_frame(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit ITRF frame."""
    from org.orekit.frames import FramesFactory

    cfg = _resolve_universe_config(universe)
    return FramesFactory.getITRF(
        get_itrf_version(cfg),
        get_iers_conventions(cfg),
        bool(cfg["use_simple_eop"]),
    )


def get_earth_shape(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit OneAxisEllipsoid Earth shape."""
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.utils import Constants

    cfg = _resolve_universe_config(universe)
    shape_cfg = cfg["earth_shape_model"]
    itrf = get_itrf_frame(cfg)

    if isinstance(shape_cfg, str):
        predefined = {
            "WGS84": (Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING),
            "GRS80": (Constants.GRS80_EARTH_EQUATORIAL_RADIUS, Constants.GRS80_EARTH_FLATTENING),
            "IERS96": (
                Constants.IERS96_EARTH_EQUATORIAL_RADIUS,
                Constants.IERS96_EARTH_FLATTENING,
            ),
            "IERS2003": (
                Constants.IERS2003_EARTH_EQUATORIAL_RADIUS,
                Constants.IERS2003_EARTH_FLATTENING,
            ),
            "IERS2010": (
                Constants.IERS2010_EARTH_EQUATORIAL_RADIUS,
                Constants.IERS2010_EARTH_FLATTENING,
            ),
        }
        radius, flattening = predefined[shape_cfg]
        return OneAxisEllipsoid(radius, flattening, itrf)

    return OneAxisEllipsoid(shape_cfg["equatorial_radius"], shape_cfg["flattening"], itrf)


def get_mu(universe: Mapping[str, Any] | None = None) -> float:
    """Return configured Earth gravitational parameter (m^3/s^2)."""
    from org.orekit.utils import Constants

    cfg = _resolve_universe_config(universe)
    mu_cfg = cfg["gravitational_parameter"]
    if isinstance(mu_cfg, (float, int)):
        return float(mu_cfg)

    predefined = {
        "WGS84": Constants.WGS84_EARTH_MU,
        "GRS80": Constants.GRS80_EARTH_MU,
        "EGM96": Constants.EGM96_EARTH_MU,
        "EIGEN5C": Constants.EIGEN5C_EARTH_MU,
        "IERS2010": Constants.IERS2010_EARTH_MU,
    }
    return float(predefined[mu_cfg])


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

    Supports:
    - structured schema (recommended): ``schema_version: 1`` + ``spacecraft`` block
    - legacy schema: flat mapping or ``spacecraft`` key with direct dataclass fields
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    # Structured schema (v1): schema_version + spacecraft.model block
    if isinstance(data.get("spacecraft"), dict):
        spacecraft_block = data["spacecraft"]
        model_block = spacecraft_block.get("model")
        if isinstance(model_block, dict):
            return _parse_structured_spacecraft_v1(data)

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


def _parse_structured_spacecraft_v1(data: dict[str, Any]) -> SpacecraftSpec:
    """Parse fixed-structure spacecraft schema version 1."""
    version = data.get("schema_version", 1)
    if int(version) != 1:
        raise ValueError(f"Unsupported spacecraft schema_version '{version}'. Expected 1.")

    spacecraft = data.get("spacecraft")
    if not isinstance(spacecraft, dict):
        raise TypeError("Structured spacecraft config requires a top-level 'spacecraft' mapping.")

    model = spacecraft.get("model")
    if not isinstance(model, dict):
        raise TypeError("Structured spacecraft config requires 'spacecraft.model' mapping.")

    model_type = str(model.get("type", "isotropic")).strip().lower()
    mass = float(spacecraft.get("mass", 1000.0))

    if model_type == "isotropic":
        drag = model.get("drag", {})
        srp = model.get("srp", {})
        if not isinstance(drag, dict) or not isinstance(srp, dict):
            raise TypeError("Isotropic model requires 'drag' and 'srp' mappings.")
        return SpacecraftSpec(
            mass=mass,
            drag_area=float(drag.get("area", 10.0)),
            drag_coeff=float(drag.get("coeff", 2.2)),
            srp_area=float(srp.get("area", 10.0)),
            srp_coeff=float(srp.get("coeff", 1.5)),
        )

    if model_type == "box_wing":
        box = model.get("box", {})
        solar_array = model.get("solar_array", {})
        dims = box.get("dimensions_m", {}) if isinstance(box, dict) else {}
        if not isinstance(box, dict) or not isinstance(solar_array, dict) or not isinstance(dims, dict):
            raise TypeError("Box-wing model requires 'box', 'box.dimensions_m', and 'solar_array' mappings.")

        axis = solar_array.get("axis", (0.0, 1.0, 0.0))
        if isinstance(axis, list):
            axis = tuple(axis)

        return SpacecraftSpec(
            mass=mass,
            use_box_wing=True,
            x_length=float(dims.get("x", 1.0)),
            y_length=float(dims.get("y", 1.0)),
            z_length=float(dims.get("z", 1.0)),
            solar_array_area=float(solar_array.get("area_m2", 20.0)),
            solar_array_axis=axis,
            box_drag_coeff=float(box.get("drag_coeff", 2.2)),
            box_lift_coeff=float(box.get("lift_coeff", 0.0)),
            box_abs_coeff=float(box.get("abs_coeff", 0.7)),
            box_ref_coeff=float(box.get("ref_coeff", 0.3)),
        )

    raise ValueError(
        f"Unsupported spacecraft.model.type '{model_type}'. Supported: 'isotropic', 'box_wing'."
    )


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
