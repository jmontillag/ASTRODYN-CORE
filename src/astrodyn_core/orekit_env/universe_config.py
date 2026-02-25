"""Universe model loading, validation, and cached resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

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
    """Load, validate, and cache a universe configuration from YAML.

    The YAML payload may either contain universe keys at the root level or under
    a top-level ``universe`` mapping. Values are normalized and validated using
    :func:`load_universe_from_dict`.

    This function updates the module-global active configuration cache used by
    :func:`get_universe_config`.

    Args:
        path: Path to a YAML file containing universe settings.

    Returns:
        A normalized copy of the loaded universe configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML payload or ``universe`` section is not a mapping.
        ValueError: If a configured option is unsupported or invalid.
        KeyError: If a custom Earth shape mapping is missing required keys.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    config_path = Path(path)
    with open(config_path) as fh:
        data = yaml.safe_load(fh) or {}
    universe = load_universe_from_dict(data)

    global _UNIVERSE_CONFIG
    _UNIVERSE_CONFIG = dict(universe)
    return dict(universe)


def load_default_universe_config() -> dict[str, Any]:
    """Load and cache the repository default universe configuration.

    Returns:
        A normalized copy of the default universe configuration.
    """
    return load_universe_config(_DEFAULT_UNIVERSE_CONFIG_PATH)


def get_universe_config() -> dict[str, Any]:
    """Return the active universe configuration, loading defaults on demand.

    If no configuration has been explicitly loaded yet, this function first uses
    the built-in defaults and then, when present, loads the repository default
    YAML file.

    Returns:
        A copy of the active normalized universe configuration.
    """
    global _UNIVERSE_CONFIG
    if _UNIVERSE_CONFIG is None:
        _UNIVERSE_CONFIG = dict(_DEFAULT_UNIVERSE_CONFIG)
        if _DEFAULT_UNIVERSE_CONFIG_PATH.exists():
            _UNIVERSE_CONFIG = load_universe_config(_DEFAULT_UNIVERSE_CONFIG_PATH)
    return dict(_UNIVERSE_CONFIG)


def load_universe_from_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a parsed universe configuration mapping.

    The input may be either:

    - a mapping containing universe keys directly, or
    - a mapping with a top-level ``universe`` section

    The returned configuration is normalized to canonical strings/booleans and
    validated against the supported IERS/ITRF/Earth-shape/MU options.

    Args:
        data: Parsed configuration mapping.

    Returns:
        A normalized universe configuration mapping. The input mapping is not
        modified.

    Raises:
        TypeError: If ``data`` or the nested ``universe`` section is not a
            mapping, or if configured values have the wrong type.
        ValueError: If a configured option is unsupported or malformed.
        KeyError: If a custom ``earth_shape_model`` mapping is missing required
            keys.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping for universe config, got {type(data).__name__}")

    raw = data.get("universe", data)
    if not isinstance(raw, Mapping):
        raise TypeError("The 'universe' section must be a mapping.")

    cfg = dict(_DEFAULT_UNIVERSE_CONFIG)
    cfg.update(dict(raw))

    cfg["iers_conventions"] = str(cfg["iers_conventions"]).strip().upper()
    cfg["itrf_version"] = str(cfg["itrf_version"]).strip().upper()
    cfg["use_simple_eop"] = coerce_bool(cfg.get("use_simple_eop", True))

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


def coerce_bool(value: Any) -> bool:
    """Coerce bool-like YAML values into ``bool``.

    Recognized string values (case-insensitive):

    - truthy: ``1``, ``true``, ``yes``, ``on``
    - falsy: ``0``, ``false``, ``no``, ``off``

    Unrecognized values fall back to Python's ``bool(value)`` semantics.

    Args:
        value: Arbitrary value parsed from config/YAML input.

    Returns:
        The coerced boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def resolve_universe_config(universe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve explicit or globally configured universe settings.

    Args:
        universe: Optional explicit universe mapping. If provided, it is
            validated and normalized. If omitted, the current active config is
            returned (loading defaults if necessary).

    Returns:
        A normalized universe configuration mapping.

    Raises:
        TypeError: If ``universe`` is provided but is not a valid mapping.
        ValueError: If ``universe`` contains unsupported values.
        KeyError: If a custom Earth shape mapping is missing required keys.
    """
    if universe is None:
        return get_universe_config()
    return load_universe_from_dict(universe)
