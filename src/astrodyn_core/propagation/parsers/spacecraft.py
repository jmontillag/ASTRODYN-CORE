"""Spacecraft configuration parsers for SpacecraftSpec construction."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

import yaml

from astrodyn_core.propagation.spacecraft import SpacecraftSpec


def load_spacecraft_config(path: str | Path) -> SpacecraftSpec:
    """Load a spacecraft configuration YAML and return a ``SpacecraftSpec``.

    Args:
        path: Spacecraft YAML file path.

    Returns:
        Parsed spacecraft spec.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    return load_spacecraft_from_dict(data)


def load_spacecraft_from_dict(data: dict[str, Any]) -> SpacecraftSpec:
    """Build a ``SpacecraftSpec`` from an already-parsed dictionary.

    Supports both a flat dataclass-like mapping and the structured schema-v1
    ``spacecraft.model`` layout.

    Args:
        data: Parsed spacecraft configuration mapping.

    Returns:
        Parsed spacecraft spec.

    Raises:
        TypeError: If ``data`` is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    if isinstance(data.get("spacecraft"), dict):
        spacecraft_block = data["spacecraft"]
        model_block = spacecraft_block.get("model")
        if isinstance(model_block, dict):
            return parse_structured_spacecraft_v1(data)

    if "spacecraft" in data and isinstance(data["spacecraft"], dict):
        data = data["spacecraft"]

    valid_keys = {f.name for f in dc_fields(SpacecraftSpec)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    if "solar_array_axis" in filtered and isinstance(filtered["solar_array_axis"], list):
        filtered["solar_array_axis"] = tuple(filtered["solar_array_axis"])

    return SpacecraftSpec(**filtered)


def parse_structured_spacecraft_v1(data: dict[str, Any]) -> SpacecraftSpec:
    """Parse fixed-structure spacecraft schema version 1.

    Args:
        data: Structured spacecraft configuration mapping.

    Returns:
        Parsed spacecraft spec.

    Raises:
        ValueError: If the schema version or model type is unsupported.
        TypeError: If required structured mappings are missing/invalid.
    """
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
