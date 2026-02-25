"""Dynamics configuration parsers for PropagatorSpec construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.parsers.forces import parse_forces
from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec


def load_dynamics_config(
    path: str | Path,
    spacecraft: str | Path | None = None,
) -> PropagatorSpec:
    """Load a dynamics configuration YAML and return a ``PropagatorSpec``.

    Args:
        path: Dynamics YAML file path.
        spacecraft: Optional spacecraft YAML file to parse and attach to the
            resulting spec.

    Returns:
        Parsed propagator spec.
    """
    from astrodyn_core.propagation.parsers.spacecraft import load_spacecraft_config

    with open(path) as fh:
        data = yaml.safe_load(fh)

    spec = load_dynamics_from_dict(data)

    if spacecraft is not None:
        sc = load_spacecraft_config(spacecraft)
        spec = spec.with_spacecraft(sc)

    return spec


def load_dynamics_from_dict(data: dict[str, Any]) -> PropagatorSpec:
    """Build a ``PropagatorSpec`` from an already-parsed dictionary.

    Args:
        data: Parsed dynamics configuration mapping.

    Returns:
        Parsed propagator spec.

    Raises:
        TypeError: If ``data`` is not a dict.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict, got {type(data).__name__}")

    prop_raw = data.get("propagator", {})
    kind_str = prop_raw.get("kind", "numerical")
    kind = PropagatorKind(kind_str.strip().lower())
    position_angle_type = prop_raw.get("position_angle_type", "MEAN")
    dsst_propagation_type = prop_raw.get("dsst_propagation_type", "MEAN")
    dsst_state_type = prop_raw.get("dsst_state_type", "OSCULATING")
    mass_kg = prop_raw.get("mass_kg", 1000.0)

    integrator = parse_integrator(data.get("integrator"))
    attitude = parse_attitude(data.get("attitude"))
    force_specs = parse_forces(data.get("forces"))

    tle = None
    tle_raw = data.get("tle")
    if tle_raw is not None:
        from astrodyn_core.propagation.specs import TLESpec

        tle = TLESpec(line1=tle_raw["line1"], line2=tle_raw["line2"])

    return PropagatorSpec(
        kind=kind,
        mass_kg=mass_kg,
        position_angle_type=position_angle_type,
        dsst_propagation_type=dsst_propagation_type,
        dsst_state_type=dsst_state_type,
        integrator=integrator,
        tle=tle,
        force_specs=force_specs,
        attitude=attitude,
    )


def parse_integrator(raw: dict[str, Any] | None) -> IntegratorSpec | None:
    """Parse the ``integrator`` section into ``IntegratorSpec``.

    Args:
        raw: Integrator section mapping or ``None``.

    Returns:
        Parsed integrator spec, or ``None`` when no section is provided.
    """
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


def parse_attitude(raw: dict[str, Any] | str | None) -> AttitudeSpec | None:
    """Parse the ``attitude`` section.

    Args:
        raw: Attitude config as mapping, simple mode string, or ``None``.

    Returns:
        Parsed attitude spec, or ``None`` when omitted.
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        return AttitudeSpec(mode=raw)

    return AttitudeSpec(mode=raw.get("mode", "inertial"))
