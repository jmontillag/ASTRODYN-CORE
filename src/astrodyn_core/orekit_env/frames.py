"""Orekit frame and convention resolvers from universe configuration."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.orekit_env.universe_config import resolve_universe_config


def get_iers_conventions(universe: Mapping[str, Any] | None = None) -> Any:
    """Resolve the configured Orekit ``IERSConventions`` enum value.

    Args:
        universe: Optional universe configuration mapping. If omitted, the
            active global universe configuration is used.

    Returns:
        The Orekit ``IERSConventions`` enum member selected by the configuration.

    Raises:
        ModuleNotFoundError: If Orekit is not installed in the active Python
            environment.
        RuntimeError: If the JVM/Orekit bridge is not initialized and the local
            Orekit installation requires explicit initialization before import.
    """
    from org.orekit.utils import IERSConventions

    cfg = resolve_universe_config(universe)
    return getattr(IERSConventions, cfg["iers_conventions"])


def get_itrf_version(universe: Mapping[str, Any] | None = None) -> Any:
    """Resolve the configured Orekit ``ITRFVersion`` enum value.

    Args:
        universe: Optional universe configuration mapping. If omitted, the
            active global universe configuration is used.

    Returns:
        The Orekit ``ITRFVersion`` enum member selected by the configuration.

    Raises:
        ModuleNotFoundError: If Orekit is not installed in the active Python
            environment.
    """
    from org.orekit.frames import ITRFVersion

    cfg = resolve_universe_config(universe)
    return getattr(ITRFVersion, cfg["itrf_version"])


def get_itrf_frame(universe: Mapping[str, Any] | None = None) -> Any:
    """Build the configured Orekit ITRF terrestrial reference frame.

    The frame is resolved from:

    - ITRF version (for example ``ITRF_2020``)
    - IERS conventions (for example ``IERS_2010``)
    - ``use_simple_eop`` Earth-orientation-parameter handling flag

    Args:
        universe: Optional universe configuration mapping. If omitted, the
            active global universe configuration is used.

    Returns:
        The Orekit frame instance returned by ``FramesFactory.getITRF(...)``.

    Raises:
        ModuleNotFoundError: If Orekit is not installed in the active Python
            environment.
    """
    from org.orekit.frames import FramesFactory

    cfg = resolve_universe_config(universe)
    return FramesFactory.getITRF(
        get_itrf_version(cfg),
        get_iers_conventions(cfg),
        bool(cfg["use_simple_eop"]),
    )
