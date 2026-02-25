"""Orekit frame and convention resolvers from universe configuration."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.orekit_env.universe_config import resolve_universe_config


def get_iers_conventions(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit IERSConventions enum."""
    from org.orekit.utils import IERSConventions

    cfg = resolve_universe_config(universe)
    return getattr(IERSConventions, cfg["iers_conventions"])


def get_itrf_version(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit ITRFVersion enum."""
    from org.orekit.frames import ITRFVersion

    cfg = resolve_universe_config(universe)
    return getattr(ITRFVersion, cfg["itrf_version"])


def get_itrf_frame(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit ITRF frame."""
    from org.orekit.frames import FramesFactory

    cfg = resolve_universe_config(universe)
    return FramesFactory.getITRF(
        get_itrf_version(cfg),
        get_iers_conventions(cfg),
        bool(cfg["use_simple_eop"]),
    )
