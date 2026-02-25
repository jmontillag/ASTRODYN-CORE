"""Earth shape and gravitational parameter resolvers from universe configuration."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.orekit_env.frames import get_itrf_frame
from astrodyn_core.orekit_env.universe_config import resolve_universe_config


def get_earth_shape(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Orekit OneAxisEllipsoid Earth shape."""
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.utils import Constants

    cfg = resolve_universe_config(universe)
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

    cfg = resolve_universe_config(universe)
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
