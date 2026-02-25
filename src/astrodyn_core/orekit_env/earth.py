"""Earth shape and gravitational parameter resolvers from universe configuration."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.orekit_env.frames import get_itrf_frame
from astrodyn_core.orekit_env.universe_config import resolve_universe_config


def get_earth_shape(universe: Mapping[str, Any] | None = None) -> Any:
    """Build the configured Orekit ``OneAxisEllipsoid`` Earth model.

    The ellipsoid uses the ITRF frame resolved from the same universe
    configuration. The shape can be one of the predefined model names (for
    example ``WGS84`` or ``GRS80``) or a custom mapping with:

    - ``equatorial_radius`` (meters)
    - ``flattening`` (dimensionless)

    Args:
        universe: Optional universe configuration mapping. If omitted, the
            active global universe configuration is used.

    Returns:
        An Orekit ``OneAxisEllipsoid`` configured from the resolved Earth shape
        and ITRF frame.

    Raises:
        ModuleNotFoundError: If Orekit is not installed in the active Python
            environment.
    """
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
    """Resolve the configured Earth gravitational parameter ``mu``.

    Args:
        universe: Optional universe configuration mapping. If omitted, the
            active global universe configuration is used.

    Returns:
        Earth's gravitational parameter in ``m^3 / s^2``.

    Raises:
        ModuleNotFoundError: If Orekit is not installed in the active Python
            environment and a predefined MU constant must be resolved.
    """
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
