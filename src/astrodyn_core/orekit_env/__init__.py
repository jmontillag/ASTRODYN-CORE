"""Shared Orekit environment configuration and resolver helpers.

This package centralizes the environment settings used across subsystems:

- universe configuration loading/validation
- IERS/ITRF frame resolution
- Earth shape and gravitational parameter resolution

Most users import these helpers indirectly through higher-level APIs, but the
functions here are stable and useful for advanced workflows and tests.
"""

from astrodyn_core.orekit_env.earth import get_earth_shape, get_mu
from astrodyn_core.orekit_env.frames import get_iers_conventions, get_itrf_frame, get_itrf_version
from astrodyn_core.orekit_env.universe_config import (
    coerce_bool,
    get_universe_config,
    load_default_universe_config,
    load_universe_config,
    load_universe_from_dict,
    resolve_universe_config,
)

__all__ = [
    "coerce_bool",
    "get_earth_shape",
    "get_iers_conventions",
    "get_itrf_frame",
    "get_itrf_version",
    "get_mu",
    "get_universe_config",
    "load_default_universe_config",
    "load_universe_config",
    "load_universe_from_dict",
    "resolve_universe_config",
]
