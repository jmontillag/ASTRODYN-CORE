"""Shared Orekit environment config and resolver helpers."""

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
