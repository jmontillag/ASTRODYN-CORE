"""Compatibility facade for propagation configuration APIs.

.. deprecated::
    This module is a backward-compatibility facade from Phase B.  All public
    symbols have been moved to their canonical modules:

    - ``propagation.universe``           : get_mu, get_earth_shape, load_universe_config, ...
    - ``propagation.parsers.dynamics``   : load_dynamics_config, load_dynamics_from_dict, ...
    - ``propagation.parsers.spacecraft`` : load_spacecraft_config, load_spacecraft_from_dict, ...
    - ``propagation.parsers.forces``     : parse_gravity, parse_drag, parse_srp, ...

    Import from the canonical modules or use ``PropagationClient`` instead.
    This facade and its private underscore aliases will be removed in a
    future release.
"""

from __future__ import annotations

import warnings as _warnings

from astrodyn_core.propagation.parsers.dynamics import (
    load_dynamics_config,
    load_dynamics_from_dict,
    parse_attitude,
    parse_integrator,
)
from astrodyn_core.propagation.parsers.forces import (
    FORCE_PARSERS,
    parse_drag,
    parse_forces,
    parse_gravity,
    parse_ocean_tides,
    parse_relativity,
    parse_solid_tides,
    parse_srp,
    parse_third_body,
)
from astrodyn_core.propagation.parsers.spacecraft import (
    load_spacecraft_config,
    load_spacecraft_from_dict,
    parse_structured_spacecraft_v1,
)
from astrodyn_core.propagation.universe import (
    coerce_bool,
    get_earth_shape,
    get_iers_conventions,
    get_itrf_frame,
    get_itrf_version,
    get_mu,
    get_universe_config,
    load_default_universe_config,
    load_universe_config,
    load_universe_from_dict,
    resolve_universe_config,
)


# ---------------------------------------------------------------------------
# Deprecated private aliases — will be removed in a future release.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    _DEPRECATED_ALIASES = {
        "_coerce_bool": coerce_bool,
        "_resolve_universe_config": resolve_universe_config,
        "_parse_integrator": parse_integrator,
        "_parse_attitude": parse_attitude,
        "_parse_forces": parse_forces,
        "_parse_gravity": parse_gravity,
        "_parse_drag": parse_drag,
        "_parse_srp": parse_srp,
        "_parse_third_body": parse_third_body,
        "_parse_relativity": parse_relativity,
        "_parse_solid_tides": parse_solid_tides,
        "_parse_ocean_tides": parse_ocean_tides,
        "_parse_structured_spacecraft_v1": parse_structured_spacecraft_v1,
        "_FORCE_PARSERS": FORCE_PARSERS,
    }
    if name in _DEPRECATED_ALIASES:
        _warnings.warn(
            f"Importing '{name}' from 'astrodyn_core.propagation.config' is deprecated. "
            f"Import '{name.lstrip('_')}' from its canonical module instead. "
            "This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module 'astrodyn_core.propagation.config' has no attribute {name!r}")


__all__ = [
    "get_earth_shape",
    "get_iers_conventions",
    "get_itrf_frame",
    "get_itrf_version",
    "get_mu",
    "get_universe_config",
    "load_default_universe_config",
    "load_dynamics_config",
    "load_dynamics_from_dict",
    "load_spacecraft_config",
    "load_spacecraft_from_dict",
    "load_universe_config",
    "load_universe_from_dict",
]
