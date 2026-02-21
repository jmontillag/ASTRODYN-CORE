"""Propagation configuration API re-exports.

Convenience re-export module.  All symbols originate from their canonical
submodules (``propagation.universe``, ``propagation.parsers.dynamics``,
``propagation.parsers.forces``, ``propagation.parsers.spacecraft``).
For new code, prefer importing from the canonical modules or using
``PropagationClient``.
"""

from __future__ import annotations

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
