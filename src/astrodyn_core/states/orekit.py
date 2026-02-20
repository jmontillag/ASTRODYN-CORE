"""Compatibility facade for Orekit conversion and trajectory helpers.

.. deprecated::
    This module is a backward-compatibility facade from Phase B.  All public
    symbols have been moved to their canonical modules:

    - ``states.orekit_dates``      : to_orekit_date, from_orekit_date
    - ``states.orekit_resolvers``  : resolve_frame, resolve_mu
    - ``states.orekit_convert``    : to_orekit_orbit, state_to_record
    - ``states.orekit_ephemeris``  : state_series_to_ephemeris, scenario_to_ephemeris
    - ``states.orekit_export``     : export_trajectory_from_propagator

    Import from the canonical modules or use ``StateFileClient`` instead.
    This facade and its private underscore aliases will be removed in a
    future release.
"""

from __future__ import annotations

import warnings as _warnings

from astrodyn_core.states.orekit_convert import state_to_record, to_orekit_orbit
from astrodyn_core.states.orekit_dates import from_orekit_date, to_orekit_date
from astrodyn_core.states.orekit_ephemeris import (
    resolve_interpolation_samples,
    scenario_to_ephemeris,
    state_series_to_ephemeris,
)
from astrodyn_core.states.orekit_export import (
    export_trajectory_from_propagator,
    is_precomputed_ephemeris,
    resolve_sampling_ephemeris,
    validate_requested_epochs,
)
from astrodyn_core.states.orekit_resolvers import resolve_frame, resolve_mu


# ---------------------------------------------------------------------------
# Deprecated private aliases — will be removed in a future release.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    _DEPRECATED_ALIASES = {
        "_resolve_interpolation_samples": resolve_interpolation_samples,
        "_resolve_sampling_ephemeris": resolve_sampling_ephemeris,
        "_is_precomputed_ephemeris": is_precomputed_ephemeris,
        "_validate_requested_epochs": validate_requested_epochs,
        "_state_to_record": state_to_record,
    }
    if name in _DEPRECATED_ALIASES:
        _warnings.warn(
            f"Importing '{name}' from 'astrodyn_core.states.orekit' is deprecated. "
            f"Import '{name.lstrip('_')}' from its canonical module instead. "
            "This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module 'astrodyn_core.states.orekit' has no attribute {name!r}")


__all__ = [
    "to_orekit_orbit",
    "to_orekit_date",
    "from_orekit_date",
    "resolve_frame",
    "resolve_mu",
    "state_series_to_ephemeris",
    "scenario_to_ephemeris",
    "export_trajectory_from_propagator",
]
