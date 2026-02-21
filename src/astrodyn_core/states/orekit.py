"""Orekit conversion and trajectory helper re-exports.

Convenience re-export module.  All symbols originate from their canonical
submodules (``states.orekit_dates``, ``states.orekit_resolvers``,
``states.orekit_convert``, ``states.orekit_ephemeris``,
``states.orekit_export``).  For new code, prefer importing from the canonical
modules or using ``StateFileClient``.
"""

from __future__ import annotations

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
