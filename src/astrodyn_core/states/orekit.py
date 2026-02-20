"""Compatibility façade for Orekit conversion and trajectory helpers."""

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

# Backward-compatible private aliases.
_resolve_interpolation_samples = resolve_interpolation_samples
_resolve_sampling_ephemeris = resolve_sampling_ephemeris
_is_precomputed_ephemeris = is_precomputed_ephemeris
_validate_requested_epochs = validate_requested_epochs
_state_to_record = state_to_record

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
