"""Scenario maneuver planning and execution helpers.

.. deprecated::
    This module is a backward-compatibility facade from Phase B.  All public
    symbols have been moved to their canonical modules:

    - ``mission.models``      : CompiledManeuver, ResolvedTimelineEvent
    - ``mission.timeline``    : resolve_timeline_events, resolve_trigger_date, ...
    - ``mission.intents``     : resolve_delta_v_vector, intent_raise_perigee, ...
    - ``mission.kinematics``  : local_to_inertial_delta_v, local_basis_vectors, ...
    - ``mission.simulation``  : compile_scenario_maneuvers, simulate_scenario_series, ...

    Import from the canonical modules or use ``MissionClient`` instead.
    This facade and its private underscore aliases will be removed in a
    future release.
"""

from __future__ import annotations

import warnings as _warnings

from astrodyn_core.mission.intents import (
    intent_change_inclination,
    intent_raise_perigee,
    intent_raise_semimajor_axis,
    resolve_delta_v_vector,
)
from astrodyn_core.mission.kinematics import (
    local_basis_vectors,
    local_to_inertial_delta_v,
    rotate_vector_about_axis,
    to_vector_tuple,
    tuple_to_vector,
    unit,
)
from astrodyn_core.mission.models import CompiledManeuver, ResolvedTimelineEvent
from astrodyn_core.mission.simulation import (
    apply_impulse_to_state,
    compile_scenario_maneuvers,
    export_scenario_series,
    keplerian_propagate_state,
    simulate_scenario_series,
    state_to_record,
)
from astrodyn_core.mission.timeline import (
    delta_time_to_target_mean_anomaly,
    normalize_angle,
    parse_duration_seconds,
    resolve_maneuver_trigger,
    resolve_timeline_events,
    resolve_trigger_date,
    true_to_mean_anomaly,
)


# ---------------------------------------------------------------------------
# Deprecated private aliases — will be removed in a future release.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    _DEPRECATED_ALIASES = {
        "_resolve_delta_v_vector": resolve_delta_v_vector,
        "_intent_raise_perigee": intent_raise_perigee,
        "_intent_raise_semimajor_axis": intent_raise_semimajor_axis,
        "_intent_change_inclination": intent_change_inclination,
        "_local_to_inertial_delta_v": local_to_inertial_delta_v,
        "_local_basis_vectors": local_basis_vectors,
        "_rotate_vector_about_axis": rotate_vector_about_axis,
        "_unit": unit,
        "_to_vector_tuple": to_vector_tuple,
        "_tuple_to_vector": tuple_to_vector,
        "_resolve_trigger_date": resolve_trigger_date,
        "_resolve_maneuver_trigger": resolve_maneuver_trigger,
        "_resolve_timeline_events": resolve_timeline_events,
        "_parse_duration_seconds": parse_duration_seconds,
        "_delta_time_to_target_mean_anomaly": delta_time_to_target_mean_anomaly,
        "_true_to_mean_anomaly": true_to_mean_anomaly,
        "_normalize_angle": normalize_angle,
        "_keplerian_propagate_state": keplerian_propagate_state,
        "_apply_impulse_to_state": apply_impulse_to_state,
        "_state_to_record": state_to_record,
    }
    if name in _DEPRECATED_ALIASES:
        _warnings.warn(
            f"Importing '{name}' from 'astrodyn_core.mission.maneuvers' is deprecated. "
            f"Import '{name.lstrip('_')}' from its canonical module instead. "
            "This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module 'astrodyn_core.mission.maneuvers' has no attribute {name!r}")


__all__ = [
    "CompiledManeuver",
    "ResolvedTimelineEvent",
    "compile_scenario_maneuvers",
    "simulate_scenario_series",
    "export_scenario_series",
]
