"""Scenario maneuver planning and execution helpers.

Convenience re-export module.  All symbols originate from their canonical
submodules (``mission.models``, ``mission.timeline``, ``mission.intents``,
``mission.kinematics``, ``mission.simulation``).  For new code, prefer
importing from the canonical modules or using ``MissionClient``.
"""

from __future__ import annotations

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


__all__ = [
    "CompiledManeuver",
    "ResolvedTimelineEvent",
    "compile_scenario_maneuvers",
    "simulate_scenario_series",
    "export_scenario_series",
]
