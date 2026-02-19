#!/usr/bin/env python
"""Demonstrate practical uses of ScenarioStateFile beyond state-series storage.

This example shows how a scenario object can be used to:
1. Load mission-level metadata and timeline sections.
2. Build a propagator context directly from `initial_state`.
3. Query maneuver schedule and evaluate state at maneuver epochs.
4. Resolve active attitude mode at arbitrary epochs.
5. Save an enriched scenario file with derived metadata.

Run from repo root:
    python examples/demo_scenario_object_uses.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from astrodyn_core import (  # noqa: E402
    BuildContext,
    PropagatorFactory,
    PropagatorKind,
    PropagatorSpec,
    ProviderRegistry,
    ScenarioStateFile,
    StateFileClient,
    register_default_orekit_providers,
)


def _parse_epoch_utc(epoch: str) -> datetime:
    normalized = epoch.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_attitude_mode_at(scenario: ScenarioStateFile, epoch_utc: datetime) -> str:
    selected_mode = "unknown"
    selected_start: datetime | None = None

    for entry in scenario.attitude_timeline:
        start_raw = entry.params.get("start_epoch")
        if not isinstance(start_raw, str):
            continue
        start_dt = _parse_epoch_utc(start_raw)
        if start_dt <= epoch_utc and (selected_start is None or start_dt > selected_start):
            selected_mode = entry.mode
            selected_start = start_dt
    return selected_mode


def main() -> None:
    state_dir = Path(__file__).resolve().parent / "state_files"
    scenario_path = state_dir / "leo_mission_timeline.yaml"
    output_path = state_dir / "generated_scenario_enriched.yaml"

    client = StateFileClient()
    scenario = client.load_state_file(scenario_path)

    if scenario.initial_state is None:
        raise RuntimeError("Scenario needs initial_state for this demo.")

    mission_name = str(scenario.metadata.get("mission", "unknown"))
    print(f"Scenario mission: {mission_name}")
    print(f"Maneuvers loaded: {len(scenario.maneuvers)}")
    print(f"Attitude entries: {len(scenario.attitude_timeline)}")

    # Build propagation context directly from scenario.initial_state.
    ctx = BuildContext.from_state_record(
        scenario.initial_state,
        universe=scenario.universe,
        metadata={"source_scenario": scenario_path.name},
    )

    registry = ProviderRegistry()
    register_default_orekit_providers(registry)
    factory = PropagatorFactory(registry=registry)
    spec = PropagatorSpec(
        kind=PropagatorKind.KEPLERIAN,
        mass_kg=float(scenario.initial_state.mass_kg or 450.0),
    )
    builder = factory.build_builder(spec, ctx)
    propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

    previewed_maneuvers = 0
    for maneuver in scenario.maneuvers:
        trigger_type = str(maneuver.trigger.get("type", "")).strip().lower()
        epoch_raw = maneuver.trigger.get("epoch")
        if trigger_type != "epoch" or not isinstance(epoch_raw, str):
            continue

        epoch_dt = _parse_epoch_utc(epoch_raw)
        date = client.to_orekit_date(epoch_raw)
        state = propagator.propagate(date)
        position_norm_km = state.getPVCoordinates().getPosition().getNorm() / 1000.0
        active_mode = _active_attitude_mode_at(scenario, epoch_dt)

        print(
            f"Maneuver '{maneuver.name}' at {epoch_raw}: "
            f"|r|={position_norm_km:.3f} km, attitude_mode={active_mode}"
        )
        previewed_maneuvers += 1

    enriched_metadata = dict(scenario.metadata)
    enriched_metadata["derived_preview_maneuver_count"] = previewed_maneuvers
    enriched_metadata["derived_preview_frame"] = scenario.initial_state.frame

    enriched = ScenarioStateFile(
        schema_version=scenario.schema_version,
        universe=scenario.universe,
        spacecraft=scenario.spacecraft,
        initial_state=scenario.initial_state,
        state_series=scenario.state_series,
        maneuvers=scenario.maneuvers,
        attitude_timeline=scenario.attitude_timeline,
        metadata=enriched_metadata,
    )
    client.save_state_file(output_path, enriched)
    print(f"Saved enriched scenario: {output_path.name}")


if __name__ == "__main__":
    main()
