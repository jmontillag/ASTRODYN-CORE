"""Load/save helpers for scenario state files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from astrodyn_core.states.models import OrbitStateRecord, ScenarioStateFile


def load_state_file(path: str | Path) -> ScenarioStateFile:
    """Load a state file (YAML or JSON) into a typed ScenarioStateFile."""
    data = _read_mapping(path)

    if _looks_like_orbit_state(data):
        return ScenarioStateFile(initial_state=OrbitStateRecord.from_mapping(data))
    return ScenarioStateFile.from_mapping(data)


def load_initial_state(path: str | Path) -> OrbitStateRecord:
    """Load only the initial state record from a state file."""
    scenario = load_state_file(path)
    if scenario.initial_state is None:
        raise ValueError(f"No initial_state found in state file '{path}'.")
    return scenario.initial_state


def save_state_file(path: str | Path, scenario: ScenarioStateFile) -> Path:
    """Write a ScenarioStateFile as YAML or JSON based on extension."""
    if not isinstance(scenario, ScenarioStateFile):
        raise TypeError("scenario must be a ScenarioStateFile instance.")
    payload = scenario.to_mapping()
    output_path = Path(path)
    _write_mapping(output_path, payload)
    return output_path


def save_initial_state(path: str | Path, state: OrbitStateRecord) -> Path:
    """Write a file containing only schema_version + initial_state."""
    scenario = ScenarioStateFile(initial_state=state)
    return save_state_file(path, scenario)


def _read_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with open(file_path) as fh:
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(fh)
        elif file_path.suffix.lower() == ".json":
            raw = json.load(fh)
        else:
            raise ValueError("Unsupported file extension. Use .yaml, .yml, or .json")

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a mapping at top level, got {type(raw).__name__}.")
    return dict(raw)


def _write_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(payload, fh, sort_keys=False)
        elif path.suffix.lower() == ".json":
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        else:
            raise ValueError("Unsupported file extension. Use .yaml, .yml, or .json")


def _looks_like_orbit_state(data: Mapping[str, Any]) -> bool:
    if "schema_version" in data:
        return False
    return "epoch" in data and "representation" in data
