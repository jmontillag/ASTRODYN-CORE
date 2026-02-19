from pathlib import Path

import pytest

from astrodyn_core.states import OrbitStateRecord, load_initial_state, load_state_file, save_initial_state


def test_load_initial_state_keplerian_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text(
        """
schema_version: 1
initial_state:
  epoch: "2026-02-19T00:00:00Z"
  frame: "GCRF"
  representation: "keplerian"
  elements:
    a_m: 6878137.0
    e: 0.0012
    i_deg: 51.6
    argp_deg: 45.0
    raan_deg: 120.0
    anomaly_deg: 0.0
    anomaly_type: "MEAN"
  mu_m3_s2: "WGS84"
  mass_kg: 450.0
""".strip()
    )

    record = load_initial_state(path)
    assert record.representation == "keplerian"
    assert record.frame == "GCRF"
    assert record.elements is not None
    assert record.elements["a_m"] == 6878137.0
    assert record.mass_kg == 450.0


def test_invalid_representation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_repr.yaml"
    path.write_text(
        """
schema_version: 1
initial_state:
  epoch: "2026-02-19T00:00:00Z"
  frame: "GCRF"
  representation: "polar"
  mu_m3_s2: "WGS84"
""".strip()
    )

    with pytest.raises(ValueError, match="representation"):
        load_state_file(path)


def test_invalid_frame_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_frame.yaml"
    path.write_text(
        """
schema_version: 1
initial_state:
  epoch: "2026-02-19T00:00:00Z"
  frame: "MARS_FIXED"
  representation: "cartesian"
  position_m: [7000000.0, 0.0, 0.0]
  velocity_mps: [0.0, 7500.0, 0.0]
  mu_m3_s2: "WGS84"
""".strip()
    )

    with pytest.raises(ValueError, match="frame"):
        load_state_file(path)


def test_save_and_reload_initial_state(tmp_path: Path) -> None:
    path = tmp_path / "saved_state.yaml"
    original = OrbitStateRecord(
        epoch="2026-02-19T00:00:00Z",
        frame="GCRF",
        representation="cartesian",
        position_m=(7000000.0, 1000.0, -2000.0),
        velocity_mps=(10.0, 7500.0, 5.0),
        mu_m3_s2="WGS84",
        mass_kg=300.0,
    )

    save_initial_state(path, original)
    loaded = load_initial_state(path)

    assert loaded.representation == "cartesian"
    assert loaded.position_m == original.position_m
    assert loaded.velocity_mps == original.velocity_mps
    assert loaded.mass_kg == original.mass_kg
