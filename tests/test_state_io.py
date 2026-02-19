from pathlib import Path

import pytest

from astrodyn_core.states import (
    OutputEpochSpec,
    OrbitStateRecord,
    StateFileClient,
    StateSeries,
)

CLIENT = StateFileClient()


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

    record = CLIENT.load_initial_state(path)
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
        CLIENT.load_state_file(path)


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
        CLIENT.load_state_file(path)


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

    CLIENT.save_initial_state(path, original)
    loaded = CLIENT.load_initial_state(path)

    assert loaded.representation == "cartesian"
    assert loaded.position_m == original.position_m
    assert loaded.velocity_mps == original.velocity_mps
    assert loaded.mass_kg == original.mass_kg


def test_load_series_maneuvers_and_attitude_timeline(tmp_path: Path) -> None:
    path = tmp_path / "timeline.yaml"
    path.write_text(
        """
schema_version: 1
state_series:
  - name: "arc-1"
    states:
      - epoch: "2026-02-19T00:00:00Z"
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
      - epoch: "2026-02-19T00:15:00Z"
        frame: "GCRF"
        representation: "keplerian"
        elements:
          a_m: 6878137.0
          e: 0.0012
          i_deg: 51.6
          argp_deg: 45.0
          raan_deg: 120.0
          anomaly_deg: 58.0
          anomaly_type: "MEAN"
        mu_m3_s2: "WGS84"
maneuvers:
  - name: "burn-1"
    trigger:
      type: "epoch"
      epoch: "2026-02-19T00:45:00Z"
    model:
      type: "impulsive"
      dv_mps: [0.0, 5.0, 0.0]
attitude_timeline:
  - mode: "nadir"
    frame: "GCRF"
    params:
      start_epoch: "2026-02-19T00:00:00Z"
""".strip()
    )

    scenario = CLIENT.load_state_file(path)
    assert len(scenario.state_series) == 1
    assert len(scenario.state_series[0].states) == 2
    assert len(scenario.maneuvers) == 1
    assert scenario.maneuvers[0].name == "burn-1"
    assert len(scenario.attitude_timeline) == 1
    assert scenario.attitude_timeline[0].mode == "nadir"


def test_load_compact_state_series_rows() -> None:
    scenario = CLIENT.load_state_file("examples/state_files/leo_state_series.yaml")
    assert len(scenario.state_series) == 1
    series = scenario.state_series[0]
    assert len(series.states) == 3
    assert series.states[0].frame == "GCRF"
    assert series.states[0].elements is not None
    assert series.states[0].elements["anomaly_deg"] == 0.0
    assert series.states[1].elements["anomaly_deg"] == 115.0


def test_load_compact_cartesian_columns_rows(tmp_path: Path) -> None:
    path = tmp_path / "compact_cartesian.yaml"
    path.write_text(
        """
schema_version: 1
state_series:
  - name: "cart-arc"
    defaults:
      representation: "cartesian"
      frame: "GCRF"
      mu_m3_s2: "WGS84"
    columns: [epoch, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps, mass_kg]
    rows:
      - ["2026-02-19T00:00:00Z", 7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0, 450.0]
      - ["2026-02-19T00:02:00Z", 6995000.0, 900000.0, 0.0, -1000.0, 7400.0, 0.0, 449.9]
""".strip()
    )

    scenario = CLIENT.load_state_file(path)
    series = scenario.state_series[0]
    assert len(series.states) == 2
    first = series.states[0]
    assert first.representation == "cartesian"
    assert first.position_m == (7000000.0, 0.0, 0.0)
    assert first.velocity_mps == (0.0, 7500.0, 0.0)


def test_save_compact_dense_rows_emits_inline_yaml_rows(tmp_path: Path) -> None:
    series = StateSeries(
        name="dense",
        states=(
            OrbitStateRecord(
                epoch="2026-02-19T00:00:00Z",
                frame="GCRF",
                representation="cartesian",
                position_m=(7000000.0, 0.0, 0.0),
                velocity_mps=(0.0, 7500.0, 0.0),
                mu_m3_s2="WGS84",
                mass_kg=450.0,
            ),
        ),
    )
    path = tmp_path / "dense.yaml"
    CLIENT.save_state_series(path, series, dense_yaml=True)
    text = path.read_text()
    assert "rows:" in text
    assert "- [" in text


def test_hdf5_state_series_round_trip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    assert h5py is not None

    series = StateSeries(
        name="h5-roundtrip",
        states=(
            OrbitStateRecord(
                epoch="2026-02-19T00:00:00Z",
                frame="GCRF",
                representation="cartesian",
                position_m=(7000000.0, 0.0, 0.0),
                velocity_mps=(0.0, 7500.0, 0.0),
                mu_m3_s2="WGS84",
                mass_kg=450.0,
            ),
            OrbitStateRecord(
                epoch="2026-02-19T00:02:00Z",
                frame="GCRF",
                representation="cartesian",
                position_m=(6999000.0, 900000.0, 0.0),
                velocity_mps=(-950.0, 7400.0, 0.0),
                mu_m3_s2="WGS84",
                mass_kg=449.9,
            ),
        ),
        interpolation={"method": "orekit_ephemeris", "samples": 8},
    )

    path = tmp_path / "series.h5"
    CLIENT.save_state_series(path, series)
    loaded = CLIENT.load_state_series(path)

    assert loaded.name == series.name
    assert loaded.interpolation["samples"] == 8
    assert len(loaded.states) == 2
    assert loaded.states[0].position_m == series.states[0].position_m
    assert loaded.states[1].velocity_mps == series.states[1].velocity_mps


def test_output_epoch_spec_step_and_explicit_modes() -> None:
    step_spec = OutputEpochSpec(
        start_epoch="2026-02-19T00:00:00Z",
        end_epoch="2026-02-19T00:10:00Z",
        step_seconds=120.0,
    )
    step_epochs = step_spec.epochs()
    assert len(step_epochs) == 6
    assert step_epochs[0] == "2026-02-19T00:00:00Z"
    assert step_epochs[-1] == "2026-02-19T00:10:00Z"

    explicit_spec = OutputEpochSpec(
        explicit_epochs=(
            "2026-02-19T00:00:00Z",
            "2026-02-19T00:03:00Z",
            "2026-02-19T00:09:00Z",
        )
    )
    assert explicit_spec.epochs()[1] == "2026-02-19T00:03:00Z"
