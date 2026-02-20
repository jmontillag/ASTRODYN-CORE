from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from astrodyn_core import AstrodynClient, OrbitStateRecord
from astrodyn_core.mission import MissionClient
from astrodyn_core.states import StateFileClient
from astrodyn_core.tle import TLEClient
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty import UncertaintyClient


def _write_month_file(path: Path, line_pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{line1}\n{line2}" for line1, line2 in line_pairs) + "\n"
    path.write_text(text)


def test_tle_client_resolves_spec_from_local_cache(tmp_path: Path) -> None:
    client = TLEClient(base_dir=tmp_path)
    file_path = client.get_tle_file_path(25544, 2024, 1)
    _write_month_file(
        file_path,
        [
            (
                "1 25544U 98067A   24001.20000000  .00016717  00000-0  10270-3 0  9002",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001",
            )
        ],
    )

    spec = client.resolve_tle_spec_for_epoch(
        25544,
        datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    )
    assert spec.line1.startswith("1 ")
    assert spec.line2.startswith("2 ")


def test_astrodyn_client_composes_state_and_tle_clients(tmp_path: Path) -> None:
    app = AstrodynClient(
        default_mass_kg=777.0,
        tle_base_dir=tmp_path,
    )
    assert isinstance(app.state, StateFileClient)
    assert isinstance(app.mission, MissionClient)
    assert isinstance(app.uncertainty, UncertaintyClient)
    assert isinstance(app.tle, TLEClient)
    assert app.state.default_mass_kg == 777.0
    assert app.mission.default_mass_kg == 777.0
    assert app.uncertainty.default_mass_kg == 777.0
    assert Path(app.tle.base_dir) == tmp_path


def test_mission_client_simulate_delegates_with_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_simulate(
        propagator,
        scenario,
        epoch_spec,
        *,
        series_name,
        representation,
        frame,
        mu_m3_s2,
        interpolation_samples,
        universe,
        default_mass_kg,
    ):
        captured["interpolation_samples"] = interpolation_samples
        captured["default_mass_kg"] = default_mass_kg
        captured["universe"] = universe
        return "series", tuple()

    monkeypatch.setattr("astrodyn_core.mission.client._simulate_scenario_series", _fake_simulate)

    client = MissionClient(universe={"earth": {}}, default_mass_kg=321.0, interpolation_samples=11)
    result = client.simulate_scenario_series("prop", "scenario", "epoch_spec")

    assert result == ("series", tuple())
    assert captured["interpolation_samples"] == 11
    assert captured["default_mass_kg"] == 321.0
    assert captured["universe"] == {"earth": {}}


def test_state_file_client_reuses_cached_delegates() -> None:
    client = StateFileClient(default_mass_kg=555.0)

    mission_a = client._mission_client()
    mission_b = client._mission_client()
    uncertainty_a = client._uncertainty_client()
    uncertainty_b = client._uncertainty_client()

    assert mission_a is mission_b
    assert uncertainty_a is uncertainty_b


def test_astrodyn_client_end_to_end_facade_flow(tmp_path: Path, monkeypatch) -> None:
    app = AstrodynClient(tle_base_dir=tmp_path / "tle")

    initial = OrbitStateRecord(
        epoch="2026-02-19T00:00:00Z",
        frame="GCRF",
        representation="cartesian",
        position_m=(7_000_000.0, 0.0, 0.0),
        velocity_mps=(0.0, 7_500.0, 0.0),
        mu_m3_s2="WGS84",
        mass_kg=450.0,
    )
    init_path = tmp_path / "initial_state.yaml"
    app.state.save_initial_state(init_path, initial)
    loaded = app.state.load_initial_state(init_path)
    assert loaded.frame == "GCRF"

    tle_file = app.tle.get_tle_file_path(25544, 2024, 1)
    _write_month_file(
        tle_file,
        [
            (
                "1 25544U 98067A   24001.20000000  .00016717  00000-0  10270-3 0  9002",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001",
            )
        ],
    )
    spec = app.tle.resolve_tle_spec_for_epoch(
        25544,
        datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    )
    assert spec.line1.startswith("1 ")

    cov_series = CovarianceSeries(
        name="cov-e2e",
        records=(
            CovarianceRecord.from_numpy(
                epoch="2026-02-19T00:00:00Z",
                matrix=np.eye(6),
            ),
        ),
    )
    cov_path = tmp_path / "cov.yaml"
    app.uncertainty.save_covariance_series(cov_path, cov_series)
    loaded_cov = app.uncertainty.load_covariance_series(cov_path)
    assert loaded_cov.name == "cov-e2e"

    def _fake_simulate(*args, **kwargs):
        return "series", tuple()

    monkeypatch.setattr("astrodyn_core.mission.client._simulate_scenario_series", _fake_simulate)
    assert app.mission.simulate_scenario_series("prop", "scenario", "epoch_spec") == ("series", tuple())
