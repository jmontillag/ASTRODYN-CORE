from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrodyn_core.propagation.specs import TLESpec
from astrodyn_core.tle import TLEQuery
from astrodyn_core.tle.downloader import download_tles_for_month, get_tle_file_path
from astrodyn_core.tle.parser import find_best_tle_in_file, parse_tle_epoch, parse_tle_file
from astrodyn_core.tle.resolver import resolve_tle_record, resolve_tle_spec


def _write_month_file(path: Path, line_pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{line1}\n{line2}" for line1, line2 in line_pairs) + "\n"
    path.write_text(text)


def test_parse_tle_epoch() -> None:
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002"
    epoch = parse_tle_epoch(line1)
    assert epoch.tzinfo is not None
    assert epoch.year == 2024
    assert epoch.month == 1
    assert epoch.day == 1
    assert epoch.hour == 12


def test_parse_tle_file_and_find_best(tmp_path: Path) -> None:
    tle_file = tmp_path / "25544_2024-01.tle"
    _write_month_file(
        tle_file,
        [
            (
                "1 25544U 98067A   24001.10000000  .00016717  00000-0  10270-3 0  9002",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
            ),
            (
                "1 25544U 98067A   24001.60000000  .00016717  00000-0  10270-3 0  9003",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001",
            ),
        ],
    )

    records = parse_tle_file(tle_file)
    assert len(records) == 2

    target = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    best = find_best_tle_in_file(tle_file, target)
    assert best is not None
    assert best.line1.startswith("1 25544")
    assert best.epoch <= target


def test_resolve_tle_record_from_local_cache(tmp_path: Path) -> None:
    base = tmp_path / "tle"
    current = get_tle_file_path(25544, 2024, 1, base)
    previous = get_tle_file_path(25544, 2023, 12, base)

    _write_month_file(
        previous,
        [
            (
                "1 25544U 98067A   23365.90000000  .00016717  00000-0  10270-3 0  9001",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
            )
        ],
    )
    _write_month_file(
        current,
        [
            (
                "1 25544U 98067A   24001.20000000  .00016717  00000-0  10270-3 0  9002",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001",
            ),
            (
                "1 25544U 98067A   24001.70000000  .00016717  00000-0  10270-3 0  9003",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000002",
            ),
        ],
    )

    query = TLEQuery(
        norad_id=25544,
        target_epoch=datetime(2024, 1, 1, 16, 0, 0, tzinfo=timezone.utc),
        base_dir=base,
        allow_download=False,
    )
    record = resolve_tle_record(query)
    assert record.norad_id == 25544
    assert record.epoch <= query.target_epoch


def test_resolve_tle_spec_from_local_cache(tmp_path: Path) -> None:
    base = tmp_path / "tle"
    current = get_tle_file_path(25544, 2024, 1, base)
    _write_month_file(
        current,
        [
            (
                "1 25544U 98067A   24001.20000000  .00016717  00000-0  10270-3 0  9002",
                "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001",
            )
        ],
    )

    query = TLEQuery(
        norad_id=25544,
        target_epoch=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        base_dir=base,
    )
    spec = resolve_tle_spec(query)
    assert isinstance(spec, TLESpec)
    assert spec.line1.startswith("1 ")
    assert spec.line2.startswith("2 ")


def test_resolve_requires_local_cache_when_download_disabled(tmp_path: Path) -> None:
    query = TLEQuery(
        norad_id=25544,
        target_epoch=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        base_dir=tmp_path / "missing",
        allow_download=False,
    )
    with pytest.raises(FileNotFoundError):
        resolve_tle_record(query)


class _FakeSpaceTrackClient:
    def gp_history(self, **_: object) -> str:
        return (
            "1 25544U 98067A   24001.20000000  .00016717  00000-0  10270-3 0  9002\n"
            "2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000001\n"
        )


def test_download_tles_for_month_writes_cache(tmp_path: Path) -> None:
    result = download_tles_for_month(
        norad_id=25544,
        year=2024,
        month=1,
        space_track_client=_FakeSpaceTrackClient(),
        base_dir=tmp_path,
    )
    assert result.success
    assert result.file_path is not None
    assert result.file_path.exists()
    assert result.tle_count == 1
