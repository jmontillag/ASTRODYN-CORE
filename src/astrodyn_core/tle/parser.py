"""Parsing and selection helpers for two-line element (TLE) files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from astrodyn_core.tle.models import TLERecord


def parse_tle_epoch(tle_line1: str) -> datetime:
    """Parse TLE epoch (line1 columns 19-32) into UTC datetime."""
    if len(tle_line1) < 32:
        raise ValueError("TLE line1 is too short to contain an epoch field.")

    epoch_str = tle_line1[18:32].strip()
    year_2digit = int(epoch_str[0:2])
    year = 2000 + year_2digit if year_2digit < 57 else 1900 + year_2digit
    day_of_year = float(epoch_str[2:])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def parse_norad_id(tle_line1: str) -> int:
    """Parse NORAD id from line1 columns 3-7."""
    if len(tle_line1) < 7:
        raise ValueError("TLE line1 is too short to contain NORAD id.")
    return int(tle_line1[2:7].strip())


def parse_tle_file(path: str | Path) -> tuple[TLERecord, ...]:
    """Read a .tle file and return sorted TLE records (ascending by epoch)."""
    file_path = Path(path)
    lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]

    records: list[TLERecord] = []
    idx = 0
    while idx < len(lines) - 1:
        line1 = lines[idx]
        line2 = lines[idx + 1]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                records.append(
                    TLERecord(
                        line1=line1,
                        line2=line2,
                        norad_id=parse_norad_id(line1),
                        epoch=parse_tle_epoch(line1),
                    )
                )
                idx += 2
                continue
            except (ValueError, IndexError):
                pass
        idx += 1

    records.sort(key=lambda item: item.epoch)
    return tuple(records)


def find_best_tle(records: Iterable[TLERecord], target_epoch: datetime) -> TLERecord | None:
    """Return latest record with epoch <= target_epoch."""
    if target_epoch.tzinfo is None:
        target_epoch = target_epoch.replace(tzinfo=timezone.utc)
    target_epoch = target_epoch.astimezone(timezone.utc)

    best: TLERecord | None = None
    for record in records:
        if record.epoch <= target_epoch and (best is None or record.epoch > best.epoch):
            best = record
    return best


def find_best_tle_in_file(path: str | Path, target_epoch: datetime) -> TLERecord | None:
    """Parse a file and return latest record with epoch <= target_epoch."""
    return find_best_tle(parse_tle_file(path), target_epoch)
