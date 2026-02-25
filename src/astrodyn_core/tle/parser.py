"""Parsing and selection helpers for two-line element (TLE) files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from astrodyn_core.tle.models import TLERecord


def parse_tle_epoch(tle_line1: str) -> datetime:
    """Parse the TLE epoch field from line 1 into a UTC datetime.

    Args:
        tle_line1: TLE line 1 text containing the epoch field at columns 19-32.

    Returns:
        Parsed timezone-aware UTC epoch.

    Raises:
        ValueError: If line 1 is too short to contain an epoch field.
    """
    if len(tle_line1) < 32:
        raise ValueError("TLE line1 is too short to contain an epoch field.")

    epoch_str = tle_line1[18:32].strip()
    year_2digit = int(epoch_str[0:2])
    year = 2000 + year_2digit if year_2digit < 57 else 1900 + year_2digit
    day_of_year = float(epoch_str[2:])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)


def parse_norad_id(tle_line1: str) -> int:
    """Parse the NORAD catalog identifier from TLE line 1.

    Args:
        tle_line1: TLE line 1 text.

    Returns:
        Parsed NORAD identifier.

    Raises:
        ValueError: If line 1 is too short to contain the NORAD field.
    """
    if len(tle_line1) < 7:
        raise ValueError("TLE line1 is too short to contain NORAD id.")
    return int(tle_line1[2:7].strip())


def parse_tle_file(path: str | Path) -> tuple[TLERecord, ...]:
    """Parse a local ``.tle`` file into sorted TLE records.

    The parser scans the file line-by-line, accepting adjacent line pairs that
    look like TLE line 1 / line 2 records and skipping malformed pairs.

    Args:
        path: Path to a local ``.tle`` file.

    Returns:
        Tuple of parsed records sorted by ascending epoch.
    """
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
    """Select the latest record with epoch less than or equal to ``target_epoch``.

    Args:
        records: Candidate TLE records.
        target_epoch: Desired epoch (naive datetimes are treated as UTC).

    Returns:
        The best matching record, or ``None`` if no record qualifies.
    """
    if target_epoch.tzinfo is None:
        target_epoch = target_epoch.replace(tzinfo=timezone.utc)
    target_epoch = target_epoch.astimezone(timezone.utc)

    best: TLERecord | None = None
    for record in records:
        if record.epoch <= target_epoch and (best is None or record.epoch > best.epoch):
            best = record
    return best


def find_best_tle_in_file(path: str | Path, target_epoch: datetime) -> TLERecord | None:
    """Parse a file and select the best record for a target epoch.

    Args:
        path: Path to a local ``.tle`` file.
        target_epoch: Desired epoch (naive datetimes are treated as UTC).

    Returns:
        The best matching record, or ``None`` if no record qualifies.
    """
    return find_best_tle(parse_tle_file(path), target_epoch)
