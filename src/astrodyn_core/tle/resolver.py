"""High-level TLE resolution: query -> TLERecord/TLESpec."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from typing import Any

from astrodyn_core.propagation.specs import TLESpec
from astrodyn_core.tle.downloader import ensure_tles_available, get_tle_file_path
from astrodyn_core.tle.models import TLEQuery, TLERecord
from astrodyn_core.tle.parser import find_best_tle_in_file


def _candidate_month_paths(query: TLEQuery) -> tuple[Path, ...]:
    target = query.target_epoch
    year, month = target.year, target.month

    current = get_tle_file_path(query.norad_id, year, month, query.base_dir)
    if month == 1:
        previous = get_tle_file_path(query.norad_id, year - 1, 12, query.base_dir)
    else:
        previous = get_tle_file_path(query.norad_id, year, month - 1, query.base_dir)
    return previous, current


def resolve_tle_record(query: TLEQuery, *, space_track_client: Any | None = None) -> TLERecord:
    """Resolve best TLE record for a NORAD id and target epoch.

    If ``query.allow_download`` is True, an authenticated Space-Track client must
    be provided and missing cache files will be downloaded.
    """
    if query.allow_download:
        if space_track_client is None:
            raise ValueError("space_track_client is required when allow_download=True.")
        ensure_tles_available(
            query.norad_id,
            query.target_epoch,
            space_track_client,
            base_dir=query.base_dir,
        )

    candidate_files = [path for path in _candidate_month_paths(query) if path.exists()]
    if not candidate_files:
        raise FileNotFoundError(
            f"No TLE cache files found for NORAD {query.norad_id} around {query.target_epoch.isoformat()} "
            f"in '{query.base_dir}'."
        )

    best: TLERecord | None = None
    for file_path in candidate_files:
        record = find_best_tle_in_file(file_path, query.target_epoch)
        if record is not None and (best is None or record.epoch > best.epoch):
            best = record

    if best is None:
        raise RuntimeError(
            f"No TLE record found for NORAD {query.norad_id} before {query.target_epoch.isoformat()}."
        )
    return best


def resolve_tle_spec(query: TLEQuery, *, space_track_client: Any | None = None) -> TLESpec:
    """Resolve a query into the existing propagation-layer ``TLESpec``."""
    record = resolve_tle_record(query, space_track_client=space_track_client)
    return TLESpec(line1=record.line1, line2=record.line2)
