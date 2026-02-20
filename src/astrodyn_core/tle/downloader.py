"""Download/cache utilities for monthly TLE files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrodyn_core.tle.models import TLEDownloadResult


def get_tle_file_path(norad_id: int, year: int, month: int, base_dir: str | Path = "data/tle") -> Path:
    """Return canonical monthly cache path: {base}/{norad}/{norad}_YYYY-MM.tle."""
    return Path(base_dir) / str(norad_id) / f"{norad_id}_{year:04d}-{month:02d}.tle"


def download_tles_for_month(
    norad_id: int,
    year: int,
    month: int,
    space_track_client: Any,
    *,
    base_dir: str | Path = "data/tle",
) -> TLEDownloadResult:
    """Download one month of TLEs via an authenticated Space-Track client.

    The client must implement: ``gp_history(norad_cat_id=..., epoch=..., orderby=..., format='tle')``.
    """
    if norad_id <= 0:
        raise ValueError("norad_id must be positive.")
    if month < 1 or month > 12:
        raise ValueError("month must be in [1, 12].")

    start_str = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end_str = f"{year + 1:04d}-01-01"
    else:
        end_str = f"{year:04d}-{month + 1:02d}-01"

    try:
        payload = space_track_client.gp_history(
            norad_cat_id=norad_id,
            epoch=f">{start_str},<{end_str}",
            orderby="epoch asc",
            format="tle",
        )
    except Exception as exc:
        return TLEDownloadResult(
            norad_id=norad_id,
            year=year,
            month=month,
            success=False,
            message=f"Space-Track request failed: {exc}",
        )

    text = str(payload or "").strip()
    if not text:
        return TLEDownloadResult(
            norad_id=norad_id,
            year=year,
            month=month,
            success=False,
            message="No TLE records returned.",
        )

    out_path = get_tle_file_path(norad_id, year, month, base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")

    tle_count = len([line for line in text.splitlines() if line.strip()]) // 2
    return TLEDownloadResult(
        norad_id=norad_id,
        year=year,
        month=month,
        success=True,
        file_path=out_path,
        message=f"Downloaded {tle_count} TLEs.",
        tle_count=tle_count,
    )


def ensure_tles_available(
    norad_id: int,
    target_epoch: datetime,
    space_track_client: Any,
    *,
    base_dir: str | Path = "data/tle",
) -> tuple[Path, ...]:
    """Ensure current month cache exists; include previous month when target day <= 8."""
    if target_epoch.tzinfo is None:
        target_epoch = target_epoch.replace(tzinfo=timezone.utc)
    target_epoch = target_epoch.astimezone(timezone.utc)

    year = target_epoch.year
    month = target_epoch.month

    candidate_months: list[tuple[int, int]] = [(year, month)]
    if target_epoch.day <= 8:
        if month == 1:
            candidate_months.append((year - 1, 12))
        else:
            candidate_months.append((year, month - 1))

    resolved_paths: list[Path] = []
    for y, m in candidate_months:
        file_path = get_tle_file_path(norad_id, y, m, base_dir)
        if not file_path.exists():
            result = download_tles_for_month(norad_id, y, m, space_track_client, base_dir=base_dir)
            if not result.success and (y, m) == (year, month):
                raise RuntimeError(
                    f"Failed to download required TLE month {y:04d}-{m:02d} for NORAD {norad_id}: "
                    f"{result.message}"
                )
        if file_path.exists():
            resolved_paths.append(file_path)

    return tuple(resolved_paths)
