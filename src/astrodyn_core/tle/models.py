"""TLE domain models for retrieval and selection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TLERecord:
    """Parsed two-line element set with normalized epoch metadata.

    Attributes:
        line1: TLE line 1 text (must start with ``"1 "``).
        line2: TLE line 2 text (must start with ``"2 "``).
        norad_id: NORAD catalog identifier parsed from line 1.
        epoch: TLE epoch normalized to timezone-aware UTC.
    """

    line1: str
    line2: str
    norad_id: int
    epoch: datetime

    def __post_init__(self) -> None:
        if not self.line1.startswith("1 "):
            raise ValueError("TLERecord.line1 must start with '1 '.")
        if not self.line2.startswith("2 "):
            raise ValueError("TLERecord.line2 must start with '2 '.")
        if self.norad_id <= 0:
            raise ValueError("TLERecord.norad_id must be positive.")
        epoch = self.epoch
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "epoch", epoch.astimezone(timezone.utc))

    def as_two_line_string(self) -> str:
        """Render the TLE as a standard two-line string block.

        Returns:
            The original TLE lines separated by a newline.
        """
        return f"{self.line1}\n{self.line2}"


@dataclass(frozen=True, slots=True)
class TLEQuery:
    """Input query for local/download-backed TLE resolution.

    Attributes:
        norad_id: NORAD catalog identifier to resolve.
        target_epoch: Desired propagation epoch, normalized to UTC.
        base_dir: Root directory of the monthly TLE cache hierarchy.
        allow_download: Whether missing cache files may be downloaded from
            Space-Track.
    """

    norad_id: int
    target_epoch: datetime
    base_dir: str | Path = "data/tle"
    allow_download: bool = False

    def __post_init__(self) -> None:
        if self.norad_id <= 0:
            raise ValueError("TLEQuery.norad_id must be positive.")
        epoch = self.target_epoch
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "target_epoch", epoch.astimezone(timezone.utc))
        object.__setattr__(self, "base_dir", Path(self.base_dir))


@dataclass(frozen=True, slots=True)
class TLEDownloadResult:
    """Result of one monthly TLE download attempt.

    Attributes:
        norad_id: NORAD catalog identifier requested.
        year: Requested UTC year.
        month: Requested UTC month.
        success: Whether the download attempt produced a usable cache file.
        file_path: Local cache file path when download succeeded.
        message: Human-readable status or error message.
        tle_count: Number of TLE sets written to the cache file.
    """

    norad_id: int
    year: int
    month: int
    success: bool
    file_path: Path | None = None
    message: str = ""
    tle_count: int = 0
