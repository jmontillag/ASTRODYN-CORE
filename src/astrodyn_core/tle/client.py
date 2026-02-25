"""High-level client API for TLE cache, parsing, and resolution workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from astrodyn_core.propagation.specs import TLESpec
from astrodyn_core.tle.downloader import (
    download_tles_for_month as _download_tles_for_month,
    ensure_tles_available as _ensure_tles_available,
    get_tle_file_path as _get_tle_file_path,
)
from astrodyn_core.tle.models import TLEDownloadResult, TLEQuery, TLERecord
from astrodyn_core.tle.parser import (
    find_best_tle as _find_best_tle,
    find_best_tle_in_file as _find_best_tle_in_file,
    parse_norad_id as _parse_norad_id,
    parse_tle_epoch as _parse_tle_epoch,
    parse_tle_file as _parse_tle_file,
)
from astrodyn_core.tle.resolver import (
    resolve_tle_record as _resolve_tle_record,
    resolve_tle_spec as _resolve_tle_spec,
)


@dataclass(slots=True)
class TLEClient:
    """Facade for TLE cache, parsing, and resolution workflows.

    Args:
        base_dir: Root directory for monthly cached TLE files.
        allow_download: Default policy for downloading missing cache files.
        space_track_client: Optional authenticated Space-Track client used for
            download-enabled workflows.
    """

    base_dir: str | Path = "data/tle"
    allow_download: bool = False
    space_track_client: Any | None = None

    def build_query(
        self,
        norad_id: int,
        target_epoch: datetime,
        *,
        base_dir: str | Path | None = None,
        allow_download: bool | None = None,
    ) -> TLEQuery:
        """Build a normalized :class:`TLEQuery` from raw inputs.

        Args:
            norad_id: NORAD catalog identifier.
            target_epoch: Epoch to resolve (naive datetimes are treated as UTC).
            base_dir: Optional cache root override for this query.
            allow_download: Optional download-policy override for this query.

        Returns:
            A validated query object with normalized UTC epoch and ``Path``
            cache root.
        """
        return TLEQuery(
            norad_id=norad_id,
            target_epoch=target_epoch,
            base_dir=self._resolve_base_dir(base_dir),
            allow_download=self._resolve_allow_download(allow_download),
        )

    def get_tle_file_path(
        self,
        norad_id: int,
        year: int,
        month: int,
        *,
        base_dir: str | Path | None = None,
    ) -> Path:
        """Return the canonical monthly TLE cache path.

        Args:
            norad_id: NORAD catalog identifier.
            year: UTC year.
            month: UTC month in ``[1, 12]``.
            base_dir: Optional cache root override.

        Returns:
            Canonical file path ``{base}/{norad}/{norad}_YYYY-MM.tle``.
        """
        return _get_tle_file_path(norad_id, year, month, self._resolve_base_dir(base_dir))

    def download_tles_for_month(
        self,
        norad_id: int,
        year: int,
        month: int,
        *,
        space_track_client: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> TLEDownloadResult:
        """Download one month of TLE history into the local cache.

        Args:
            norad_id: NORAD catalog identifier.
            year: UTC year.
            month: UTC month in ``[1, 12]``.
            space_track_client: Optional client override. Falls back to the
                client stored on this instance.
            base_dir: Optional cache root override.

        Returns:
            A structured download result describing success/failure and output.
        """
        return _download_tles_for_month(
            norad_id,
            year,
            month,
            self._resolve_space_track_client(space_track_client),
            base_dir=self._resolve_base_dir(base_dir),
        )

    def ensure_tles_available(
        self,
        norad_id: int,
        target_epoch: datetime,
        *,
        space_track_client: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> tuple[Path, ...]:
        """Ensure the required monthly cache files exist for a target epoch.

        The current month is required. The previous month is also included when
        the target day is early in the month, to avoid missing the latest TLE
        before the requested epoch.

        Args:
            norad_id: NORAD catalog identifier.
            target_epoch: Epoch to resolve.
            space_track_client: Optional client override for downloads.
            base_dir: Optional cache root override.

        Returns:
            Tuple of existing local monthly cache file paths.
        """
        return _ensure_tles_available(
            norad_id,
            target_epoch,
            self._resolve_space_track_client(space_track_client),
            base_dir=self._resolve_base_dir(base_dir),
        )

    def parse_tle_epoch(self, tle_line1: str) -> datetime:
        """Parse a TLE line-1 epoch field into a UTC datetime.

        Args:
            tle_line1: TLE line 1 text.

        Returns:
            Parsed timezone-aware UTC epoch.
        """
        return _parse_tle_epoch(tle_line1)

    def parse_norad_id(self, tle_line1: str) -> int:
        """Parse the NORAD catalog identifier from TLE line 1.

        Args:
            tle_line1: TLE line 1 text.

        Returns:
            Parsed NORAD ID.
        """
        return _parse_norad_id(tle_line1)

    def parse_tle_file(self, path: str | Path) -> tuple[TLERecord, ...]:
        """Parse a local ``.tle`` file into sorted records.

        Args:
            path: Local file path.

        Returns:
            Parsed records sorted by ascending epoch.
        """
        return _parse_tle_file(path)

    def find_best_tle(self, records: Iterable[TLERecord], target_epoch: datetime) -> TLERecord | None:
        """Select the latest TLE record at or before a target epoch.

        Args:
            records: Candidate records (order does not matter).
            target_epoch: Desired epoch.

        Returns:
            The best matching record, or ``None`` if no record qualifies.
        """
        return _find_best_tle(records, target_epoch)

    def find_best_tle_in_file(self, path: str | Path, target_epoch: datetime) -> TLERecord | None:
        """Parse a file and select the latest record at or before a target epoch.

        Args:
            path: Local ``.tle`` file path.
            target_epoch: Desired epoch.

        Returns:
            The best matching record, or ``None`` if no record qualifies.
        """
        return _find_best_tle_in_file(path, target_epoch)

    def resolve_tle_record(
        self,
        query: TLEQuery,
        *,
        space_track_client: Any | None = None,
    ) -> TLERecord:
        """Resolve a query to a concrete :class:`TLERecord`.

        Args:
            query: TLE query descriptor.
            space_track_client: Optional client override used if downloads are
                enabled on the query.

        Returns:
            The best TLE record with epoch less than or equal to the query epoch.

        Raises:
            ValueError: If downloads are enabled but no Space-Track client is
                available.
            FileNotFoundError: If required cache files are missing and downloads
                are disabled (or fail to produce files).
            RuntimeError: If cache files exist but no qualifying record is found.
        """
        return _resolve_tle_record(
            query,
            space_track_client=self._resolve_space_track_client(
                space_track_client,
                required=query.allow_download,
            ),
        )

    def resolve_tle_spec(
        self,
        query: TLEQuery,
        *,
        space_track_client: Any | None = None,
    ) -> TLESpec:
        """Resolve a query into a propagation-layer :class:`TLESpec`.

        Args:
            query: TLE query descriptor.
            space_track_client: Optional client override used if downloads are
                enabled on the query.

        Returns:
            A propagation-ready ``TLESpec`` built from the resolved record.
        """
        return _resolve_tle_spec(
            query,
            space_track_client=self._resolve_space_track_client(
                space_track_client,
                required=query.allow_download,
            ),
        )

    def resolve_tle_record_for_epoch(
        self,
        norad_id: int,
        target_epoch: datetime,
        *,
        base_dir: str | Path | None = None,
        allow_download: bool | None = None,
        space_track_client: Any | None = None,
    ) -> TLERecord:
        """Convenience wrapper: build a query and resolve a TLE record.

        Args:
            norad_id: NORAD catalog identifier.
            target_epoch: Desired epoch.
            base_dir: Optional cache root override.
            allow_download: Optional download-policy override.
            space_track_client: Optional client override for download-enabled
                queries.

        Returns:
            Resolved best TLE record.
        """
        query = self.build_query(
            norad_id,
            target_epoch,
            base_dir=base_dir,
            allow_download=allow_download,
        )
        return self.resolve_tle_record(query, space_track_client=space_track_client)

    def resolve_tle_spec_for_epoch(
        self,
        norad_id: int,
        target_epoch: datetime,
        *,
        base_dir: str | Path | None = None,
        allow_download: bool | None = None,
        space_track_client: Any | None = None,
    ) -> TLESpec:
        """Convenience wrapper: build a query and resolve a propagation TLE spec.

        Args:
            norad_id: NORAD catalog identifier.
            target_epoch: Desired epoch.
            base_dir: Optional cache root override.
            allow_download: Optional download-policy override.
            space_track_client: Optional client override for download-enabled
                queries.

        Returns:
            A propagation-layer ``TLESpec``.
        """
        query = self.build_query(
            norad_id,
            target_epoch,
            base_dir=base_dir,
            allow_download=allow_download,
        )
        return self.resolve_tle_spec(query, space_track_client=space_track_client)

    def _resolve_base_dir(self, base_dir: str | Path | None) -> Path:
        """Resolve a per-call cache root override against the client default."""
        return Path(base_dir) if base_dir is not None else Path(self.base_dir)

    def _resolve_allow_download(self, allow_download: bool | None) -> bool:
        """Resolve a per-call download policy override against the client default."""
        if allow_download is None:
            return bool(self.allow_download)
        return bool(allow_download)

    def _resolve_space_track_client(self, space_track_client: Any | None, *, required: bool = True) -> Any:
        """Resolve a per-call Space-Track client override.

        Args:
            space_track_client: Optional client override.
            required: Whether to raise if no client is available.

        Returns:
            The resolved client object, or ``None`` when not required.

        Raises:
            ValueError: If ``required`` is true and no client is available.
        """
        resolved = space_track_client if space_track_client is not None else self.space_track_client
        if required and resolved is None:
            raise ValueError(
                "space_track_client is required for this operation. "
                "Provide it as an argument or set TLEClient(space_track_client=...)."
            )
        return resolved
