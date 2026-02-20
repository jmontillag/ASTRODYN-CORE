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
    """Single entrypoint for TLE retrieval, parsing, and query resolution."""

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
        return _ensure_tles_available(
            norad_id,
            target_epoch,
            self._resolve_space_track_client(space_track_client),
            base_dir=self._resolve_base_dir(base_dir),
        )

    def parse_tle_epoch(self, tle_line1: str):
        return _parse_tle_epoch(tle_line1)

    def parse_norad_id(self, tle_line1: str) -> int:
        return _parse_norad_id(tle_line1)

    def parse_tle_file(self, path: str | Path) -> tuple[TLERecord, ...]:
        return _parse_tle_file(path)

    def find_best_tle(self, records: Iterable[TLERecord], target_epoch: datetime) -> TLERecord | None:
        return _find_best_tle(records, target_epoch)

    def find_best_tle_in_file(self, path: str | Path, target_epoch: datetime) -> TLERecord | None:
        return _find_best_tle_in_file(path, target_epoch)

    def resolve_tle_record(
        self,
        query: TLEQuery,
        *,
        space_track_client: Any | None = None,
    ) -> TLERecord:
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
        query = self.build_query(
            norad_id,
            target_epoch,
            base_dir=base_dir,
            allow_download=allow_download,
        )
        return self.resolve_tle_spec(query, space_track_client=space_track_client)

    def _resolve_base_dir(self, base_dir: str | Path | None) -> Path:
        return Path(base_dir) if base_dir is not None else Path(self.base_dir)

    def _resolve_allow_download(self, allow_download: bool | None) -> bool:
        if allow_download is None:
            return bool(self.allow_download)
        return bool(allow_download)

    def _resolve_space_track_client(self, space_track_client: Any | None, *, required: bool = True) -> Any:
        resolved = space_track_client if space_track_client is not None else self.space_track_client
        if required and resolved is None:
            raise ValueError(
                "space_track_client is required for this operation. "
                "Provide it as an argument or set TLEClient(space_track_client=...)."
            )
        return resolved
