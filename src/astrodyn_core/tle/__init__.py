"""TLE retrieval, parsing, and resolution helpers.

This module intentionally keeps network/cache logic separate from propagation
providers. Use resolver outputs to feed existing ``PropagatorSpec(kind='tle')``.
"""

from astrodyn_core.tle.client import TLEClient
from astrodyn_core.tle.downloader import (
    download_tles_for_month,
    ensure_tles_available,
    get_tle_file_path,
)
from astrodyn_core.tle.models import TLEDownloadResult, TLEQuery, TLERecord
from astrodyn_core.tle.parser import (
    find_best_tle,
    find_best_tle_in_file,
    parse_norad_id,
    parse_tle_epoch,
    parse_tle_file,
)
from astrodyn_core.tle.resolver import resolve_tle_record, resolve_tle_spec

__all__ = [
    "TLEClient",
    "TLEDownloadResult",
    "TLEQuery",
    "TLERecord",
    "download_tles_for_month",
    "ensure_tles_available",
    "find_best_tle",
    "find_best_tle_in_file",
    "get_tle_file_path",
    "parse_norad_id",
    "parse_tle_epoch",
    "parse_tle_file",
    "resolve_tle_record",
    "resolve_tle_spec",
]
