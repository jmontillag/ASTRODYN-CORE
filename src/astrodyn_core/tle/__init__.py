"""TLE retrieval, parsing, and resolution helpers.

This module intentionally keeps network/cache logic separate from propagation
providers. Use resolver outputs to feed existing ``PropagatorSpec(kind='tle')``.

Public API
----------
TLEClient              Facade for TLE retrieval, parsing, and resolution.
TLEQuery               Query descriptor for TLE resolution.
TLERecord              Parsed TLE record with metadata.
TLEDownloadResult      Result of a TLE download operation.

All TLE workflow functions (download, parse, resolve) are available as methods
on ``TLEClient``.  Direct function imports remain accessible from submodules
for advanced usage.
"""

from astrodyn_core.tle.client import TLEClient
from astrodyn_core.tle.models import TLEDownloadResult, TLEQuery, TLERecord

__all__ = [
    "TLEClient",
    "TLEQuery",
    "TLERecord",
    "TLEDownloadResult",
]
