"""Ephemeris-based propagator creation from standard file formats.

Supports OEM, OCM, SP3, and CPF ephemeris formats.  Local files are parsed
directly; remote data (SP3 via EDC FTP, CPF via EDC API) is downloaded,
cached, and parsed automatically.

Public API
----------
EphemerisClient     Facade for ephemeris parsing and propagator creation.
EphemerisSpec       Immutable specification for a propagator request.
EphemerisFormat     Enum of supported file formats (OEM, OCM, SP3, CPF).

All parsing and factory functions are available as methods on
``EphemerisClient``.  Direct function imports remain accessible from
submodules for advanced usage.
"""

from astrodyn_core.ephemeris.client import EphemerisClient
from astrodyn_core.ephemeris.models import EphemerisFormat, EphemerisSpec

__all__ = [
    "EphemerisClient",
    "EphemerisSpec",
    "EphemerisFormat",
]
