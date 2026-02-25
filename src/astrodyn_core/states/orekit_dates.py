"""Date conversion helpers between state-file epochs and Orekit AbsoluteDate."""

from __future__ import annotations

from datetime import timezone
from typing import Any

from astrodyn_core.states.validation import parse_epoch_utc


def to_orekit_date(epoch: str):
    """Convert an ISO-8601 epoch string into Orekit ``AbsoluteDate`` (UTC).

    Args:
        epoch: ISO-8601 UTC epoch string.

    Returns:
        Orekit ``AbsoluteDate``.

    Raises:
        RuntimeError: If Orekit Python helpers are unavailable.
    """
    parsed = parse_epoch_utc(epoch)
    try:
        from orekit.pyhelpers import datetime_to_absolutedate
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    return datetime_to_absolutedate(parsed)


def from_orekit_date(date: Any) -> str:
    """Convert Orekit ``AbsoluteDate`` into an ISO-8601 UTC epoch string.

    Args:
        date: Orekit ``AbsoluteDate`` instance.

    Returns:
        ISO-8601 UTC epoch string with ``Z`` suffix.

    Raises:
        RuntimeError: If Orekit Python helpers are unavailable.
    """
    try:
        from orekit.pyhelpers import absolutedate_to_datetime
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    parsed = absolutedate_to_datetime(date, tz_aware=True)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
