"""Declarative attitude specification.

Supports common predefined attitude modes as well as pass-through for
user-supplied Orekit ``AttitudeProvider`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_ATTITUDE_MODES = frozenset(
    {
        "qsw",
        "vvlh",
        "tnw",
        "nadir",
        "inertial",
    }
)


@dataclass(frozen=True, slots=True)
class AttitudeSpec:
    """Attitude configuration for propagation.

    Parameters
    ----------
    mode : str
        One of the predefined attitude modes: ``"qsw"`` / ``"vvlh"``
        (Local Vertical Local Horizontal), ``"tnw"`` (Tangential Normal),
        ``"nadir"`` (Nadir pointing), ``"inertial"`` (fixed inertial frame).
    provider : Any, optional
        A pre-built Orekit ``AttitudeProvider`` instance.  When set, ``mode``
        is ignored and this provider is used directly.  This is the escape
        hatch for custom attitude laws not covered by the predefined modes.
    """

    mode: str = "inertial"
    provider: Any | None = None

    def __post_init__(self) -> None:
        normalized = self.mode.strip().lower()
        object.__setattr__(self, "mode", normalized)

        # Only validate mode when no custom provider is given
        if self.provider is None and normalized not in SUPPORTED_ATTITUDE_MODES:
            raise ValueError(
                f"Unsupported attitude mode '{normalized}'. "
                f"Supported: {sorted(SUPPORTED_ATTITUDE_MODES)}. "
                f"Or pass a custom AttitudeProvider via the 'provider' parameter."
            )
