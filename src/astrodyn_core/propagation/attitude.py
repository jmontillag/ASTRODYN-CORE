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

    Attributes:
        mode: Predefined attitude mode (``qsw``, ``vvlh``, ``tnw``, ``nadir``,
            ``inertial``). Ignored when ``provider`` is supplied.
        provider: Optional pre-built Orekit ``AttitudeProvider`` instance used
            as a pass-through escape hatch for custom attitude laws.
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
