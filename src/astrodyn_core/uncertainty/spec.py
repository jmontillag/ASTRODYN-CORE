"""Uncertainty propagation configuration spec."""

from __future__ import annotations

from dataclasses import dataclass, field


_VALID_METHODS = {"stm"}
_VALID_ORBIT_TYPES = {"CARTESIAN", "KEPLERIAN", "EQUINOCTIAL"}
_VALID_POSITION_ANGLES = {"MEAN", "TRUE", "ECCENTRIC"}


@dataclass(frozen=True, slots=True)
class UncertaintySpec:
    """Configuration for covariance/uncertainty propagation.

    method:
      - ``"stm"`` (default): State Transition Matrix method using Orekit's
        ``setupMatricesComputation``. Linear approximation; fast and accurate
        for short arcs or near-linear dynamics.

    stm_name:
        Internal name for the STM additional state. Only used with method="stm".

    include_mass:
        If True, extend the STM to 7×7 (state + mass). The initial covariance
        must then be 7×7. If False (default), 6×6 state-only covariance.

    orbit_type:
        Orbit element type for the STM state vector:
        ``"CARTESIAN"`` (default), ``"KEPLERIAN"``, or ``"EQUINOCTIAL"``.

    position_angle:
        Position angle convention used for non-Cartesian orbit types.
        ``"MEAN"`` (default), ``"TRUE"``, or ``"ECCENTRIC"``.
    """

    method: str = "stm"
    stm_name: str = "stm"
    include_mass: bool = False
    orbit_type: str = "CARTESIAN"
    position_angle: str = "MEAN"

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"UncertaintySpec.method must be one of {_VALID_METHODS!r}, got {self.method!r}."
            )
        if not self.stm_name.strip():
            raise ValueError("UncertaintySpec.stm_name cannot be empty.")
        orbit_type_upper = self.orbit_type.strip().upper()
        if orbit_type_upper not in _VALID_ORBIT_TYPES:
            raise ValueError(
                f"UncertaintySpec.orbit_type must be one of {_VALID_ORBIT_TYPES!r}, "
                f"got {self.orbit_type!r}."
            )
        object.__setattr__(self, "orbit_type", orbit_type_upper)
        pa_upper = self.position_angle.strip().upper()
        if pa_upper not in _VALID_POSITION_ANGLES:
            raise ValueError(
                f"UncertaintySpec.position_angle must be one of {_VALID_POSITION_ANGLES!r}, "
                f"got {self.position_angle!r}."
            )
        object.__setattr__(self, "position_angle", pa_upper)

    @property
    def state_dimension(self) -> int:
        """State vector dimension: 6 (orbit only) or 7 (orbit + mass)."""
        return 7 if self.include_mass else 6

    @classmethod
    def from_mapping(cls, data: dict) -> UncertaintySpec:
        return cls(
            method=str(data.get("method", "stm")),
            stm_name=str(data.get("stm_name", "stm")),
            include_mass=bool(data.get("include_mass", False)),
            orbit_type=str(data.get("orbit_type", "CARTESIAN")),
            position_angle=str(data.get("position_angle", "MEAN")),
        )

    def to_mapping(self) -> dict:
        return {
            "method": self.method,
            "stm_name": self.stm_name,
            "include_mass": self.include_mass,
            "orbit_type": self.orbit_type,
            "position_angle": self.position_angle,
        }
