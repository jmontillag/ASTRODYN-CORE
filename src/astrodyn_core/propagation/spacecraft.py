"""Spacecraft physical model specification.

Defines the physical properties of the spacecraft that affect drag and solar
radiation pressure force models.  Supports both a simple isotropic model and
a detailed box-and-solar-array model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SpacecraftSpec:
    """Physical spacecraft model used by drag and SRP assembly.

    When ``use_box_wing=False`` (default), the isotropic parameters are used:
    ``drag_area``, ``drag_coeff``, ``srp_area``, ``srp_coeff``.

    When ``use_box_wing=True``, the box-and-solar-array geometry is used
    instead.  The isotropic parameters are ignored in that case.
    """

    mass: float = 1000.0

    # --- Isotropic model ---
    drag_area: float = 10.0
    drag_coeff: float = 2.2
    srp_area: float = 10.0
    srp_coeff: float = 1.5

    # --- Box-and-solar-array model ---
    use_box_wing: bool = False
    x_length: float = 1.0
    y_length: float = 1.0
    z_length: float = 1.0
    solar_array_area: float = 20.0
    solar_array_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    box_drag_coeff: float = 2.2
    box_lift_coeff: float = 0.0
    box_abs_coeff: float = 0.7
    box_ref_coeff: float = 0.3

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("SpacecraftSpec.mass must be positive.")
        if self.drag_area < 0:
            raise ValueError("SpacecraftSpec.drag_area cannot be negative.")
        if self.srp_area < 0:
            raise ValueError("SpacecraftSpec.srp_area cannot be negative.")
        if not (0 <= self.box_abs_coeff <= 1.0):
            raise ValueError("SpacecraftSpec.box_abs_coeff must be in [0, 1].")
        if not (0 <= self.box_ref_coeff <= 1.0):
            raise ValueError("SpacecraftSpec.box_ref_coeff must be in [0, 1].")

        # Normalize solar_array_axis to a unit vector
        ax, ay, az = self.solar_array_axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 1e-9:
            normalized = (ax / norm, ay / norm, az / norm)
            object.__setattr__(self, "solar_array_axis", normalized)
