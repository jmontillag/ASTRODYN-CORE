"""Declarative force model specifications.

Each spec represents a single force model that can be assembled into an
Orekit ForceModel via the assembly module.  Specs are pure data — no Orekit
imports happen at module level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

# ---------------------------------------------------------------------------
# Valid body names accepted by ThirdBodySpec
# ---------------------------------------------------------------------------
SUPPORTED_THIRD_BODIES = frozenset(
    {
        "sun",
        "moon",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "mercury",
        "uranus",
        "neptune",
    }
)

SUPPORTED_ATMOSPHERE_MODELS = frozenset(
    {
        "nrlmsise00",
        "dtm2000",
        "harrispriester",
        "jb2008",
        "simpleexponential",
    }
)

SUPPORTED_SPACE_WEATHER_SOURCES = frozenset({"cssi", "msafe"})

SUPPORTED_SOLAR_ACTIVITY_STRENGTHS = frozenset({"weak", "average", "strong"})


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GravitySpec:
    """Earth gravity field model.

    Set degree=order=0 for a simple point-mass (Keplerian) model.
    """

    degree: int = 0
    order: int = 0
    normalized: bool = True

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("GravitySpec.degree must be >= 0.")
        if self.order < 0:
            raise ValueError("GravitySpec.order must be >= 0.")
        if self.order > self.degree:
            raise ValueError("GravitySpec.order cannot exceed degree.")


# ---------------------------------------------------------------------------
# Atmospheric drag
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DragSpec:
    """Atmospheric drag force model.

    Parameters for SimpleExponential atmosphere (ref_rho, ref_alt, scale_height)
    are only required when ``atmosphere_model="simpleexponential"``.

    ``space_weather_source`` and ``solar_activity_strength`` apply to models
    that consume space-weather data (NRLMSISE00, DTM2000, JB2008).
    """

    atmosphere_model: str = "nrlmsise00"

    # Space-weather configuration (for NRLMSISE00 / DTM2000 / JB2008)
    space_weather_source: str = "cssi"
    solar_activity_strength: str = "average"
    space_weather_data: str = "default"

    # SimpleExponential atmosphere parameters
    ref_rho: float | None = None
    ref_alt: float | None = None
    scale_height: float | None = None

    def __post_init__(self) -> None:
        model = self.atmosphere_model.strip().lower()
        object.__setattr__(self, "atmosphere_model", model)

        source = self.space_weather_source.strip().lower()
        object.__setattr__(self, "space_weather_source", source)

        strength = self.solar_activity_strength.strip().lower()
        object.__setattr__(self, "solar_activity_strength", strength)

        if model not in SUPPORTED_ATMOSPHERE_MODELS:
            raise ValueError(
                f"Unsupported atmosphere_model '{model}'. "
                f"Supported: {sorted(SUPPORTED_ATMOSPHERE_MODELS)}"
            )

        if model == "simpleexponential":
            if self.ref_rho is None or self.ref_alt is None or self.scale_height is None:
                raise ValueError(
                    "ref_rho, ref_alt, and scale_height are required "
                    "for atmosphere_model='simpleexponential'."
                )
            if self.scale_height <= 0:
                raise ValueError("scale_height must be positive.")

        if source not in SUPPORTED_SPACE_WEATHER_SOURCES:
            raise ValueError(
                f"Unsupported space_weather_source '{source}'. "
                f"Supported: {sorted(SUPPORTED_SPACE_WEATHER_SOURCES)}"
            )

        if strength not in SUPPORTED_SOLAR_ACTIVITY_STRENGTHS:
            raise ValueError(
                f"Unsupported solar_activity_strength '{strength}'. "
                f"Supported: {sorted(SUPPORTED_SOLAR_ACTIVITY_STRENGTHS)}"
            )


# ---------------------------------------------------------------------------
# Solar radiation pressure
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SRPSpec:
    """Solar radiation pressure force model."""

    enable_moon_eclipse: bool = False
    enable_albedo: bool = False


# ---------------------------------------------------------------------------
# Third-body gravity
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ThirdBodySpec:
    """Third-body gravitational attractions.

    ``bodies`` is a sequence of lowercase celestial body names.
    Common choices: ``["sun", "moon"]``.
    """

    bodies: tuple[str, ...] = ("sun", "moon")

    def __post_init__(self) -> None:
        normalized = tuple(b.strip().lower() for b in self.bodies)
        object.__setattr__(self, "bodies", normalized)
        for body in normalized:
            if body not in SUPPORTED_THIRD_BODIES:
                raise ValueError(
                    f"Unsupported third body '{body}'. Supported: {sorted(SUPPORTED_THIRD_BODIES)}"
                )


# ---------------------------------------------------------------------------
# Relativistic correction
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RelativitySpec:
    """Relativistic correction (Schwarzschild effect).

    Presence of this spec in the force list enables the correction.
    """


# ---------------------------------------------------------------------------
# Tidal effects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SolidTidesSpec:
    """Solid Earth tides perturbation."""


@dataclass(frozen=True, slots=True)
class OceanTidesSpec:
    """Ocean tides perturbation."""

    degree: int = 5
    order: int = 5

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("OceanTidesSpec.degree must be >= 0.")
        if self.order < 0:
            raise ValueError("OceanTidesSpec.order must be >= 0.")


# ---------------------------------------------------------------------------
# Union type for type hints
# ---------------------------------------------------------------------------
ForceSpec = Union[
    GravitySpec,
    DragSpec,
    SRPSpec,
    ThirdBodySpec,
    RelativitySpec,
    SolidTidesSpec,
    OceanTidesSpec,
]
