"""Orekit assembly — translates declarative specs into Orekit objects.

All Orekit imports are lazy (function-level) so this module can be imported
without a running JVM.  The public entry points are:

- ``assemble_force_models``   — force specs -> list of Orekit ForceModel
- ``assemble_attitude_provider`` — attitude spec -> Orekit AttitudeProvider

Shared ingredient helpers (used by both numerical and DSST assembly):

- ``get_celestial_body``          — name -> Orekit CelestialBody
- ``build_atmosphere``            — DragSpec -> Orekit Atmosphere
- ``build_spacecraft_drag_shape`` — SpacecraftSpec -> DragSensitive or None
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from astrodyn_core.orekit_env import (
    get_earth_shape,
    get_iers_conventions,
    get_mu,
    get_universe_config,
    load_universe_from_dict,
)
from astrodyn_core.propagation.attitude import AttitudeSpec
from astrodyn_core.propagation.forces import (
    DragSpec,
    ForceSpec,
    GravitySpec,
    OceanTidesSpec,
    RelativitySpec,
    SRPSpec,
    SolidTidesSpec,
    ThirdBodySpec,
)
from astrodyn_core.propagation.spacecraft import SpacecraftSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================


def assemble_force_models(
    force_specs: Sequence[ForceSpec],
    spacecraft: SpacecraftSpec,
    initial_orbit: Any,
    mu: float | None = None,
    universe: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Translate a sequence of ``ForceSpec`` objects into Orekit force models.

    Args:
        force_specs: Declarative force specifications.
        spacecraft: Physical spacecraft model (needed for drag / SRP shapes).
        initial_orbit: Orekit orbit used to derive ``mu`` when not provided.
        mu: Optional gravitational parameter override.
        universe: Optional universe config used by Earth/mu/IERS resolvers.

    Returns:
        Orekit ``ForceModel`` instances ready to add to a builder.
    """
    if mu is None:
        if universe is None:
            mu = float(initial_orbit.getMu())
        else:
            mu = float(get_mu(universe))

    models: list[Any] = []
    # Track gravity model for tides (they may need the tide system)
    gravity_model: Any = None

    for spec in force_specs:
        if isinstance(spec, GravitySpec):
            model = _build_gravity(spec, universe)
            if model is not None:
                gravity_model = model
                models.append(model)
        elif isinstance(spec, DragSpec):
            model = _build_drag(spec, spacecraft, universe)
            if model is not None:
                models.append(model)
        elif isinstance(spec, SRPSpec):
            model = _build_srp(spec, spacecraft, universe)
            if model is not None:
                models.append(model)
        elif isinstance(spec, ThirdBodySpec):
            models.extend(_build_third_body(spec))
        elif isinstance(spec, RelativitySpec):
            models.append(_build_relativity(mu))
        elif isinstance(spec, SolidTidesSpec):
            model = _build_solid_tides(gravity_model, mu, universe)
            if model is not None:
                models.append(model)
        elif isinstance(spec, OceanTidesSpec):
            model = _build_ocean_tides(spec, mu, universe)
            if model is not None:
                models.append(model)
        else:
            raise TypeError(f"Unknown force spec type: {type(spec).__name__}")

    return models


def assemble_attitude_provider(
    attitude: AttitudeSpec,
    initial_orbit: Any,
    universe: Mapping[str, Any] | None = None,
) -> Any | None:
    """Translate an ``AttitudeSpec`` into an Orekit ``AttitudeProvider``.

    Returns *None* if no attitude law can be resolved (should not happen
    if validation passed, but kept defensive).

    Args:
        attitude: Declarative attitude specification.
        initial_orbit: Orekit initial orbit used to resolve frames.
        universe: Optional universe config (used for nadir pointing Earth shape).

    Returns:
        Orekit ``AttitudeProvider`` or ``None`` if no mode could be resolved.
    """
    # Pass-through escape hatch
    if attitude.provider is not None:
        return attitude.provider

    try:
        from org.orekit.frames import LOFType
        from org.orekit.attitudes import LofOffset, NadirPointing, FrameAlignedProvider
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Ensure orekit>=13.1 is installed "
            "and the JVM is initialised."
        ) from exc

    frame = initial_orbit.getFrame()
    mode = attitude.mode

    if mode in ("qsw", "vvlh"):
        return LofOffset(frame, LOFType.VVLH)
    if mode == "tnw":
        return LofOffset(frame, LOFType.TNW)
    if mode == "nadir":
        earth_shape = _get_earth_shape(universe)
        return NadirPointing(frame, earth_shape)
    if mode == "inertial":
        return FrameAlignedProvider(frame)

    logger.warning("Could not resolve attitude mode '%s'. Returning None.", mode)
    return None


# ============================================================================
# Internal helpers — Orekit object construction
# ============================================================================


def _get_earth_shape(universe: Mapping[str, Any] | None = None) -> Any:
    """Return configured Earth body shape."""
    return get_earth_shape(universe)


def get_celestial_body(name: str) -> Any:
    """Resolve a celestial body name to an Orekit ``CelestialBody``.

    Args:
        name: Lowercase body name (for example ``"sun"`` or ``"moon"``).

    Returns:
        Orekit celestial body instance.

    Raises:
        ValueError: If no mapping exists for ``name``.
    """
    from org.orekit.bodies import CelestialBodyFactory

    _BODY_MAP = {
        "sun": CelestialBodyFactory.getSun,
        "moon": CelestialBodyFactory.getMoon,
        "mercury": CelestialBodyFactory.getMercury,
        "venus": CelestialBodyFactory.getVenus,
        "mars": CelestialBodyFactory.getMars,
        "jupiter": CelestialBodyFactory.getJupiter,
        "saturn": CelestialBodyFactory.getSaturn,
        "uranus": CelestialBodyFactory.getUranus,
        "neptune": CelestialBodyFactory.getNeptune,
    }
    getter = _BODY_MAP.get(name)
    if getter is None:
        raise ValueError(f"No CelestialBody factory for '{name}'.")
    return getter()


def _get_sun() -> Any:
    return get_celestial_body("sun")


def _get_moon() -> Any:
    return get_celestial_body("moon")


def _get_iers_conventions(universe: Mapping[str, Any] | None = None) -> Any:
    return get_iers_conventions(universe)


def _get_use_simple_eop(universe: Mapping[str, Any] | None = None) -> bool:
    if universe is None:
        cfg = get_universe_config()
    else:
        cfg = load_universe_from_dict(universe)
    return bool(cfg.get("use_simple_eop", True))


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------


def _build_gravity(spec: GravitySpec, universe: Mapping[str, Any] | None = None) -> Any | None:
    if spec.degree == 0 and spec.order == 0:
        return None  # point mass — handled by Keplerian term in Orekit

    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory

    earth_shape = _get_earth_shape(universe)
    if spec.normalized:
        provider = GravityFieldFactory.getNormalizedProvider(spec.degree, spec.order)
    else:
        provider = GravityFieldFactory.getUnnormalizedProvider(spec.degree, spec.order)

    return HolmesFeatherstoneAttractionModel(earth_shape.getBodyFrame(), provider)


# ---------------------------------------------------------------------------
# Drag
# ---------------------------------------------------------------------------


def _build_drag(
    spec: DragSpec, sc: SpacecraftSpec, universe: Mapping[str, Any] | None = None
) -> Any | None:
    from org.orekit.forces.drag import DragForce, IsotropicDrag

    atmosphere = build_atmosphere(spec, universe)
    if atmosphere is None:
        return None

    drag_shape = build_spacecraft_drag_shape(sc)
    if drag_shape is None:
        drag_shape = IsotropicDrag(float(sc.drag_area), float(sc.drag_coeff))

    return DragForce(atmosphere, drag_shape)


def build_spacecraft_drag_shape(sc: SpacecraftSpec) -> Any | None:
    """Build a box-wing drag/SRP shape when configured.

    Args:
        sc: Spacecraft physical model.

    Returns:
        Orekit box-and-solar-array shape, or ``None`` to indicate isotropic
        drag/SRP models should be used.
    """
    if not sc.use_box_wing:
        return None

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.forces import BoxAndSolarArraySpacecraft

    sun = _get_sun()
    sa_axis = Vector3D(
        float(sc.solar_array_axis[0]),
        float(sc.solar_array_axis[1]),
        float(sc.solar_array_axis[2]),
    ).normalize()

    return BoxAndSolarArraySpacecraft(
        float(sc.x_length),
        float(sc.y_length),
        float(sc.z_length),
        sun,
        float(sc.solar_array_area),
        sa_axis,
        float(sc.box_drag_coeff),
        float(sc.box_abs_coeff),
        float(sc.box_ref_coeff),
    )


def build_atmosphere(spec: DragSpec, universe: Mapping[str, Any] | None = None) -> Any | None:
    """Create an Orekit atmosphere model from ``DragSpec``.

    Args:
        spec: Drag force model specification.
        universe: Optional universe config used for Earth shape/IERS settings.

    Returns:
        Orekit atmosphere model instance, or ``None`` if the drag model is not
        resolvable.
    """
    model = spec.atmosphere_model
    earth_shape = _get_earth_shape(universe)
    sun = _get_sun()

    if model == "simpleexponential":
        from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere

        return SimpleExponentialAtmosphere(
            earth_shape,
            float(spec.ref_rho),  # type: ignore[arg-type]
            float(spec.ref_alt),  # type: ignore[arg-type]
            float(spec.scale_height),  # type: ignore[arg-type]
        )

    if model == "harrispriester":
        from org.orekit.models.earth.atmosphere import HarrisPriester

        return HarrisPriester(sun, earth_shape)

    # Models requiring space-weather data
    if model in ("nrlmsise00", "dtm2000", "jb2008"):
        atmos_params = _get_space_weather_provider(spec)
        return _build_weather_atmosphere(model, atmos_params, sun, earth_shape)

    return None


def _get_space_weather_provider(spec: DragSpec) -> Any:
    """Resolve a space-weather data provider for weather-driven atmospheres."""
    from org.orekit.data import DataContext
    from org.orekit.models.earth.atmosphere.data import (
        CssiSpaceWeatherData,
        MarshallSolarActivityFutureEstimation,
    )
    from org.orekit.time import TimeScalesFactory

    utc = TimeScalesFactory.getUTC()
    data_manager = DataContext.getDefault().getDataProvidersManager()

    model = spec.atmosphere_model

    # JB2008 has its own data provider
    if model == "jb2008":
        from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData

        return JB2008SpaceEnvironmentData(
            "(?i)(SOLFSMY)(.*)(\\.txt)",
            "(?i)(DTCFILE)(.*)(\\.txt)",
            data_manager,
            utc,
        )

    source = spec.space_weather_source

    if source == "cssi":
        return CssiSpaceWeatherData(
            CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES,
            data_manager,
            utc,
        )

    if source == "msafe":
        strength_str = spec.solar_activity_strength.upper()
        strength_enum = getattr(MarshallSolarActivityFutureEstimation.StrengthLevel, strength_str)
        return MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            strength_enum,
        )

    # Fallback
    logger.warning("Unknown space_weather_source '%s', defaulting to CSSI.", source)
    return CssiSpaceWeatherData(
        CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES,
        data_manager,
        utc,
    )


def _build_weather_atmosphere(model: str, atmos_params: Any, sun: Any, earth_shape: Any) -> Any:
    """Instantiate a weather-driven atmosphere model by identifier."""
    if model == "nrlmsise00":
        from org.orekit.models.earth.atmosphere import NRLMSISE00

        return NRLMSISE00(atmos_params, sun, earth_shape)

    if model == "dtm2000":
        from org.orekit.models.earth.atmosphere import DTM2000

        return DTM2000(atmos_params, sun, earth_shape)

    if model == "jb2008":
        from org.orekit.models.earth.atmosphere import JB2008

        return JB2008(atmos_params, sun, earth_shape)

    raise ValueError(f"No weather-driven atmosphere builder for '{model}'.")


# ---------------------------------------------------------------------------
# SRP
# ---------------------------------------------------------------------------


def _build_srp(spec: SRPSpec, sc: SpacecraftSpec, universe: Mapping[str, Any] | None = None) -> Any:
    from org.orekit.forces.radiation import (
        IsotropicRadiationSingleCoefficient,
        SolarRadiationPressure,
    )

    sun = _get_sun()
    earth_shape = _get_earth_shape(universe)

    srp_shape = build_spacecraft_drag_shape(sc)
    if srp_shape is None:
        srp_shape = IsotropicRadiationSingleCoefficient(float(sc.srp_area), float(sc.srp_coeff))

    srp_model = SolarRadiationPressure(sun, earth_shape, srp_shape)

    if spec.enable_moon_eclipse:
        from org.orekit.utils import Constants

        moon = _get_moon()
        srp_model.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)

    if spec.enable_albedo:
        _add_albedo(sc, earth_shape, sun)
        # Note: albedo is a separate force model; we return the SRP model here
        # and albedo should be appended separately if needed. For now, log
        # a reminder.  A future version may return multiple models.
        logger.info(
            "enable_albedo is set but albedo is handled as a separate force "
            "model (KnockeRediffusedForceModel). Consider adding it explicitly."
        )

    return srp_model


def _add_albedo(sc: SpacecraftSpec, earth_shape: Any, sun: Any) -> Any:
    """Build KnockeRediffusedForceModel for Earth albedo.

    Returns the force model; caller is responsible for adding it to the builder.
    """
    import math

    from org.orekit.forces.radiation import (
        IsotropicRadiationSingleCoefficient,
        KnockeRediffusedForceModel,
    )

    shape = build_spacecraft_drag_shape(sc)
    if shape is None:
        shape = IsotropicRadiationSingleCoefficient(float(sc.srp_area), float(sc.srp_coeff))

    angular_res = math.radians(360.0 / 48.0)
    return KnockeRediffusedForceModel(sun, shape, earth_shape.getEquatorialRadius(), angular_res)


# ---------------------------------------------------------------------------
# Third-body
# ---------------------------------------------------------------------------


def _build_third_body(spec: ThirdBodySpec) -> list[Any]:
    from org.orekit.forces.gravity import ThirdBodyAttraction

    models = []
    for body_name in spec.bodies:
        body = get_celestial_body(body_name)
        models.append(ThirdBodyAttraction(body))
    return models


# ---------------------------------------------------------------------------
# Relativity
# ---------------------------------------------------------------------------


def _build_relativity(mu: float) -> Any:
    from org.orekit.forces.gravity import Relativity

    return Relativity(mu)


# ---------------------------------------------------------------------------
# Tides
# ---------------------------------------------------------------------------


def _build_solid_tides(
    gravity_model: Any, mu: float, universe: Mapping[str, Any] | None = None
) -> Any:
    from org.orekit.forces.gravity import SolidTides
    from org.orekit.time import TimeScalesFactory

    earth_shape = _get_earth_shape(universe)
    iers = _get_iers_conventions(universe)
    use_simple_eop = _get_use_simple_eop(universe)
    ut1 = TimeScalesFactory.getUT1(iers, use_simple_eop)

    # Determine tide system from gravity model if available
    if gravity_model is not None:
        tide_system = gravity_model.getTideSystem()
    else:
        from org.orekit.forces.gravity.potential import TideSystem

        tide_system = TideSystem.TIDE_FREE

    sun = _get_sun()
    moon = _get_moon()

    return SolidTides(
        earth_shape.getBodyFrame(),
        earth_shape.getEquatorialRadius(),
        mu,
        tide_system,
        iers,
        ut1,
        [sun, moon],
    )


def _build_ocean_tides(
    spec: OceanTidesSpec, mu: float, universe: Mapping[str, Any] | None = None
) -> Any:
    from org.orekit.forces.gravity import OceanTides
    from org.orekit.time import TimeScalesFactory

    earth_shape = _get_earth_shape(universe)
    iers = _get_iers_conventions(universe)
    use_simple_eop = _get_use_simple_eop(universe)
    ut1 = TimeScalesFactory.getUT1(iers, use_simple_eop)

    return OceanTides(
        earth_shape.getBodyFrame(),
        earth_shape.getEquatorialRadius(),
        mu,
        spec.degree,
        spec.order,
        iers,
        ut1,
    )
