"""DSST force model assembly — translates ForceSpec objects into DSSTForceModel instances.

The DSST propagator uses a distinct set of force model classes from
``org.orekit.propagation.semianalytical.dsst.forces``.  This module
translates the same ``ForceSpec`` dataclasses used by the numerical
propagator into their DSST-specific Orekit counterparts:

=================  ============================================
ForceSpec          DSST Orekit class(es)
=================  ============================================
GravitySpec        DSSTZonal + DSSTTesseral
DragSpec           DSSTAtmosphericDrag
SRPSpec            DSSTSolarRadiationPressure
ThirdBodySpec      DSSTThirdBody
RelativitySpec     (not available for DSST — skipped with warning)
SolidTidesSpec     (not available for DSST — skipped with warning)
OceanTidesSpec     (not available for DSST — skipped with warning)
=================  ============================================

All Orekit imports are lazy so this module can be imported without a JVM.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Mapping, Sequence

from astrodyn_core.propagation.assembly import (
    build_atmosphere,
    build_spacecraft_drag_shape,
    get_celestial_body,
)
from astrodyn_core.propagation.config import get_earth_shape, get_mu
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

# Force spec types that have no DSST equivalent
_UNSUPPORTED_DSST_SPECS = (RelativitySpec, SolidTidesSpec, OceanTidesSpec)


def assemble_dsst_force_models(
    force_specs: Sequence[ForceSpec],
    spacecraft: SpacecraftSpec,
    initial_orbit: Any,
    mu: float | None = None,
    universe: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Translate ``ForceSpec`` objects into Orekit DSST force models.

    Parameters
    ----------
    force_specs:
        Declarative force specifications (same types as numerical).
    spacecraft:
        Physical spacecraft model (needed for drag / SRP shapes).
    initial_orbit:
        Orekit ``Orbit`` — used to derive mu if not provided.
    mu:
        Gravitational parameter override.  If *None*, taken from
        ``initial_orbit.getMu()`` or the universe config.
    universe:
        Optional universe configuration dict.

    Returns
    -------
    list
        Orekit ``DSSTForceModel`` instances ready for ``DSSTPropagatorBuilder.addForceModel()``.
    """
    if mu is None:
        if universe is None:
            mu = float(initial_orbit.getMu())
        else:
            mu = float(get_mu(universe))

    models: list[Any] = []

    for spec in force_specs:
        if isinstance(spec, GravitySpec):
            models.extend(_build_dsst_gravity(spec, mu, universe))
        elif isinstance(spec, DragSpec):
            model = _build_dsst_drag(spec, spacecraft, mu, universe)
            if model is not None:
                models.append(model)
        elif isinstance(spec, SRPSpec):
            model = _build_dsst_srp(spec, spacecraft, mu, universe)
            if model is not None:
                models.append(model)
        elif isinstance(spec, ThirdBodySpec):
            models.extend(_build_dsst_third_body(spec, mu))
        elif isinstance(spec, _UNSUPPORTED_DSST_SPECS):
            warnings.warn(
                f"{type(spec).__name__} is not supported for DSST propagation "
                f"and will be ignored.",
                stacklevel=2,
            )
        else:
            raise TypeError(f"Unknown force spec type: {type(spec).__name__}")

    return models


# ---------------------------------------------------------------------------
# DSST gravity: zonal + tesseral
# ---------------------------------------------------------------------------


def _build_dsst_gravity(
    spec: GravitySpec,
    mu: float,
    universe: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Build DSSTZonal and (optionally) DSSTTesseral force models."""
    if spec.degree == 0 and spec.order == 0:
        return []  # point mass only

    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTTesseral, DSSTZonal
    from org.orekit.utils import Constants

    # DSST requires unnormalized provider
    provider = GravityFieldFactory.getUnnormalizedProvider(spec.degree, spec.order)

    # Body-fixed frame: ITRF is ideal but TOD is a good performance trade-off
    earth_shape = get_earth_shape(universe)
    body_frame = earth_shape.getBodyFrame()

    models: list[Any] = []

    # Zonal harmonics (always added when degree >= 2)
    models.append(DSSTZonal(body_frame, provider))

    # Tesseral harmonics (only if order > 0)
    if spec.order > 0:
        rotation_rate = Constants.WGS84_EARTH_ANGULAR_VELOCITY
        models.append(DSSTTesseral(body_frame, rotation_rate, provider))

    return models


# ---------------------------------------------------------------------------
# DSST drag
# ---------------------------------------------------------------------------


def _build_dsst_drag(
    spec: DragSpec,
    sc: SpacecraftSpec,
    mu: float,
    universe: Mapping[str, Any] | None = None,
) -> Any | None:
    """Build a DSSTAtmosphericDrag force model."""
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTAtmosphericDrag

    atmosphere = build_atmosphere(spec, universe)
    if atmosphere is None:
        return None

    drag_shape = build_spacecraft_drag_shape(sc)
    if drag_shape is None:
        drag_shape = IsotropicDrag(float(sc.drag_area), float(sc.drag_coeff))

    drag_force = DragForce(atmosphere, drag_shape)
    return DSSTAtmosphericDrag(drag_force, mu)


# ---------------------------------------------------------------------------
# DSST SRP
# ---------------------------------------------------------------------------


def _build_dsst_srp(
    spec: SRPSpec,
    sc: SpacecraftSpec,
    mu: float,
    universe: Mapping[str, Any] | None = None,
) -> Any:
    """Build a DSSTSolarRadiationPressure force model."""
    from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTSolarRadiationPressure

    sun = get_celestial_body("sun")
    earth_shape = get_earth_shape(universe)

    srp_shape = build_spacecraft_drag_shape(sc)
    if srp_shape is None:
        srp_shape = IsotropicRadiationSingleCoefficient(float(sc.srp_area), float(sc.srp_coeff))

    return DSSTSolarRadiationPressure(sun, earth_shape, srp_shape, mu)


# ---------------------------------------------------------------------------
# DSST third body
# ---------------------------------------------------------------------------


def _build_dsst_third_body(spec: ThirdBodySpec, mu: float) -> list[Any]:
    """Build DSSTThirdBody force models (one per body)."""
    from org.orekit.propagation.semianalytical.dsst.forces import DSSTThirdBody

    models: list[Any] = []
    for body_name in spec.bodies:
        body = get_celestial_body(body_name)
        models.append(DSSTThirdBody(body, mu))
    return models
