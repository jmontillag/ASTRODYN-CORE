"""GEqOE analytical propagator provider.

Satisfies the ``PropagatorProvider`` protocol so that the GEqOE Taylor-series
propagator can be built through the standard ``PropagatorFactory`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.specs import PropagatorSpec


@dataclass(frozen=True, slots=True)
class GEqOEProvider:
    """Provider for the J2 Taylor-series GEqOE analytical propagator.

    Registered under ``kind="geqoe"``.  Only supports direct propagator
    construction (no Orekit ``PropagatorBuilder`` lane).
    """

    kind: str = "geqoe"
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=False,
        supports_propagator=True,
        supports_stm=True,
        is_analytical=True,
        supports_custom_output=True,
    )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Build a ``GEqOEPropagator`` from the spec and context.

        Requires Orekit to be initialised (for ``AbstractPropagator`` base
        class and ``SpacecraftState`` conversion).
        """
        from astrodyn_core.propagation.providers.geqoe.propagator import (
            make_orekit_geqoe_propagator,
        )

        body_constants = context.require_body_constants()
        initial_orbit = context.require_initial_orbit()

        order = spec.orekit_options.get("taylor_order", 4)
        mass_kg = spec.mass_kg

        return make_orekit_geqoe_propagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=mass_kg,
        )
