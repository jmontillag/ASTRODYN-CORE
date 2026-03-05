"""Provider implementations."""

from typing import Any

from astrodyn_core.propagation.providers.orekit_native import register_default_orekit_providers


def register_analytical_providers(registry: Any) -> None:
    """Register built-in analytical (non-Orekit-native) providers.

    Currently registers:
    - ``GEqOEProvider`` (kind ``"geqoe"``) — J2 Taylor-series propagator in
      Generalized Equinoctial Orbital Elements.
    - ``AdaptiveGEqOEProvider`` (kind ``"geqoe-adaptive"``) — Adaptive
      checkpoint-based J2 Taylor-series propagator.

    Args:
        registry: Provider registry to mutate.
    """
    from astrodyn_core.propagation.providers.geqoe.adaptive import AdaptiveGEqOEProvider
    from astrodyn_core.propagation.providers.geqoe.numerical import NumericalGEqOEProvider
    from astrodyn_core.propagation.providers.geqoe.provider import GEqOEProvider

    registry.register_propagator_provider(GEqOEProvider())
    registry.register_propagator_provider(AdaptiveGEqOEProvider())
    registry.register_propagator_provider(NumericalGEqOEProvider())


def register_all_providers(registry: Any) -> None:
    """Register all built-in providers (Orekit-native + analytical).

    Args:
        registry: Provider registry to mutate.
    """
    register_default_orekit_providers(registry)
    register_analytical_providers(registry)


__all__ = [
    "register_default_orekit_providers",
    "register_analytical_providers",
    "register_all_providers",
]
