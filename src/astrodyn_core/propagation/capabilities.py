"""Capability metadata for providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Describes what a provider supports.

    Attributes
    ----------
    supports_builder:
        Provider implements ``build_builder()`` returning a propagator builder.
    supports_propagator:
        Provider implements ``build_propagator()`` returning a propagator.
    supports_stm:
        Propagator supports State Transition Matrix extraction.
    supports_field_state:
        Propagator supports Taylor-algebra / Field-based state propagation.
    supports_multi_satellite:
        Provider supports simultaneous multi-satellite propagation.
    supports_bounded_output:
        Propagator output is bounded to a fixed time interval.
        ``True`` for ephemeris-based propagators, ``False`` for unbounded or
        not applicable (default).
    is_analytical:
        Propagator is a custom/analytical implementation (not Orekit-native
        numerical).  Analytical providers typically use ``body_constants``
        from :class:`BuildContext` instead of Orekit force model objects.
    supports_custom_output:
        Propagator can expose backend-specific output (e.g. raw numpy
        arrays, internal element states) alongside standard Orekit-compatible
        ``SpacecraftState`` results.
    """

    supports_builder: bool = True
    supports_propagator: bool = True
    supports_stm: bool = False
    supports_field_state: bool = False
    supports_multi_satellite: bool = False
    supports_bounded_output: bool = False
    is_analytical: bool = False
    supports_custom_output: bool = False
