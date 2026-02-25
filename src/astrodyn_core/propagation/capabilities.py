"""Capability metadata for providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Describes what a provider supports.

    Attributes:
        supports_builder: Provider implements ``build_builder()``.
        supports_propagator: Provider implements ``build_propagator()``.
        supports_stm: Propagator supports State Transition Matrix extraction.
        supports_field_state: Propagator supports field/Taylor state propagation.
        supports_multi_satellite: Provider supports multi-satellite propagation.
        supports_bounded_output: Output is bounded to a fixed time interval.
        is_analytical: Custom/analytical implementation (not Orekit-native
            numerical).
        supports_custom_output: Provider can expose backend-specific output in
            addition to standard Orekit-compatible states.
    """

    supports_builder: bool = True
    supports_propagator: bool = True
    supports_stm: bool = False
    supports_field_state: bool = False
    supports_multi_satellite: bool = False
    supports_bounded_output: bool = False
    is_analytical: bool = False
    supports_custom_output: bool = False
