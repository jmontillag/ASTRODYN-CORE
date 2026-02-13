"""Capability metadata for providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Describes what a provider supports."""

    supports_builder: bool = True
    supports_propagator: bool = True
    supports_stm: bool = False
    supports_field_state: bool = False
    supports_multi_satellite: bool = False
    supports_bounded_output: bool = False
