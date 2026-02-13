"""Provider registry for builder and propagator lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrodyn_core.propagation.specs import PropagatorKind


@dataclass(slots=True)
class ProviderRegistry:
    """In-memory provider registry keyed by propagator kind."""

    _builder_providers: dict[PropagatorKind, Any] = field(default_factory=dict)
    _propagator_providers: dict[PropagatorKind, Any] = field(default_factory=dict)

    def register_builder_provider(self, provider: Any) -> None:
        self._builder_providers[provider.kind] = provider

    def register_propagator_provider(self, provider: Any) -> None:
        self._propagator_providers[provider.kind] = provider

    def get_builder_provider(self, kind: PropagatorKind) -> Any:
        try:
            return self._builder_providers[kind]
        except KeyError as exc:
            raise KeyError(f"No builder provider registered for kind={kind.value}.") from exc

    def get_propagator_provider(self, kind: PropagatorKind) -> Any:
        try:
            return self._propagator_providers[kind]
        except KeyError as exc:
            raise KeyError(f"No propagator provider registered for kind={kind.value}.") from exc

    def available_builder_kinds(self) -> list[str]:
        return sorted(k.value for k in self._builder_providers)

    def available_propagator_kinds(self) -> list[str]:
        return sorted(k.value for k in self._propagator_providers)
