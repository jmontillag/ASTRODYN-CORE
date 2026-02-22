"""Provider registry for builder and propagator lanes.

The registry accepts any string as a ``kind`` key, which allows custom
(analytical, semi-analytical, or hybrid) propagators to be registered
alongside the built-in Orekit-native providers without modifying the
:class:`~astrodyn_core.propagation.specs.PropagatorKind` enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderRegistry:
    """In-memory provider registry keyed by propagator kind (any string)."""

    _builder_providers: dict[str, Any] = field(default_factory=dict)
    _propagator_providers: dict[str, Any] = field(default_factory=dict)

    def register_builder_provider(self, provider: Any) -> None:
        self._builder_providers[provider.kind] = provider

    def register_propagator_provider(self, provider: Any) -> None:
        self._propagator_providers[provider.kind] = provider

    def get_builder_provider(self, kind: str) -> Any:
        try:
            return self._builder_providers[kind]
        except KeyError as exc:
            available = ", ".join(sorted(self._builder_providers)) or "(none)"
            raise KeyError(
                f"No builder provider registered for kind={kind!r}. Available: {available}."
            ) from exc

    def get_propagator_provider(self, kind: str) -> Any:
        try:
            return self._propagator_providers[kind]
        except KeyError as exc:
            available = ", ".join(sorted(self._propagator_providers)) or "(none)"
            raise KeyError(
                f"No propagator provider registered for kind={kind!r}. Available: {available}."
            ) from exc

    def available_builder_kinds(self) -> list[str]:
        return sorted(str(k) for k in self._builder_providers)

    def available_propagator_kinds(self) -> list[str]:
        return sorted(str(k) for k in self._propagator_providers)
