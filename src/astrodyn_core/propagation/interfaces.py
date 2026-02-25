"""Core interfaces and context objects for propagation providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.specs import PropagatorSpec

if TYPE_CHECKING:
    from astrodyn_core.states.models import OrbitStateRecord


@dataclass(slots=True)
class BuildContext:
    """Runtime context required by provider implementations.

    Orekit-native providers use ``initial_orbit`` and ``force_models``.
    Custom/analytical providers may instead rely on ``body_constants`` for
    physical parameters like ``mu``, ``J2``, and equatorial radius.

    Attributes:
        initial_orbit: Orekit initial orbit for the propagation build.
        position_tolerance: Position tolerance used by some Orekit builders.
        attitude_provider: Orekit ``AttitudeProvider`` override.
        force_models: Assembled Orekit force models for numerical/DSST builds.
        universe: Universe configuration mapping used for assembly/resolvers.
        metadata: Free-form metadata (for example initial mass).
        body_constants: Optional analytical constants mapping (``mu``, ``j2``,
            ``re``).
    """

    initial_orbit: Any | None = None
    position_tolerance: float = 10.0
    attitude_provider: Any | None = None
    force_models: Sequence[Any] = field(default_factory=tuple)
    universe: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    body_constants: Mapping[str, float] | None = None

    @classmethod
    def from_state_record(
        cls,
        state_record: OrbitStateRecord,
        *,
        universe: Mapping[str, Any] | None = None,
        position_tolerance: float = 10.0,
        attitude_provider: Any | None = None,
        force_models: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
        body_constants: Mapping[str, float] | None = None,
    ) -> BuildContext:
        """Construct a BuildContext from a serializable state record.

        Args:
            state_record: Serializable initial state record.
            universe: Optional universe configuration mapping.
            position_tolerance: Builder position tolerance.
            attitude_provider: Optional pre-built Orekit attitude provider.
            force_models: Optional pre-assembled Orekit force models.
            metadata: Optional metadata merged into the context.
            body_constants: Optional analytical constants mapping with keys
                ``mu`` (m^3/s^2), ``j2`` (dimensionless), and ``re`` (m).

        Returns:
            Build context with an Orekit ``initial_orbit`` converted from the
            state record.
        """
        from astrodyn_core.states import StateFileClient

        initial_orbit = StateFileClient(universe=universe).to_orekit_orbit(state_record)
        merged_metadata = dict(metadata or {})
        if state_record.mass_kg is not None:
            merged_metadata.setdefault("initial_mass_kg", state_record.mass_kg)

        return cls(
            initial_orbit=initial_orbit,
            position_tolerance=position_tolerance,
            attitude_provider=attitude_provider,
            force_models=force_models,
            universe=universe,
            metadata=merged_metadata,
            body_constants=body_constants,
        )

    def require_initial_orbit(self) -> Any:
        """Return ``initial_orbit`` or raise if it was not provided."""
        if self.initial_orbit is None:
            raise ValueError("initial_orbit is required for this propagation kind.")
        return self.initial_orbit

    def require_body_constants(self) -> Mapping[str, float]:
        """Return body constants or raise if not provided.

        Analytical providers should call this to get ``mu``, ``j2``, and ``re``.
        If not explicitly set on the context, the constants are resolved from
        Orekit ``Constants.WGS84_*`` at call time.

        Returns:
            Mapping containing ``mu``, ``j2``, and ``re``.

        Raises:
            RuntimeError: If constants are not provided and Orekit is
                unavailable for lazy resolution.
        """
        if self.body_constants is not None:
            return self.body_constants
        # Lazy resolution from Orekit Constants -- never hardcoded.
        try:
            from org.orekit.utils import Constants
        except Exception as exc:
            raise RuntimeError(
                "body_constants not provided and Orekit is unavailable. "
                "Pass body_constants explicitly or ensure Orekit is initialized."
            ) from exc
        return {
            "mu": float(Constants.WGS84_EARTH_MU),
            "j2": float(-Constants.WGS84_EARTH_C20),
            "re": float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS),
        }


class BuilderProvider(Protocol):
    """Provider that can build a propagator builder.

    For Orekit-native providers this returns an Orekit PropagatorBuilder.
    Custom/analytical providers may return their own builder type or raise
    NotImplementedError if they only support direct propagator construction.
    """

    kind: str
    capabilities: CapabilityDescriptor

    def build_builder(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Return a provider-specific propagator builder instance."""


class PropagatorProvider(Protocol):
    """Provider that can create a propagator directly.

    For Orekit-native providers this returns an Orekit Propagator (or a
    subclass of ``AbstractPropagator``).  Custom/analytical providers should
    also return an ``AbstractPropagator`` subclass so that the result
    integrates with downstream workflows (trajectory export, STM, etc.).
    """

    kind: str
    capabilities: CapabilityDescriptor

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext) -> Any:
        """Return a provider-specific propagator instance."""
