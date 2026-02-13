from dataclasses import dataclass

from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.factory import PropagatorFactory
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec


@dataclass
class DummyProvider:
    kind: PropagatorKind
    capabilities: CapabilityDescriptor = CapabilityDescriptor()

    def build_builder(self, spec: PropagatorSpec, context: BuildContext):
        return {"kind": spec.kind.value, "lane": "builder"}

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext):
        return {"kind": spec.kind.value, "lane": "propagator"}


def test_factory_uses_registry_provider() -> None:
    factory = PropagatorFactory()
    provider = DummyProvider(PropagatorKind.NUMERICAL)
    factory.registry.register_builder_provider(provider)
    factory.registry.register_propagator_provider(provider)

    spec = PropagatorSpec(
        kind=PropagatorKind.NUMERICAL,
        integrator=IntegratorSpec(kind="dp853", min_step=0.1, max_step=30.0, position_tolerance=10.0),
    )
    ctx = BuildContext(initial_orbit=object())

    builder = factory.build_builder(spec, ctx)
    propagator = factory.build_propagator(spec, ctx)

    assert builder["lane"] == "builder"
    assert propagator["lane"] == "propagator"
