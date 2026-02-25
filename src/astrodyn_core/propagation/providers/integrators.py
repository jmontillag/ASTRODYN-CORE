"""Orekit integrator builder helpers."""

from __future__ import annotations

from astrodyn_core.propagation.specs import IntegratorSpec


def create_orekit_integrator_builder(spec: IntegratorSpec):
    """Create an Orekit ``ODEIntegratorBuilder`` from ``IntegratorSpec``.

    Args:
        spec: Declarative integrator configuration.

    Returns:
        Orekit integrator builder instance matching ``spec.kind``.

    Raises:
        RuntimeError: If Orekit classes are unavailable.
        ValueError: If the integrator kind is unsupported or required fields are
            missing.
    """

    try:
        from org.orekit.propagation.conversion import (
            AdamsBashforthIntegratorBuilder,
            ClassicalRungeKuttaIntegratorBuilder,
            DormandPrince54IntegratorBuilder,
            DormandPrince853IntegratorBuilder,
            GillIntegratorBuilder,
            GraggBulirschStoerIntegratorBuilder,
            MidpointIntegratorBuilder,
        )
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    kind = spec.kind

    if kind in {"dormandprince853", "dormand_prince_853", "dp853"}:
        return DormandPrince853IntegratorBuilder(
            _required(spec.min_step, "min_step"),
            _required(spec.max_step, "max_step"),
            _required(spec.position_tolerance, "position_tolerance"),
        )

    if kind in {"dormandprince54", "dormand_prince_54", "dp54"}:
        return DormandPrince54IntegratorBuilder(
            _required(spec.min_step, "min_step"),
            _required(spec.max_step, "max_step"),
            _required(spec.position_tolerance, "position_tolerance"),
        )

    if kind in {"graggbulirschstoer", "gragg_bulirsch_stoer", "gbs"}:
        return GraggBulirschStoerIntegratorBuilder(
            _required(spec.min_step, "min_step"),
            _required(spec.max_step, "max_step"),
            _required(spec.position_tolerance, "position_tolerance"),
        )

    if kind in {"adamsbashforth", "adams_bashforth", "ab"}:
        return AdamsBashforthIntegratorBuilder(
            _required(spec.n_steps, "n_steps"),
            _required(spec.min_step, "min_step"),
            _required(spec.max_step, "max_step"),
            _required(spec.position_tolerance, "position_tolerance"),
        )

    if kind in {"classicalrungekutta", "classical_runge_kutta", "rk4"}:
        return ClassicalRungeKuttaIntegratorBuilder(_required(spec.step, "step"))

    if kind == "gill":
        return GillIntegratorBuilder(_required(spec.step, "step"))

    if kind == "midpoint":
        return MidpointIntegratorBuilder(_required(spec.step, "step"))

    raise ValueError(f"Unsupported integrator kind: {spec.kind}")


def _required(value, field_name: str):
    """Return a required spec field value or raise a clear error."""
    if value is None:
        raise ValueError(f"IntegratorSpec.{field_name} is required for this integrator kind.")
    return value
