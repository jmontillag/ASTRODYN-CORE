"""GEqOE Taylor propagator using heyoka.py automatic differentiation.

State vector: [nu, p1, p2, K, q1, q2] in km/s units.
K = generalized eccentric longitude (no Kepler solve needed in the RHS).

Reference: Baù, Hernando-Ayuso & Bombardelli (2021), Celest. Mech. Dyn. Astr. 133:50.
"""

from astrodyn_core.geqoe_taylor.constants import A_J2, GM_MOON, GM_SUN, J2, J3, J4, J5, J6, MU, RE
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.integrator import (
    build_state_integrator,
    build_stm_integrator,
    build_thrust_sensitivity_integrator,
    build_thrust_state_integrator,
    build_thrust_stm_integrator,
    extract_endpoint_jacobian,
    extract_stm,
    extract_variational_matrices,
    parameter_names_from_map,
    propagate,
)
from astrodyn_core.geqoe_taylor.perturbations.composite import CompositePerturbation
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.perturbations.third_body import ThirdBodyPerturbation
from astrodyn_core.geqoe_taylor.perturbations.thrust import ContinuousThrustPerturbation
from astrodyn_core.geqoe_taylor.perturbations.zonal import ZonalPerturbation
from astrodyn_core.geqoe_taylor.shooting import (
    ArcPropagationResult,
    DecisionTrackingPenaltySpec,
    DecisionTrackingTerm,
    InertialPositionMeasurementModel,
    InertialRangeMeasurementModel,
    LinearizedCovarianceResult,
    MeasurementModel,
    MeasurementObjectiveSpec,
    MeasurementResidualEvaluation,
    MultiArcShootingProblem,
    SampledMeasurement,
    SampledMeasurementResult,
    ShootingArc,
    ShootingEvaluation,
    ShootingOptimizationResult,
    ShootingSolveSpec,
    SmoothnessPenaltySpec,
    TerminalConstraintSpec,
)
from astrodyn_core.geqoe_taylor.thrust import (
    ConstantRTNThrustLaw,
    ContinuousThrustLaw,
    CubicHermiteRTNThrustLaw,
    FourierKRTNThrustLaw,
)

__all__ = [
    "MU", "J2", "J3", "J4", "J5", "J6", "RE", "A_J2", "GM_SUN", "GM_MOON",
    "cart2geqoe", "geqoe2cart",
    "J2Perturbation", "ThirdBodyPerturbation", "CompositePerturbation",
    "ZonalPerturbation", "ContinuousThrustPerturbation",
    "ContinuousThrustLaw", "ConstantRTNThrustLaw", "CubicHermiteRTNThrustLaw",
    "FourierKRTNThrustLaw",
    "build_state_integrator", "build_stm_integrator",
    "build_thrust_state_integrator", "build_thrust_stm_integrator",
    "build_thrust_sensitivity_integrator",
    "propagate", "extract_stm", "extract_endpoint_jacobian",
    "extract_variational_matrices",
    "parameter_names_from_map",
    "MeasurementModel", "SampledMeasurement",
    "SampledMeasurementResult", "MeasurementResidualEvaluation",
    "LinearizedCovarianceResult",
    "MeasurementObjectiveSpec", "DecisionTrackingTerm",
    "DecisionTrackingPenaltySpec",
    "InertialPositionMeasurementModel", "InertialRangeMeasurementModel",
    "ShootingArc", "ArcPropagationResult",
    "ShootingEvaluation", "ShootingOptimizationResult",
    "TerminalConstraintSpec", "SmoothnessPenaltySpec", "ShootingSolveSpec",
    "MultiArcShootingProblem",
]
