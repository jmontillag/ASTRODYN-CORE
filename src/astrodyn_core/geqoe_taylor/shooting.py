"""Multiple-shooting helpers for the GEqOE Taylor thrust backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import heyoka as hy
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize
from scipy.sparse import csr_matrix, lil_matrix

from astrodyn_core.geqoe_taylor.constants import MU
from astrodyn_core.geqoe_taylor.integrator import (
    build_thrust_sensitivity_integrator,
    extract_variational_matrices,
    parameter_names_from_map,
    propagate_grid,
)
from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel

STATE_NAMES = ("nu", "p1", "p2", "K", "q1", "q2", "m")
STATE_DIM = len(STATE_NAMES)


class MeasurementModel(Protocol):
    """Protocol for arc-sampled observation models used by shooting residuals."""

    output_dimension: int
    output_names: tuple[str, ...]

    def evaluate(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        """Return the predicted measurement from a propagated 7-state sample."""

    def state_jacobian(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        """Return d(measurement)/d(state) for the propagated 7-state sample."""


def _normalize_weight_matrix(
    weight_matrix: np.ndarray | list[float] | float | None,
    output_dimension: int,
) -> np.ndarray:
    """Normalize a scalar/vector/matrix weighting input to a 2-D left factor."""
    if weight_matrix is None:
        return np.eye(output_dimension, dtype=float)

    weights = np.asarray(weight_matrix, dtype=float)
    if weights.ndim == 0:
        return np.eye(output_dimension, dtype=float) * float(weights)
    if weights.ndim == 1:
        if weights.shape != (output_dimension,):
            raise ValueError(
                "1-D weight inputs must match the measurement output dimension."
            )
        return np.diag(weights)
    if weights.shape != (output_dimension, output_dimension):
        raise ValueError(
            "2-D weight inputs must be square with size output_dimension."
        )
    return np.array(weights, dtype=float, copy=True)


def _normalize_output_names(model: MeasurementModel, output_dimension: int) -> tuple[str, ...]:
    names = tuple(str(name) for name in model.output_names)
    if len(names) != output_dimension:
        raise ValueError(
            "MeasurementModel.output_names must match output_dimension."
        )
    return names


@dataclass(frozen=True)
class SampledMeasurement:
    """Definition of one weighted sampled observation inside a shooting arc.

    ``sample_time_s`` is measured relative to the selected arc start. Use either
    ``arc_index`` or ``arc_name`` to bind the sample to a specific arc.
    """

    model: MeasurementModel
    value: np.ndarray | list[float]
    sample_time_s: float
    arc_index: int | None = None
    arc_name: str | None = None
    weight_matrix: np.ndarray | list[float] | float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        output_dimension = int(self.model.output_dimension)
        _normalize_output_names(self.model, output_dimension)

        if self.arc_index is None and self.arc_name is None:
            raise ValueError(
                "SampledMeasurement requires either arc_index or arc_name."
            )
        if self.arc_index is not None and int(self.arc_index) < 0:
            raise ValueError("SampledMeasurement.arc_index must be non-negative.")
        if self.sample_time_s < 0.0:
            raise ValueError("SampledMeasurement.sample_time_s must be non-negative.")

        value = np.atleast_1d(np.asarray(self.value, dtype=float))
        if value.shape != (output_dimension,):
            raise ValueError(
                "SampledMeasurement.value must match the model output dimension."
            )

        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "weight_matrix",
            _normalize_weight_matrix(self.weight_matrix, output_dimension),
        )

    @classmethod
    def from_standard_deviation(
        cls,
        *,
        model: MeasurementModel,
        value: np.ndarray | list[float],
        sigma: np.ndarray | list[float] | float,
        sample_time_s: float,
        arc_index: int | None = None,
        arc_name: str | None = None,
        name: str | None = None,
    ) -> SampledMeasurement:
        """Build a sample using diagonal whitening from standard deviations."""
        output_dimension = int(model.output_dimension)
        sigma_arr = np.asarray(sigma, dtype=float)
        if np.any(sigma_arr <= 0.0):
            raise ValueError("Measurement standard deviations must be strictly positive.")
        if sigma_arr.ndim == 0:
            weight_matrix = np.eye(output_dimension, dtype=float) / float(sigma_arr)
        elif sigma_arr.shape == (output_dimension,):
            weight_matrix = np.diag(1.0 / sigma_arr)
        else:
            raise ValueError(
                "sigma must be a scalar or a vector matching the output dimension."
            )
        return cls(
            model=model,
            value=value,
            sample_time_s=sample_time_s,
            arc_index=arc_index,
            arc_name=arc_name,
            weight_matrix=weight_matrix,
            name=name,
        )

    @classmethod
    def from_covariance(
        cls,
        *,
        model: MeasurementModel,
        value: np.ndarray | list[float],
        covariance: np.ndarray | list[list[float]],
        sample_time_s: float,
        arc_index: int | None = None,
        arc_name: str | None = None,
        name: str | None = None,
    ) -> SampledMeasurement:
        """Build a sample using the inverse Cholesky factor of a covariance."""
        output_dimension = int(model.output_dimension)
        covariance_arr = np.asarray(covariance, dtype=float)
        if covariance_arr.shape != (output_dimension, output_dimension):
            raise ValueError(
                "covariance must be square with size equal to output_dimension."
            )
        chol = np.linalg.cholesky(covariance_arr)
        weight_matrix = np.linalg.solve(chol, np.eye(output_dimension, dtype=float))
        return cls(
            model=model,
            value=value,
            sample_time_s=sample_time_s,
            arc_index=arc_index,
            arc_name=arc_name,
            weight_matrix=weight_matrix,
            name=name,
        )


class InertialPositionMeasurementModel:
    """Exact inertial-position observation model for the 7-state GEqOE system."""

    output_dimension = 3
    output_names = ("x_km", "y_km", "z_km")
    _position_cfunc: ClassVar[object | None] = None
    _jacobian_cfunc: ClassVar[object | None] = None

    @classmethod
    def _compiled_functions(cls):
        if cls._position_cfunc is not None and cls._jacobian_cfunc is not None:
            return cls._position_cfunc, cls._jacobian_cfunc

        nu, p1, p2, K, q1, q2, m, mu = hy.make_vars(
            "nu", "p1", "p2", "K", "q1", "q2", "m", "mu"
        )

        sinK = hy.sin(K)
        cosK = hy.cos(K)
        a = (mu / (nu * nu)) ** (1.0 / 3.0)
        g2 = p1 * p1 + p2 * p2
        beta = hy.sqrt(1.0 - g2)
        alpha = 1.0 / (1.0 + beta)

        X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1 * p1) * cosK - p2)
        Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2 * p2) * sinK - p1)

        q1s = q1 * q1
        q2s = q2 * q2
        q1q2 = q1 * q2
        gamma_inv = 1.0 / (1.0 + q1s + q2s)
        eX = [
            gamma_inv * (1.0 - q1s + q2s),
            gamma_inv * (2.0 * q1q2),
            gamma_inv * (-2.0 * q1),
        ]
        eY = [
            gamma_inv * (2.0 * q1q2),
            gamma_inv * (1.0 + q1s - q2s),
            gamma_inv * (2.0 * q2),
        ]

        position_exprs = [
            X * eX[0] + Y * eY[0],
            X * eX[1] + Y * eY[1],
            X * eX[2] + Y * eY[2],
        ]
        state_vars = (nu, p1, p2, K, q1, q2, m)
        jacobian_exprs = [
            hy.diff(expr, var) for expr in position_exprs for var in state_vars
        ]
        vars_list = [*state_vars, mu]
        cls._position_cfunc = hy.cfunc(position_exprs, vars_list)
        cls._jacobian_cfunc = hy.cfunc(jacobian_exprs, vars_list)
        return cls._position_cfunc, cls._jacobian_cfunc

    @staticmethod
    def _mu_from_perturbation(perturbation: PerturbationModel) -> float:
        return float(getattr(perturbation, "mu", MU))

    def evaluate(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        del time_s
        position_func, _ = self._compiled_functions()
        state_arr = np.asarray(state, dtype=float)[:STATE_DIM]
        inputs = np.concatenate([state_arr, [self._mu_from_perturbation(perturbation)]])
        return np.asarray(position_func(inputs), dtype=float)

    def state_jacobian(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        del time_s
        _, jacobian_func = self._compiled_functions()
        state_arr = np.asarray(state, dtype=float)[:STATE_DIM]
        inputs = np.concatenate([state_arr, [self._mu_from_perturbation(perturbation)]])
        jacobian = np.asarray(jacobian_func(inputs), dtype=float)
        return jacobian.reshape(self.output_dimension, STATE_DIM)


class InertialRangeMeasurementModel:
    """One-way inertial range to a fixed inertial reference point."""

    output_dimension = 1
    output_names = ("range_km",)

    def __init__(self, reference_position_km: np.ndarray | list[float]):
        reference_position = np.asarray(reference_position_km, dtype=float)
        if reference_position.shape != (3,):
            raise ValueError(
                "InertialRangeMeasurementModel reference_position_km must have shape (3,)."
            )
        self._reference_position_km = reference_position
        self._position_model = InertialPositionMeasurementModel()

    @property
    def reference_position_km(self) -> np.ndarray:
        return self._reference_position_km.copy()

    def evaluate(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        position = self._position_model.evaluate(
            state,
            time_s=time_s,
            perturbation=perturbation,
        )
        return np.array(
            [np.linalg.norm(position - self._reference_position_km)],
            dtype=float,
        )

    def state_jacobian(
        self,
        state: np.ndarray,
        *,
        time_s: float,
        perturbation: PerturbationModel,
    ) -> np.ndarray:
        position = self._position_model.evaluate(
            state,
            time_s=time_s,
            perturbation=perturbation,
        )
        diff = position - self._reference_position_km
        rho = np.linalg.norm(diff)
        if rho <= 0.0:
            raise ValueError(
                "InertialRangeMeasurementModel is singular at zero observer-target separation."
            )
        position_jacobian = self._position_model.state_jacobian(
            state,
            time_s=time_s,
            perturbation=perturbation,
        )
        line_of_sight = diff / rho
        return np.array(line_of_sight[None, :] @ position_jacobian, dtype=float)


@dataclass(frozen=True)
class ShootingArc:
    """Single propagation arc in a multiple-shooting transcription.

    Args:
        perturbation: 7-state GEqOE perturbation model.
        initial_state: nominal arc-initial state ``(nu, p1, p2, K, q1, q2, m)``.
        duration_s: arc duration in seconds.
        parameter_names: runtime parameters included in the decision vector.
            ``None`` defaults to all runtime parameters except ``mu``.
        start_time_s: optional absolute start time. If omitted, the arc uses a
            local time origin ``t = 0`` unless the problem is built with
            ``sequential_start_times=True``.
        name: optional arc label used in decision/constraint names.
    """

    perturbation: PerturbationModel
    initial_state: np.ndarray | list[float]
    duration_s: float
    parameter_names: tuple[str, ...] | list[str] | None = None
    start_time_s: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        initial_state = np.asarray(self.initial_state, dtype=float)
        if initial_state.shape != (STATE_DIM,):
            raise ValueError(
                "ShootingArc.initial_state must have shape (7,) for "
                "[nu, p1, p2, K, q1, q2, m]."
            )
        if self.duration_s <= 0.0:
            raise ValueError("ShootingArc.duration_s must be positive.")
        object.__setattr__(self, "initial_state", initial_state)
        if self.parameter_names is not None:
            object.__setattr__(self, "parameter_names", tuple(self.parameter_names))


@dataclass(frozen=True)
class ArcPropagationResult:
    """Propagated endpoint data for one shooting arc."""

    name: str
    start_time_s: float
    duration_s: float
    initial_state: np.ndarray
    final_state: np.ndarray
    state_jacobian: np.ndarray
    parameter_jacobian: np.ndarray
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ShootingEvaluation:
    """Evaluated multiple-shooting residual data."""

    arc_results: tuple[ArcPropagationResult, ...]
    continuity_residual: np.ndarray
    continuity_jacobian: csr_matrix


@dataclass(frozen=True)
class SampledMeasurementResult:
    """Predicted/observed data for one sampled measurement."""

    name: str
    arc_name: str
    arc_index: int
    sample_time_s: float
    absolute_time_s: float
    component_names: tuple[str, ...]
    observed: np.ndarray
    predicted: np.ndarray
    raw_residual: np.ndarray
    weighted_residual: np.ndarray


@dataclass(frozen=True)
class MeasurementResidualEvaluation:
    """Weighted sampled-measurement residuals attached to a shooting evaluation."""

    shooting_evaluation: ShootingEvaluation
    sample_results: tuple[SampledMeasurementResult, ...]
    residual: np.ndarray
    jacobian: csr_matrix
    residual_names: tuple[str, ...]


@dataclass(frozen=True)
class ShootingOptimizationResult:
    """Result bundle for the SciPy-based multiple-shooting solve helper."""

    x: np.ndarray
    objective: float
    evaluation: ShootingEvaluation
    continuity_residual: np.ndarray
    terminal_outputs: np.ndarray | None
    terminal_violation: np.ndarray | None
    terminal_residual: np.ndarray | None
    scipy_result: OptimizeResult


@dataclass(frozen=True)
class TerminalConstraintSpec:
    """Bounds on selected terminal outputs of the final shooting arc."""

    lower: np.ndarray | list[float] | None = None
    upper: np.ndarray | list[float] | None = None
    output_indices: tuple[int, ...] | list[int] | np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError(
                "TerminalConstraintSpec requires at least one of lower or upper."
            )

        if self.output_indices is not None:
            output_indices = tuple(int(v) for v in np.asarray(self.output_indices, dtype=int))
            object.__setattr__(self, "output_indices", output_indices)
            n_out = len(output_indices)
        else:
            n_out = None

        lower = None
        if self.lower is not None:
            lower = np.atleast_1d(np.asarray(self.lower, dtype=float))
            if lower.ndim != 1:
                raise ValueError("TerminalConstraintSpec.lower must be a 1-D array.")
            if n_out is not None and lower.shape != (n_out,):
                raise ValueError(
                    "TerminalConstraintSpec.lower must match output_indices length."
                )
            n_out = lower.shape[0]
            object.__setattr__(self, "lower", lower)

        upper = None
        if self.upper is not None:
            upper = np.atleast_1d(np.asarray(self.upper, dtype=float))
            if upper.ndim != 1:
                raise ValueError("TerminalConstraintSpec.upper must be a 1-D array.")
            if n_out is not None and upper.shape != (n_out,):
                raise ValueError(
                    "TerminalConstraintSpec.upper must match output_indices length."
                )
            n_out = upper.shape[0]
            object.__setattr__(self, "upper", upper)

        if lower is not None and upper is not None and lower.shape != upper.shape:
            raise ValueError("TerminalConstraintSpec lower/upper must have the same shape.")

    @classmethod
    def equality(
        cls,
        target_state,
        output_indices: tuple[int, ...] | list[int] | np.ndarray | None = None,
    ) -> TerminalConstraintSpec:
        """Build an equality terminal constraint from a target state or subset."""
        target = np.atleast_1d(np.asarray(target_state, dtype=float))
        if output_indices is None:
            if target.shape not in ((STATE_DIM,),):
                raise ValueError(
                    "TerminalConstraintSpec.equality() without output_indices "
                    "expects a 7-element target_state."
                )
            lower = upper = target.copy()
        else:
            idx = np.asarray(output_indices, dtype=int)
            if target.shape == (STATE_DIM,):
                lower = upper = target[idx].copy()
            elif target.shape == (len(idx),):
                lower = upper = target.copy()
            else:
                raise ValueError(
                    "TerminalConstraintSpec.equality() target size must match "
                    "the full 7-state vector or the selected output_indices."
                )
        return cls(lower=lower, upper=upper, output_indices=output_indices)

    @property
    def is_equality(self) -> bool:
        """Whether the terminal bounds collapse to equalities."""
        return (
            self.lower is not None
            and self.upper is not None
            and np.array_equal(self.lower, self.upper)
        )


@dataclass(frozen=True)
class SmoothnessPenaltySpec:
    """Quadratic smoothness penalty across arc-local decision variables.

    The keys are decision-name selectors accepted by ``build_named_bounds()``:
    either exact names such as ``arc0.thrust.t_newtons`` or shared suffixes such
    as ``thrust.t_newtons`` applied across all arcs.
    """

    selector_weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.selector_weights:
            raise ValueError("SmoothnessPenaltySpec requires at least one selector.")
        normalized = {str(k): float(v) for k, v in self.selector_weights.items()}
        if any(weight < 0.0 for weight in normalized.values()):
            raise ValueError("SmoothnessPenaltySpec weights must be non-negative.")
        object.__setattr__(self, "selector_weights", normalized)


@dataclass(frozen=True)
class MeasurementObjectiveSpec:
    """Weighted least-squares objective built from sampled measurements."""

    measurements: tuple[SampledMeasurement, ...] | list[SampledMeasurement]
    weight: float = 1.0
    hessian_mode: Literal["quasi-newton", "gauss-newton"] = "quasi-newton"

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("MeasurementObjectiveSpec.weight must be non-negative.")
        if self.hessian_mode not in {"quasi-newton", "gauss-newton"}:
            raise ValueError(
                "MeasurementObjectiveSpec.hessian_mode must be "
                "'quasi-newton' or 'gauss-newton'."
            )
        normalized_measurements = tuple(self.measurements)
        if len(normalized_measurements) == 0:
            raise ValueError("MeasurementObjectiveSpec requires at least one measurement.")
        object.__setattr__(self, "measurements", normalized_measurements)


@dataclass(frozen=True)
class DecisionTrackingTerm:
    """Quadratic tracking term on selector-matched decision variables."""

    selector: str
    target: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("DecisionTrackingTerm.sigma must be strictly positive.")

    @property
    def weight(self) -> float:
        return 1.0 / (self.sigma * self.sigma)


@dataclass(frozen=True)
class DecisionTrackingPenaltySpec:
    """Collection of quadratic tracking terms for estimation objectives."""

    terms: tuple[DecisionTrackingTerm, ...] | list[DecisionTrackingTerm]

    def __post_init__(self) -> None:
        normalized_terms = tuple(self.terms)
        if len(normalized_terms) == 0:
            raise ValueError(
                "DecisionTrackingPenaltySpec requires at least one tracking term."
            )
        object.__setattr__(self, "terms", normalized_terms)


@dataclass(frozen=True)
class ShootingSolveSpec:
    """Configuration for the SciPy-based multiple-shooting prototype solve."""

    bounds: Bounds | None = None
    measurement_objective: MeasurementObjectiveSpec | None = None
    decision_tracking_penalty: DecisionTrackingPenaltySpec | None = None
    terminal_constraint: TerminalConstraintSpec | None = None
    smoothness_penalty: SmoothnessPenaltySpec | None = None
    options: dict | None = None


@dataclass(frozen=True)
class _ArcLayout:
    name: str
    start_time_s: float
    duration_s: float
    state_slice: slice
    parameter_slice: slice
    parameter_names: tuple[str, ...]
    parameter_indices: tuple[int, ...]
    parameter_defaults: np.ndarray


@dataclass(frozen=True)
class _ResolvedMeasurement:
    measurement: SampledMeasurement
    arc_index: int
    arc_name: str
    name: str


class MultiArcShootingProblem:
    """Assemble multiple-shooting constraints on top of GEqOE thrust sensitivities.

    The flat decision vector is grouped per arc as:

    ``[x_0, p_0, x_1, p_1, ..., x_{N-1}, p_{N-1}]``

    where each ``x_i`` is a 7-state GEqOE + mass node and each ``p_i`` is the
    selected subset of arc-local runtime parameters.

    By default, arcs use local time ``t=0`` unless ``ShootingArc.start_time_s``
    is provided explicitly. Set ``sequential_start_times=True`` to assign
    cumulative absolute start times automatically.
    """

    state_dim = STATE_DIM
    state_names = STATE_NAMES

    def __init__(
        self,
        arcs: list[ShootingArc] | tuple[ShootingArc, ...],
        tol: float = 1e-15,
        compact_mode: bool = True,
        sequential_start_times: bool = False,
    ):
        if len(arcs) == 0:
            raise ValueError("MultiArcShootingProblem requires at least one arc.")

        self._arcs = tuple(arcs)
        self._tol = float(tol)
        self._compact_mode = bool(compact_mode)
        self._integrators = []
        self._templates = []
        self._layouts: list[_ArcLayout] = []
        self._decision_names: list[str] = []
        self._continuity_constraint_names: list[str] = []
        self._decision_index_map: dict[str, int] = {}
        self._arc_name_to_index: dict[str, int] = {}

        start_time_cursor = 0.0
        decision_offset = 0
        for i, arc in enumerate(self._arcs):
            if arc.start_time_s is None:
                start_time_s = start_time_cursor if sequential_start_times else 0.0
            else:
                start_time_s = float(arc.start_time_s)
            start_time_cursor = start_time_s + float(arc.duration_s)
            name = arc.name or f"arc{i}"

            ta, par_map = build_thrust_sensitivity_integrator(
                arc.perturbation,
                arc.initial_state,
                t0=start_time_s,
                tol=self._tol,
                compact_mode=self._compact_mode,
            )
            all_param_names = tuple(parameter_names_from_map(par_map))
            if arc.parameter_names is None:
                parameter_names = tuple(name for name in all_param_names if name != "mu")
            else:
                missing = set(arc.parameter_names) - set(all_param_names)
                if missing:
                    missing_names = ", ".join(sorted(missing))
                    raise KeyError(
                        f"Arc {name!r} selected unknown runtime parameters: {missing_names}"
                    )
                parameter_names = tuple(arc.parameter_names)
            parameter_indices = tuple(par_map[param_name] for param_name in parameter_names)
            parameter_defaults = np.array(ta.pars, dtype=float, copy=True)

            state_slice = slice(decision_offset, decision_offset + STATE_DIM)
            decision_offset += STATE_DIM
            parameter_slice = slice(
                decision_offset,
                decision_offset + len(parameter_names),
            )
            decision_offset += len(parameter_names)

            self._layouts.append(
                _ArcLayout(
                    name=name,
                    start_time_s=start_time_s,
                    duration_s=float(arc.duration_s),
                    state_slice=state_slice,
                    parameter_slice=parameter_slice,
                    parameter_names=parameter_names,
                    parameter_indices=parameter_indices,
                    parameter_defaults=parameter_defaults,
                )
            )
            self._integrators.append((ta, par_map))
            self._templates.append(np.array(ta.state, dtype=float, copy=True))

            self._decision_names.extend(
                [f"{name}.{state_name}" for state_name in STATE_NAMES]
            )
            self._decision_names.extend(
                [f"{name}.{param_name}" for param_name in parameter_names]
            )

        self._arc_name_to_index = {}
        for i, layout in enumerate(self._layouts):
            if layout.name in self._arc_name_to_index:
                raise ValueError(f"Duplicate shooting arc name: {layout.name!r}")
            self._arc_name_to_index[layout.name] = i

        for i in range(len(self._layouts) - 1):
            left = self._layouts[i].name
            right = self._layouts[i + 1].name
            self._continuity_constraint_names.extend(
                [f"{left}->{right}.{state_name}" for state_name in STATE_NAMES]
            )

        self._decision_index_map = {
            name: i for i, name in enumerate(self._decision_names)
        }

    @property
    def num_arcs(self) -> int:
        return len(self._layouts)

    @property
    def decision_size(self) -> int:
        return len(self._decision_names)

    @property
    def continuity_size(self) -> int:
        return max(self.num_arcs - 1, 0) * STATE_DIM

    @property
    def decision_names(self) -> tuple[str, ...]:
        return tuple(self._decision_names)

    @property
    def continuity_constraint_names(self) -> tuple[str, ...]:
        return tuple(self._continuity_constraint_names)

    def decision_index(self, name: str) -> int:
        """Return the flat decision-vector index for a named variable."""
        return self._decision_index_map[name]

    def _decision_indices_for_selector(self, selector: str) -> list[int]:
        if selector in self._decision_index_map:
            return [self._decision_index_map[selector]]
        suffix = f".{selector}"
        indices = [
            i for i, name in enumerate(self._decision_names) if name.endswith(suffix)
        ]
        if not indices:
            raise KeyError(f"Unknown decision selector: {selector!r}")
        return indices

    def build_named_bounds(
        self,
        lower: dict[str, float] | None = None,
        upper: dict[str, float] | None = None,
        default_lower: float = -np.inf,
        default_upper: float = np.inf,
    ) -> Bounds:
        """Build SciPy bounds from exact names or shared suffix selectors.

        Selectors may be full decision names like ``arc0.thrust.t_newtons`` or
        shared suffixes like ``m`` / ``thrust.t_newtons`` that apply to every
        matching arc-local variable.
        """
        lb = np.full(self.decision_size, float(default_lower), dtype=float)
        ub = np.full(self.decision_size, float(default_upper), dtype=float)

        if lower is not None:
            for selector, value in lower.items():
                for idx in self._decision_indices_for_selector(selector):
                    lb[idx] = float(value)
        if upper is not None:
            for selector, value in upper.items():
                for idx in self._decision_indices_for_selector(selector):
                    ub[idx] = float(value)

        return Bounds(lb, ub)

    def _normalize_output_indices(
        self,
        output_indices: list[int] | tuple[int, ...] | np.ndarray | None,
    ) -> np.ndarray:
        if output_indices is None:
            row_idx = np.arange(STATE_DIM, dtype=int)
        else:
            row_idx = np.asarray(output_indices, dtype=int)
            if row_idx.ndim != 1:
                raise ValueError("output_indices must be a 1-D array of integers.")
        return row_idx

    def initial_guess(self) -> np.ndarray:
        """Return the nominal decision vector from arc states and defaults."""
        guess = np.zeros(self.decision_size, dtype=float)
        for arc, layout in zip(self._arcs, self._layouts, strict=True):
            guess[layout.state_slice] = arc.initial_state
            if layout.parameter_names:
                guess[layout.parameter_slice] = layout.parameter_defaults[
                    list(layout.parameter_indices)
                ]
        return guess

    def _validate_decision_vector(self, decision_vector) -> np.ndarray:
        vector = np.asarray(decision_vector, dtype=float)
        if vector.shape != (self.decision_size,):
            raise ValueError(
                f"Expected decision vector of shape ({self.decision_size},), "
                f"got {vector.shape}."
            )
        return vector

    def _resolve_measurements(
        self,
        measurements: list[SampledMeasurement] | tuple[SampledMeasurement, ...],
    ) -> tuple[_ResolvedMeasurement, ...]:
        resolved: list[_ResolvedMeasurement] = []
        for i, measurement in enumerate(measurements):
            if measurement.arc_name is not None:
                try:
                    arc_index = self._arc_name_to_index[measurement.arc_name]
                except KeyError as exc:
                    raise KeyError(
                        f"Unknown measurement arc_name: {measurement.arc_name!r}"
                    ) from exc
                if measurement.arc_index is not None and arc_index != int(measurement.arc_index):
                    raise ValueError(
                        "SampledMeasurement arc_index/arc_name selectors disagree."
                    )
            else:
                assert measurement.arc_index is not None
                arc_index = int(measurement.arc_index)
                if arc_index >= self.num_arcs:
                    raise IndexError(
                        f"Measurement arc_index {arc_index} is out of range for "
                        f"{self.num_arcs} arcs."
                    )

            layout = self._layouts[arc_index]
            sample_time_s = float(measurement.sample_time_s)
            if sample_time_s > layout.duration_s + 1.0e-12:
                raise ValueError(
                    "SampledMeasurement.sample_time_s must lie within the "
                    "selected arc duration."
                )
            resolved.append(
                _ResolvedMeasurement(
                    measurement=measurement,
                    arc_index=arc_index,
                    arc_name=layout.name,
                    name=measurement.name or f"measurement{i}",
                )
            )
        return tuple(resolved)

    def _propagate_arcs(
        self,
        decision_vector,
        sample_times_by_arc: dict[int, tuple[float, ...]] | None = None,
    ) -> tuple[ShootingEvaluation, tuple[dict[float, np.ndarray], ...]]:
        x = self._validate_decision_vector(decision_vector)
        arc_results: list[ArcPropagationResult] = []
        sample_cache_by_arc: list[dict[float, np.ndarray]] = []

        for arc_index, (layout, (ta, par_map), template) in enumerate(
            zip(self._layouts, self._integrators, self._templates, strict=True)
        ):
            arc_state = x[layout.state_slice]
            if layout.parameter_names:
                arc_parameters = x[layout.parameter_slice]
            else:
                arc_parameters = np.zeros(0, dtype=float)

            def _reset_integrator() -> None:
                ta.time = layout.start_time_s
                ta.state[:] = template
                ta.pars[:] = layout.parameter_defaults
                ta.state[:STATE_DIM] = arc_state
                if layout.parameter_names:
                    ta.pars[list(layout.parameter_indices)] = arc_parameters

            _reset_integrator()

            requested_times = ()
            if sample_times_by_arc is not None:
                requested_times = sample_times_by_arc.get(arc_index, ())
            grid_rel = sorted(
                {0.0} | {float(t) for t in requested_times} | {layout.duration_s}
            )
            grid_abs = layout.start_time_s + np.asarray(grid_rel, dtype=float)
            states_grid = propagate_grid(ta, grid_abs)
            if len(states_grid) == len(grid_rel):
                sample_cache = {
                    rel_time: np.array(state_aug, dtype=float, copy=True)
                    for rel_time, state_aug in zip(grid_rel, states_grid, strict=True)
                }
            elif len(states_grid) == len(grid_rel) - 1 and grid_rel[0] == 0.0:
                sample_cache = {
                    0.0: np.array(ta.state, dtype=float, copy=True),
                    **{
                        rel_time: np.array(state_aug, dtype=float, copy=True)
                        for rel_time, state_aug in zip(
                            grid_rel[1:],
                            states_grid,
                            strict=True,
                        )
                    },
                }
            else:
                ta, par_map = build_thrust_sensitivity_integrator(
                    self._arcs[arc_index].perturbation,
                    arc_state,
                    t0=layout.start_time_s,
                    tol=self._tol,
                    compact_mode=self._compact_mode,
                )
                self._integrators[arc_index] = (ta, par_map)
                template = np.array(ta.state, dtype=float, copy=True)
                self._templates[arc_index] = template
                _reset_integrator()
                states_grid = propagate_grid(ta, grid_abs)
                if len(states_grid) == len(grid_rel):
                    sample_cache = {
                        rel_time: np.array(state_aug, dtype=float, copy=True)
                        for rel_time, state_aug in zip(
                            grid_rel,
                            states_grid,
                            strict=True,
                        )
                    }
                elif len(states_grid) == len(grid_rel) - 1 and grid_rel[0] == 0.0:
                    sample_cache = {
                        0.0: np.array(ta.state, dtype=float, copy=True),
                        **{
                            rel_time: np.array(state_aug, dtype=float, copy=True)
                            for rel_time, state_aug in zip(
                                grid_rel[1:],
                                states_grid,
                                strict=True,
                            )
                        },
                    }
                else:
                    sample_cache = {
                        0.0: np.array(ta.state, dtype=float, copy=True),
                    }
                    for rel_time in grid_rel[1:]:
                        ta.propagate_until(layout.start_time_s + rel_time)
                        sample_cache[rel_time] = np.array(
                            ta.state,
                            dtype=float,
                            copy=True,
                        )

            final_state, phi_x, phi_p_all, all_param_names = extract_variational_matrices(
                sample_cache[layout.duration_s],
                state_dim=STATE_DIM,
                par_map=par_map,
            )
            if layout.parameter_names:
                name_to_col = {name: i for i, name in enumerate(all_param_names)}
                col_idx = [name_to_col[name] for name in layout.parameter_names]
                phi_p = phi_p_all[:, col_idx]
            else:
                phi_p = np.zeros((STATE_DIM, 0))

            arc_results.append(
                ArcPropagationResult(
                    name=layout.name,
                    start_time_s=layout.start_time_s,
                    duration_s=layout.duration_s,
                    initial_state=arc_state.copy(),
                    final_state=np.array(final_state, dtype=float, copy=True),
                    state_jacobian=np.array(phi_x, dtype=float, copy=True),
                    parameter_jacobian=np.array(phi_p, dtype=float, copy=True),
                    parameter_names=layout.parameter_names,
                )
            )
            sample_cache_by_arc.append(sample_cache)

        continuity_residual = np.zeros(self.continuity_size, dtype=float)
        continuity_jacobian = lil_matrix(
            (self.continuity_size, self.decision_size),
            dtype=float,
        )

        for i in range(self.num_arcs - 1):
            rows = slice(i * STATE_DIM, (i + 1) * STATE_DIM)
            current = arc_results[i]
            nxt = arc_results[i + 1]
            current_layout = self._layouts[i]
            next_layout = self._layouts[i + 1]

            continuity_residual[rows] = current.final_state - nxt.initial_state
            continuity_jacobian[rows, current_layout.state_slice] = current.state_jacobian
            if current_layout.parameter_names:
                continuity_jacobian[rows, current_layout.parameter_slice] = (
                    current.parameter_jacobian
                )
            continuity_jacobian[rows, next_layout.state_slice] = -np.eye(STATE_DIM)

        evaluation = ShootingEvaluation(
            arc_results=tuple(arc_results),
            continuity_residual=continuity_residual,
            continuity_jacobian=continuity_jacobian.tocsr(),
        )
        return evaluation, tuple(sample_cache_by_arc)

    def evaluate(self, decision_vector) -> ShootingEvaluation:
        """Propagate every arc and assemble the continuity residual/Jacobian."""
        evaluation, _ = self._propagate_arcs(decision_vector)
        return evaluation

    def continuity_constraints(
        self,
        decision_vector,
        evaluation: ShootingEvaluation | None = None,
    ) -> tuple[np.ndarray, csr_matrix]:
        """Return multiple-shooting continuity residuals and sparse Jacobian."""
        result = evaluation if evaluation is not None else self.evaluate(decision_vector)
        return result.continuity_residual, result.continuity_jacobian

    def terminal_outputs(
        self,
        decision_vector,
        output_indices: list[int] | np.ndarray | None = None,
        evaluation: ShootingEvaluation | None = None,
    ) -> tuple[np.ndarray, csr_matrix]:
        """Return selected final-arc outputs and their sparse Jacobian."""
        _ = self._validate_decision_vector(decision_vector)
        result = evaluation if evaluation is not None else self.evaluate(decision_vector)
        row_idx = self._normalize_output_indices(output_indices)

        terminal = result.arc_results[-1]
        layout = self._layouts[-1]
        outputs = terminal.final_state[row_idx]
        jac = lil_matrix((len(row_idx), self.decision_size), dtype=float)
        jac[:, layout.state_slice] = terminal.state_jacobian[row_idx, :]
        if layout.parameter_names:
            jac[:, layout.parameter_slice] = terminal.parameter_jacobian[row_idx, :]
        return outputs, jac.tocsr()

    def terminal_constraints(
        self,
        decision_vector,
        target_state,
        output_indices: list[int] | np.ndarray | None = None,
        evaluation: ShootingEvaluation | None = None,
    ) -> tuple[np.ndarray, csr_matrix]:
        """Return terminal residuals and Jacobian for selected endpoint outputs."""
        row_idx = self._normalize_output_indices(output_indices)
        outputs, jac = self.terminal_outputs(
            decision_vector,
            output_indices=row_idx,
            evaluation=evaluation,
        )
        target = np.atleast_1d(np.asarray(target_state, dtype=float))
        if target.shape == (STATE_DIM,):
            target_selected = target[row_idx]
        elif target.shape == (len(row_idx),):
            target_selected = target
        else:
            raise ValueError(
                "target_state must match the full 7-state vector or the selected "
                "terminal output dimension."
        )
        return outputs - target_selected, jac

    def evaluate_measurements(
        self,
        decision_vector,
        measurements: list[SampledMeasurement] | tuple[SampledMeasurement, ...],
    ) -> MeasurementResidualEvaluation:
        """Evaluate weighted sampled-measurement residuals and exact Jacobians."""
        x = self._validate_decision_vector(decision_vector)
        resolved_measurements = self._resolve_measurements(measurements)

        sample_times_by_arc: dict[int, tuple[float, ...]] = {}
        if resolved_measurements:
            grouped_times: dict[int, list[float]] = {}
            for resolved in resolved_measurements:
                grouped_times.setdefault(resolved.arc_index, []).append(
                    float(resolved.measurement.sample_time_s)
                )
            sample_times_by_arc = {
                arc_index: tuple(times) for arc_index, times in grouped_times.items()
            }

        shooting_evaluation, sample_cache_by_arc = self._propagate_arcs(
            x,
            sample_times_by_arc=sample_times_by_arc,
        )
        if not resolved_measurements:
            return MeasurementResidualEvaluation(
                shooting_evaluation=shooting_evaluation,
                sample_results=(),
                residual=np.zeros(0, dtype=float),
                jacobian=csr_matrix((0, self.decision_size), dtype=float),
                residual_names=(),
            )

        total_rows = sum(
            int(resolved.measurement.model.output_dimension)
            for resolved in resolved_measurements
        )
        residual = np.zeros(total_rows, dtype=float)
        jacobian = lil_matrix((total_rows, self.decision_size), dtype=float)
        sample_results: list[SampledMeasurementResult] = []
        residual_names: list[str] = []

        row_start = 0
        for resolved in resolved_measurements:
            measurement = resolved.measurement
            arc_index = resolved.arc_index
            layout = self._layouts[arc_index]
            arc = self._arcs[arc_index]
            _, par_map = self._integrators[arc_index]
            sample_time_s = float(measurement.sample_time_s)
            absolute_time_s = layout.start_time_s + sample_time_s

            sampled_state_aug = sample_cache_by_arc[arc_index][sample_time_s]
            state, phi_x, phi_p_all, all_param_names = extract_variational_matrices(
                sampled_state_aug,
                state_dim=STATE_DIM,
                par_map=par_map,
            )
            if layout.parameter_names:
                name_to_col = {name: i for i, name in enumerate(all_param_names)}
                col_idx = [name_to_col[name] for name in layout.parameter_names]
                phi_p = phi_p_all[:, col_idx]
            else:
                phi_p = np.zeros((STATE_DIM, 0))

            output_dimension = int(measurement.model.output_dimension)
            output_names = _normalize_output_names(measurement.model, output_dimension)
            rows = slice(row_start, row_start + output_dimension)
            row_start += output_dimension

            predicted = np.atleast_1d(
                np.asarray(
                    measurement.model.evaluate(
                        state,
                        time_s=absolute_time_s,
                        perturbation=arc.perturbation,
                    ),
                    dtype=float,
                )
            )
            if predicted.shape != (output_dimension,):
                raise ValueError(
                    "MeasurementModel.evaluate() returned an unexpected shape."
                )

            state_jacobian = np.asarray(
                measurement.model.state_jacobian(
                    state,
                    time_s=absolute_time_s,
                    perturbation=arc.perturbation,
                ),
                dtype=float,
            )
            if state_jacobian.shape != (output_dimension, STATE_DIM):
                raise ValueError(
                    "MeasurementModel.state_jacobian() returned an unexpected shape."
                )

            raw_residual = predicted - measurement.value
            weighted_residual = measurement.weight_matrix @ raw_residual
            weighted_state_jacobian = measurement.weight_matrix @ state_jacobian

            residual[rows] = weighted_residual
            jacobian[rows, layout.state_slice] = weighted_state_jacobian @ phi_x
            if layout.parameter_names:
                jacobian[rows, layout.parameter_slice] = (
                    weighted_state_jacobian @ phi_p
                )

            component_names = tuple(
                f"{resolved.name}.{component}" for component in output_names
            )
            residual_names.extend(component_names)
            sample_results.append(
                SampledMeasurementResult(
                    name=resolved.name,
                    arc_name=resolved.arc_name,
                    arc_index=arc_index,
                    sample_time_s=sample_time_s,
                    absolute_time_s=absolute_time_s,
                    component_names=component_names,
                    observed=measurement.value.copy(),
                    predicted=predicted.copy(),
                    raw_residual=raw_residual,
                    weighted_residual=weighted_residual,
                )
            )

        return MeasurementResidualEvaluation(
            shooting_evaluation=shooting_evaluation,
            sample_results=tuple(sample_results),
            residual=residual,
            jacobian=jacobian.tocsr(),
            residual_names=tuple(residual_names),
        )

    def measurement_residuals(
        self,
        decision_vector,
        measurements: list[SampledMeasurement] | tuple[SampledMeasurement, ...],
    ) -> tuple[np.ndarray, csr_matrix]:
        """Return weighted sampled-measurement residuals and sparse Jacobian."""
        evaluation = self.evaluate_measurements(decision_vector, measurements)
        return evaluation.residual, evaluation.jacobian

    def measurement_objective(
        self,
        decision_vector,
        measurements: list[SampledMeasurement] | tuple[SampledMeasurement, ...],
        evaluation: MeasurementResidualEvaluation | None = None,
        weight: float = 1.0,
    ) -> tuple[float, np.ndarray, csr_matrix]:
        """Return a weighted least-squares objective and Gauss-Newton Hessian."""
        _ = self._validate_decision_vector(decision_vector)
        if weight < 0.0:
            raise ValueError("measurement objective weight must be non-negative.")
        result = (
            evaluation
            if evaluation is not None
            else self.evaluate_measurements(decision_vector, measurements)
        )
        residual = result.residual
        jacobian = result.jacobian
        value = 0.5 * weight * float(residual @ residual)
        gradient = np.asarray(weight * (jacobian.T @ residual), dtype=float).ravel()
        hessian = (weight * (jacobian.T @ jacobian)).tocsr()
        return value, gradient, hessian

    def decision_tracking_objective(
        self,
        decision_vector,
        tracking_penalty: DecisionTrackingPenaltySpec,
    ) -> tuple[float, np.ndarray, csr_matrix]:
        """Quadratic tracking penalties on selector-matched decision variables."""
        x = self._validate_decision_vector(decision_vector)

        value = 0.0
        gradient = np.zeros(self.decision_size, dtype=float)
        hessian = lil_matrix((self.decision_size, self.decision_size), dtype=float)

        for term in tracking_penalty.terms:
            indices = self._decision_indices_for_selector(term.selector)
            for idx in indices:
                diff = x[idx] - term.target
                value += 0.5 * term.weight * diff * diff
                gradient[idx] += term.weight * diff
                hessian[idx, idx] += term.weight

        return float(value), gradient, hessian.tocsr()

    def minimum_propellant_objective(
        self,
        decision_vector,
        evaluation: ShootingEvaluation | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return propellant use and exact gradient for the transcribed problem."""
        x = self._validate_decision_vector(decision_vector)
        result = evaluation if evaluation is not None else self.evaluate(x)

        first_layout = self._layouts[0]
        last_layout = self._layouts[-1]
        last_arc = result.arc_results[-1]

        initial_mass_index = first_layout.state_slice.start + 6
        objective = x[initial_mass_index] - last_arc.final_state[6]

        gradient = np.zeros(self.decision_size, dtype=float)
        gradient[initial_mass_index] = 1.0
        gradient[last_layout.state_slice] -= last_arc.state_jacobian[6, :]
        if last_layout.parameter_names:
            gradient[last_layout.parameter_slice] -= last_arc.parameter_jacobian[6, :]
        return float(objective), gradient

    def minimum_propellant_hessian(self) -> csr_matrix:
        """Return the exact Hessian of the minimum-propellant objective."""
        return csr_matrix((self.decision_size, self.decision_size), dtype=float)

    def control_smoothness_objective(
        self,
        decision_vector,
        selector_weights: dict[str, float] | SmoothnessPenaltySpec,
    ) -> tuple[float, np.ndarray, csr_matrix]:
        """Quadratic penalty on selector-matched decision differences across arcs."""
        x = self._validate_decision_vector(decision_vector)
        if isinstance(selector_weights, SmoothnessPenaltySpec):
            weights_map = selector_weights.selector_weights
        else:
            weights_map = {str(k): float(v) for k, v in selector_weights.items()}

        value = 0.0
        gradient = np.zeros(self.decision_size, dtype=float)
        hessian = lil_matrix((self.decision_size, self.decision_size), dtype=float)

        for selector, weight in weights_map.items():
            if weight < 0.0:
                raise ValueError("Smoothness weights must be non-negative.")
            indices = self._decision_indices_for_selector(selector)
            if len(indices) < 2 or weight == 0.0:
                continue
            for idx_left, idx_right in zip(indices[:-1], indices[1:], strict=True):
                diff = x[idx_right] - x[idx_left]
                value += 0.5 * weight * diff * diff
                gradient[idx_left] -= weight * diff
                gradient[idx_right] += weight * diff
                hessian[idx_left, idx_left] += weight
                hessian[idx_right, idx_right] += weight
                hessian[idx_left, idx_right] -= weight
                hessian[idx_right, idx_left] -= weight

        return float(value), gradient, hessian.tocsr()

    def solve(
        self,
        spec: ShootingSolveSpec,
        decision_vector0=None,
    ) -> ShootingOptimizationResult:
        """Solve the current shooting transcription with SciPy trust-constr."""
        if decision_vector0 is None:
            x0 = self.initial_guess()
        else:
            x0 = self._validate_decision_vector(decision_vector0)

        bounds = spec.bounds if spec.bounds is not None else self.build_named_bounds()
        measurement_hessian_mode = (
            None
            if spec.measurement_objective is None
            else spec.measurement_objective.hessian_mode
        )
        use_explicit_objective_hessian = (
            spec.measurement_objective is None
            or measurement_hessian_mode == "gauss-newton"
        )
        if (
            measurement_hessian_mode == "quasi-newton"
            and self.continuity_size == 0
            and spec.terminal_constraint is None
        ):
            use_explicit_objective_hessian = True

        cache: dict[str, object] = {
            "x": None,
            "evaluation": None,
            "measurement_evaluation": None,
            "objective": None,
            "gradient": None,
            "hessian": None,
            "terminal_outputs": None,
        }

        def _update_cache(x) -> None:
            x_vec = self._validate_decision_vector(x)
            cached_x = cache["x"]
            if cached_x is not None and np.array_equal(cached_x, x_vec):
                return

            measurement_evaluation = None
            if spec.measurement_objective is not None:
                measurement_evaluation = self.evaluate_measurements(
                    x_vec,
                    spec.measurement_objective.measurements,
                )
                evaluation = measurement_evaluation.shooting_evaluation
                if use_explicit_objective_hessian:
                    objective, gradient, hessian = self.measurement_objective(
                        x_vec,
                        spec.measurement_objective.measurements,
                        evaluation=measurement_evaluation,
                        weight=spec.measurement_objective.weight,
                    )
                else:
                    residual = measurement_evaluation.residual
                    jacobian = measurement_evaluation.jacobian
                    weight = spec.measurement_objective.weight
                    objective = 0.5 * weight * float(residual @ residual)
                    gradient = np.asarray(
                        weight * (jacobian.T @ residual),
                        dtype=float,
                    ).ravel()
                    hessian = None
            else:
                evaluation = self.evaluate(x_vec)
                objective, gradient = self.minimum_propellant_objective(
                    x_vec, evaluation=evaluation
                )
                hessian = self.minimum_propellant_hessian()

            if spec.decision_tracking_penalty is not None:
                tracking_value, tracking_gradient, tracking_hessian = (
                    self.decision_tracking_objective(
                        x_vec,
                        spec.decision_tracking_penalty,
                    )
                )
                objective += tracking_value
                gradient += tracking_gradient
                if use_explicit_objective_hessian:
                    assert hessian is not None
                    hessian = hessian + tracking_hessian

            if spec.smoothness_penalty is not None:
                smooth_value, smooth_gradient, smooth_hessian = (
                    self.control_smoothness_objective(
                        x_vec, spec.smoothness_penalty
                    )
                )
                objective += smooth_value
                gradient += smooth_gradient
                if use_explicit_objective_hessian:
                    assert hessian is not None
                    hessian = hessian + smooth_hessian

            terminal_outputs = None
            if spec.terminal_constraint is not None:
                terminal_outputs = self.terminal_outputs(
                    x_vec,
                    output_indices=spec.terminal_constraint.output_indices,
                    evaluation=evaluation,
                )

            cache["x"] = np.array(x_vec, dtype=float, copy=True)
            cache["evaluation"] = evaluation
            cache["measurement_evaluation"] = measurement_evaluation
            cache["objective"] = float(objective)
            cache["gradient"] = np.array(gradient, dtype=float, copy=True)
            cache["hessian"] = hessian
            cache["terminal_outputs"] = terminal_outputs

        def _objective(x) -> float:
            _update_cache(x)
            return cache["objective"]  # type: ignore[return-value]

        def _objective_jac(x) -> np.ndarray:
            _update_cache(x)
            return cache["gradient"]  # type: ignore[return-value]

        def _objective_hess(x) -> csr_matrix:
            _update_cache(x)
            return cache["hessian"]  # type: ignore[return-value]

        constraints = []
        if self.continuity_size > 0:
            def _continuity_fun(x) -> np.ndarray:
                _update_cache(x)
                evaluation = cache["evaluation"]
                assert isinstance(evaluation, ShootingEvaluation)
                return evaluation.continuity_residual

            def _continuity_jac(x) -> csr_matrix:
                _update_cache(x)
                evaluation = cache["evaluation"]
                assert isinstance(evaluation, ShootingEvaluation)
                return evaluation.continuity_jacobian

            constraints.append(
                NonlinearConstraint(
                    _continuity_fun,
                    np.zeros(self.continuity_size, dtype=float),
                    np.zeros(self.continuity_size, dtype=float),
                    jac=_continuity_jac,
                )
            )

        if spec.terminal_constraint is not None:
            terminal_spec = spec.terminal_constraint

            lower = (
                np.full_like(terminal_spec.upper, -np.inf)
                if terminal_spec.lower is None
                else terminal_spec.lower
            )
            upper = (
                np.full_like(terminal_spec.lower, np.inf)
                if terminal_spec.upper is None
                else terminal_spec.upper
            )

            def _terminal_fun(x) -> np.ndarray:
                _update_cache(x)
                terminal = cache["terminal_outputs"]
                assert terminal is not None
                return terminal[0]

            def _terminal_jac(x) -> csr_matrix:
                _update_cache(x)
                terminal = cache["terminal_outputs"]
                assert terminal is not None
                return terminal[1]

            constraints.append(
                NonlinearConstraint(
                    _terminal_fun,
                    lower,
                    upper,
                    jac=_terminal_jac,
                )
            )

        solve_options = {"verbose": 0, "gtol": 1.0e-9, "xtol": 1.0e-9, "maxiter": 200}
        if spec.options is not None:
            solve_options.update(spec.options)

        minimize_kwargs = {
            "fun": _objective,
            "x0": x0,
            "jac": _objective_jac,
            "method": "trust-constr",
            "bounds": bounds,
            "constraints": constraints,
            "options": solve_options,
        }
        if use_explicit_objective_hessian:
            minimize_kwargs["hess"] = _objective_hess

        scipy_result = minimize(**minimize_kwargs)

        if spec.measurement_objective is not None:
            final_measurement_evaluation = self.evaluate_measurements(
                scipy_result.x,
                spec.measurement_objective.measurements,
            )
            final_evaluation = final_measurement_evaluation.shooting_evaluation
            final_objective, _, _ = self.measurement_objective(
                scipy_result.x,
                spec.measurement_objective.measurements,
                evaluation=final_measurement_evaluation,
                weight=spec.measurement_objective.weight,
            )
        else:
            final_evaluation = self.evaluate(scipy_result.x)
            final_objective, _ = self.minimum_propellant_objective(
                scipy_result.x,
                evaluation=final_evaluation,
            )
        terminal_outputs = None
        terminal_violation = None
        terminal_residual = None
        if spec.terminal_constraint is not None:
            terminal_outputs, _ = self.terminal_outputs(
                scipy_result.x,
                output_indices=spec.terminal_constraint.output_indices,
                evaluation=final_evaluation,
            )
            lower = (
                np.full_like(terminal_outputs, -np.inf)
                if spec.terminal_constraint.lower is None
                else spec.terminal_constraint.lower
            )
            upper = (
                np.full_like(terminal_outputs, np.inf)
                if spec.terminal_constraint.upper is None
                else spec.terminal_constraint.upper
            )
            terminal_violation = np.maximum(lower - terminal_outputs, 0.0) + np.maximum(
                terminal_outputs - upper, 0.0
            )
            if spec.terminal_constraint.is_equality:
                assert spec.terminal_constraint.lower is not None
                terminal_residual = terminal_outputs - spec.terminal_constraint.lower

        if spec.decision_tracking_penalty is not None:
            tracking_value, _, _ = self.decision_tracking_objective(
                scipy_result.x,
                spec.decision_tracking_penalty,
            )
            final_objective += tracking_value

        if spec.smoothness_penalty is not None:
            smooth_value, _, _ = self.control_smoothness_objective(
                scipy_result.x,
                spec.smoothness_penalty,
            )
            final_objective += smooth_value

        return ShootingOptimizationResult(
            x=np.array(scipy_result.x, dtype=float, copy=True),
            objective=float(final_objective),
            evaluation=final_evaluation,
            continuity_residual=final_evaluation.continuity_residual.copy(),
            terminal_outputs=None
            if terminal_outputs is None
            else np.array(terminal_outputs, dtype=float, copy=True),
            terminal_violation=None
            if terminal_violation is None
            else np.array(terminal_violation, dtype=float, copy=True),
            terminal_residual=None
            if terminal_residual is None
            else np.array(terminal_residual, dtype=float, copy=True),
            scipy_result=scipy_result,
        )

    def solve_minimum_propellant(
        self,
        target_state=None,
        output_indices: list[int] | np.ndarray | None = None,
        decision_vector0=None,
        bounds: Bounds | None = None,
        options: dict | None = None,
        terminal_lower=None,
        terminal_upper=None,
        smoothness_penalty: SmoothnessPenaltySpec | dict[str, float] | None = None,
    ) -> ShootingOptimizationResult:
        """Convenience wrapper for the current minimum-propellant prototype solve."""
        if target_state is not None and (
            terminal_lower is not None or terminal_upper is not None
        ):
            raise ValueError(
                "Use either target_state or terminal_lower/terminal_upper, not both."
            )

        terminal_constraint = None
        if target_state is not None:
            terminal_constraint = TerminalConstraintSpec.equality(
                target_state, output_indices=output_indices
            )
        elif terminal_lower is not None or terminal_upper is not None:
            terminal_constraint = TerminalConstraintSpec(
                lower=terminal_lower,
                upper=terminal_upper,
                output_indices=output_indices,
            )

        if smoothness_penalty is not None and not isinstance(
            smoothness_penalty, SmoothnessPenaltySpec
        ):
            smoothness_penalty = SmoothnessPenaltySpec(dict(smoothness_penalty))

        spec = ShootingSolveSpec(
            bounds=bounds,
            terminal_constraint=terminal_constraint,
            smoothness_penalty=smoothness_penalty,
            options=options,
        )
        return self.solve(spec, decision_vector0=decision_vector0)

    def solve_measurement_fit(
        self,
        measurements: list[SampledMeasurement] | tuple[SampledMeasurement, ...],
        decision_vector0=None,
        bounds: Bounds | None = None,
        options: dict | None = None,
        measurement_weight: float = 1.0,
        measurement_hessian_mode: Literal["quasi-newton", "gauss-newton"] = "quasi-newton",
        terminal_constraint: TerminalConstraintSpec | None = None,
        smoothness_penalty: SmoothnessPenaltySpec | dict[str, float] | None = None,
        decision_tracking_penalty: DecisionTrackingPenaltySpec | list[DecisionTrackingTerm] | None = None,
    ) -> ShootingOptimizationResult:
        """Convenience wrapper for measurement-driven shooting estimation."""
        if smoothness_penalty is not None and not isinstance(
            smoothness_penalty, SmoothnessPenaltySpec
        ):
            smoothness_penalty = SmoothnessPenaltySpec(dict(smoothness_penalty))

        if decision_tracking_penalty is not None and not isinstance(
            decision_tracking_penalty, DecisionTrackingPenaltySpec
        ):
            decision_tracking_penalty = DecisionTrackingPenaltySpec(
                list(decision_tracking_penalty)
            )

        spec = ShootingSolveSpec(
            bounds=bounds,
            measurement_objective=MeasurementObjectiveSpec(
                measurements=measurements,
                weight=measurement_weight,
                hessian_mode=measurement_hessian_mode,
            ),
            decision_tracking_penalty=decision_tracking_penalty,
            terminal_constraint=terminal_constraint,
            smoothness_penalty=smoothness_penalty,
            options=options,
        )
        return self.solve(spec, decision_vector0=decision_vector0)
