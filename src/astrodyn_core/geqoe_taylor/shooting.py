"""Multiple-shooting helpers for the GEqOE Taylor thrust backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize
from scipy.sparse import csr_matrix, lil_matrix

from astrodyn_core.geqoe_taylor.integrator import (
    build_thrust_sensitivity_integrator,
    extract_variational_matrices,
    parameter_names_from_map,
)
from astrodyn_core.geqoe_taylor.perturbations.base import PerturbationModel

STATE_NAMES = ("nu", "p1", "p2", "K", "q1", "q2", "m")
STATE_DIM = len(STATE_NAMES)


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
class ShootingSolveSpec:
    """Configuration for the SciPy-based multiple-shooting prototype solve."""

    bounds: Bounds | None = None
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

    def evaluate(self, decision_vector) -> ShootingEvaluation:
        """Propagate every arc and assemble the continuity residual/Jacobian."""
        x = self._validate_decision_vector(decision_vector)
        arc_results: list[ArcPropagationResult] = []

        for layout, (ta, par_map), template in zip(
            self._layouts, self._integrators, self._templates, strict=True
        ):
            ta.time = layout.start_time_s
            ta.state[:] = template
            ta.pars[:] = layout.parameter_defaults

            arc_state = x[layout.state_slice]
            ta.state[:STATE_DIM] = arc_state

            if layout.parameter_names:
                arc_parameters = x[layout.parameter_slice]
                ta.pars[list(layout.parameter_indices)] = arc_parameters

            ta.propagate_until(layout.start_time_s + layout.duration_s)

            final_state, phi_x, phi_p_all, all_param_names = extract_variational_matrices(
                ta.state,
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

        return ShootingEvaluation(
            arc_results=tuple(arc_results),
            continuity_residual=continuity_residual,
            continuity_jacobian=continuity_jacobian.tocsr(),
        )

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

        cache: dict[str, object] = {
            "x": None,
            "evaluation": None,
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

            evaluation = self.evaluate(x_vec)
            objective, gradient = self.minimum_propellant_objective(
                x_vec, evaluation=evaluation
            )
            hessian = self.minimum_propellant_hessian()

            if spec.smoothness_penalty is not None:
                smooth_value, smooth_gradient, smooth_hessian = (
                    self.control_smoothness_objective(
                        x_vec, spec.smoothness_penalty
                    )
                )
                objective += smooth_value
                gradient += smooth_gradient
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

        scipy_result = minimize(
            _objective,
            x0,
            jac=_objective_jac,
            hess=_objective_hess,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            options=solve_options,
        )

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
