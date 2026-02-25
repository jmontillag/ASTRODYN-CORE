"""STM-based covariance propagation implementation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from astrodyn_core.states.models import OrbitStateRecord, OutputEpochSpec, StateSeries
from astrodyn_core.states.orekit_dates import from_orekit_date, to_orekit_date
from astrodyn_core.states.orekit_resolvers import resolve_frame
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty.records import state_to_orbit_record
from astrodyn_core.uncertainty.spec import UncertaintySpec
from astrodyn_core.uncertainty.transforms import (
    change_covariance_type,
    configure_cartesian_propagation_basis,
    frame_jacobian,
    orbit_jacobian,
    orekit_orbit_type,
    orekit_position_angle,
    transform_covariance_with_jacobian,
)
from astrodyn_core.uncertainty.matrix_io import realmatrix_to_numpy


class STMCovariancePropagator:
    """Propagate state and covariance using an Orekit State Transition Matrix.

    Args:
        propagator: Orekit propagator supporting
            ``setupMatricesComputation`` (for example numerical or DSST).
        initial_covariance: Optional initial covariance (6x6 or 7x7). If
            omitted, only STM extraction via ``propagate_with_stm`` is available.
        spec: Uncertainty propagation configuration. Defaults to STM settings.
        frame: Output frame name for record serialization.
        mu_m3_s2: Gravitational parameter for state record serialization.
        default_mass_kg: Fallback mass when Orekit states do not expose mass.
    """

    def __init__(
        self,
        propagator: Any,
        initial_covariance: np.ndarray | Sequence[Sequence[float]] | None = None,
        spec: UncertaintySpec | None = None,
        *,
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        default_mass_kg: float = 1000.0,
    ) -> None:
        if spec is None:
            spec = UncertaintySpec()
        if spec.method != "stm":
            raise ValueError(
                f"STMCovariancePropagator requires spec.method='stm', got {spec.method!r}."
            )
        self._propagator = propagator
        self._spec = spec
        self._frame = str(frame).strip().upper()
        self._output_frame_orekit = resolve_frame(self._frame)
        self._mu = mu_m3_s2
        self._default_mass = default_mass_kg
        configure_cartesian_propagation_basis(self._propagator)
        self._initial_state_orekit: Any = self._propagator.getInitialState()

        if initial_covariance is not None:
            cov_arr = np.asarray(initial_covariance, dtype=np.float64)
            n = spec.state_dimension
            if cov_arr.shape != (n, n):
                raise ValueError(
                    f"initial_covariance must have shape ({n}, {n}) for spec.include_mass="
                    f"{spec.include_mass}, got {cov_arr.shape}."
                )
            if spec.orbit_type != "CARTESIAN":
                from_ot = orekit_orbit_type(spec.orbit_type)
                from_pa = orekit_position_angle(spec.position_angle)
                cov6 = change_covariance_type(
                    cov_arr[:6, :6],
                    self._initial_state_orekit.getOrbit(),
                    self._initial_state_orekit.getDate(),
                    self._initial_state_orekit.getFrame(),
                    from_ot,
                    from_pa,
                    orekit_orbit_type("CARTESIAN"),
                    orekit_position_angle("TRUE"),
                )
                if n == 6:
                    cov_arr = cov6
                else:
                    jac = orbit_jacobian(
                        self._initial_state_orekit.getOrbit(),
                        from_orbit_type=from_ot,
                        from_pa_type=from_pa,
                        to_orbit_type=orekit_orbit_type("CARTESIAN"),
                        to_pa_type=orekit_position_angle("TRUE"),
                    )
                    cov_arr = transform_covariance_with_jacobian(cov_arr, jac)
                    cov_arr[:6, :6] = cov6
            self._initial_cov: np.ndarray | None = cov_arr
        else:
            self._initial_cov = None

        self._harvester = self._setup_stm()

    def _setup_stm(self) -> Any:
        """Configure Orekit matrices computation and return the STM harvester.

        Returns:
            Orekit matrices harvester bound to the propagator initial state.
        """
        from org.hipparchus.linear import MatrixUtils

        n = self._spec.state_dimension
        identity = MatrixUtils.createRealIdentityMatrix(n)

        harvester = self._propagator.setupMatricesComputation(
            self._spec.stm_name,
            identity,
            None,
        )
        harvester.setReferenceState(self._initial_state_orekit)
        return harvester

    def propagate_with_stm(
        self,
        target_date_or_epoch: Any,
    ) -> tuple[Any, np.ndarray]:
        """Propagate to a target epoch and return the propagated state and STM.

        Args:
            target_date_or_epoch: Orekit date or UTC epoch string.

        Returns:
            Tuple ``(state, phi)`` where ``phi`` is the propagated STM truncated
            to ``spec.state_dimension``.

        Raises:
            RuntimeError: If the propagator/harvester does not return an STM.
        """
        if isinstance(target_date_or_epoch, str):
            target_date = to_orekit_date(target_date_or_epoch)
        else:
            target_date = target_date_or_epoch

        state = self._propagator.propagate(target_date)
        stm_matrix = self._harvester.getStateTransitionMatrix(state)

        if stm_matrix is None:
            raise RuntimeError(
                "STM harvester returned None. Ensure setupMatricesComputation was called "
                "and the propagator supports STM computation (NumericalPropagator or DSST)."
            )

        phi = realmatrix_to_numpy(stm_matrix)
        n = self._spec.state_dimension
        return state, phi[:n, :n]

    def propagate_with_covariance(
        self,
        target_date_or_epoch: Any,
    ) -> tuple[Any, CovarianceRecord]:
        """Propagate to a target epoch and return state plus covariance record.

        The internal covariance is propagated in Cartesian form via
        ``P = Phi P0 Phi^T``, then transformed to the requested frame/orbit type
        for output serialization.

        Args:
            target_date_or_epoch: Orekit date or UTC epoch string.

        Returns:
            Tuple ``(state, covariance_record)``.

        Raises:
            ValueError: If no initial covariance was configured.
            RuntimeError: If the propagator/harvester does not return an STM.
        """
        if self._initial_cov is None:
            raise ValueError(
                "propagate_with_covariance() requires an initial_covariance. "
                "Either pass one at construction time or use propagate_with_stm() "
                "for STM-only propagation."
            )

        if isinstance(target_date_or_epoch, str):
            target_date = to_orekit_date(target_date_or_epoch)
        else:
            target_date = target_date_or_epoch

        state = self._propagator.propagate(target_date)
        stm_matrix = self._harvester.getStateTransitionMatrix(state)

        if stm_matrix is None:
            raise RuntimeError(
                "STM harvester returned None. Ensure setupMatricesComputation was called "
                "and the propagator supports STM computation (NumericalPropagator or DSST)."
            )

        phi = realmatrix_to_numpy(stm_matrix)
        n = self._spec.state_dimension
        phi_n = phi[:n, :n]

        cov_cart = phi_n @ self._initial_cov @ phi_n.T
        cov_cart = 0.5 * (cov_cart + cov_cart.T)

        from org.orekit.orbits import CartesianOrbit

        state_orbit = state.getOrbit()
        output_orbit = state_orbit
        if state_orbit.getFrame() != self._output_frame_orekit:
            pv_out = state.getPVCoordinates(self._output_frame_orekit)
            output_orbit = CartesianOrbit(
                pv_out,
                self._output_frame_orekit,
                state.getDate(),
                float(state_orbit.getMu()),
            )
            jac_frame = frame_jacobian(
                state_orbit.getFrame(),
                self._output_frame_orekit,
                target_date,
            )
            cov_cart = transform_covariance_with_jacobian(cov_cart, jac_frame)
            cov_cart = 0.5 * (cov_cart + cov_cart.T)

        if self._spec.orbit_type != "CARTESIAN":
            to_ot = orekit_orbit_type(self._spec.orbit_type)
            to_pa = orekit_position_angle(self._spec.position_angle)
            cov_6x6 = change_covariance_type(
                cov_cart[:6, :6],
                output_orbit,
                target_date,
                output_orbit.getFrame(),
                orekit_orbit_type("CARTESIAN"),
                orekit_position_angle("TRUE"),
                to_ot,
                to_pa,
            )
            if n == 7:
                jac_orbit = orbit_jacobian(
                    output_orbit,
                    from_orbit_type=orekit_orbit_type("CARTESIAN"),
                    from_pa_type=orekit_position_angle("TRUE"),
                    to_orbit_type=to_ot,
                    to_pa_type=to_pa,
                )
                cov_propagated = transform_covariance_with_jacobian(cov_cart, jac_orbit)
                cov_propagated[:6, :6] = cov_6x6
            else:
                cov_propagated = cov_6x6
        else:
            cov_propagated = cov_cart

        epoch_str = from_orekit_date(target_date)
        cov_record = CovarianceRecord.from_numpy(
            epoch=epoch_str,
            matrix=cov_propagated,
            frame=self._frame,
            orbit_type=self._spec.orbit_type,
            include_mass=self._spec.include_mass,
        )
        return state, cov_record

    def propagate_series(
        self,
        epoch_spec: OutputEpochSpec,
        *,
        series_name: str = "trajectory",
        covariance_name: str = "covariance",
    ) -> tuple[StateSeries, CovarianceSeries]:
        """Propagate a state/covariance series over an output epoch grid.

        Args:
            epoch_spec: Output epoch specification.
            series_name: Name for the returned state series.
            covariance_name: Name for the returned covariance series.

        Returns:
            Tuple ``(StateSeries, CovarianceSeries)`` with aligned epochs.

        Raises:
            TypeError: If ``epoch_spec`` is not an ``OutputEpochSpec``.
            ValueError: If the epoch specification produces no epochs.
        """
        if not isinstance(epoch_spec, OutputEpochSpec):
            raise TypeError("epoch_spec must be an OutputEpochSpec instance.")

        epochs = epoch_spec.epochs()
        if not epochs:
            raise ValueError("epoch_spec produced no epochs.")

        state_records: list[OrbitStateRecord] = []
        cov_records: list[CovarianceRecord] = []

        for epoch_str in epochs:
            state, cov_record = self.propagate_with_covariance(epoch_str)
            orbit_record = state_to_orbit_record(
                state,
                frame=self._frame,
                orbit_type=self._spec.orbit_type,
                mu_m3_s2=self._mu,
                default_mass_kg=self._default_mass,
                output_frame=self._output_frame_orekit,
            )
            state_records.append(orbit_record)
            cov_records.append(cov_record)

        state_series = StateSeries(
            name=series_name,
            states=tuple(state_records),
            interpolation={"method": "stm_covariance"},
        )
        cov_series = CovarianceSeries(
            name=covariance_name,
            records=tuple(cov_records),
            method=self._spec.method,
        )
        return state_series, cov_series
