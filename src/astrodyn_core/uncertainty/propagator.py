"""Covariance propagation via State Transition Matrix (STM) and future Unscented Transform."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from astrodyn_core.states.models import OrbitStateRecord, OutputEpochSpec, StateSeries
from astrodyn_core.states.orekit import from_orekit_date, resolve_frame, to_orekit_date
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty.spec import UncertaintySpec


# ---------------------------------------------------------------------------
# Helpers for RealMatrix ↔ numpy conversion
# ---------------------------------------------------------------------------

def _realmatrix_to_numpy(mat: Any) -> np.ndarray:
    """Convert an Orekit ``RealMatrix`` to a numpy array."""
    from orekit import JArray_double
    #rows = int(mat.getRowDimension())
    #cols = int(mat.getColumnDimension())
    stm_orbit_phys = np.array([JArray_double.cast_(r) for r in mat.getData()])
    return stm_orbit_phys


def _new_java_double_2d(rows: int, cols: int) -> Any:
    """Allocate a Java ``double[][]`` for Orekit Jacobian APIs."""
    from orekit import JArray_double, JArray_object

    out = JArray_object(rows)
    for i in range(rows):
        out[i] = JArray_double(cols)
    return out


def _java_double_2d_to_numpy(mat: Any, rows: int) -> np.ndarray:
    """Convert a Java ``double[][]`` into a ``(rows, n)`` numpy array."""
    from orekit import JArray_double

    return np.array([JArray_double.cast_(mat[i]) for i in range(rows)], dtype=np.float64)


def _numpy_to_realmatrix(arr: np.ndarray) -> Any:
    """Convert a 2-D numpy array to an Orekit ``RealMatrix``."""
    from org.hipparchus.linear import MatrixUtils

    rows, cols = arr.shape
    mat = MatrixUtils.createRealMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            mat.setEntry(i, j, float(arr[i, j]))
    return mat


def _change_covariance_type(
    cov_6x6: np.ndarray,
    orbit: Any,
    epoch: Any,
    frame: Any,
    from_orbit_type: Any,
    from_pa_type: Any,
    to_orbit_type: Any,
    to_pa_type: Any,
) -> np.ndarray:
    """Use ``StateCovariance.changeCovarianceType`` to re-parametrise a 6×6 covariance.

    Delegates all Jacobian computation to Orekit, which internally calls
    ``getJacobianWrtCartesian`` / ``getJacobianWrtParameters`` as needed.
    """
    from org.orekit.propagation import StateCovariance

    sc = StateCovariance(
        _numpy_to_realmatrix(cov_6x6),
        epoch,
        frame,
        from_orbit_type,
        from_pa_type,
    )
    sc_new = sc.changeCovarianceType(orbit, to_orbit_type, to_pa_type)
    return _realmatrix_to_numpy(sc_new.getMatrix())


def _orekit_orbit_type(name: str) -> Any:
    from org.orekit.orbits import OrbitType

    mapping = {
        "CARTESIAN": OrbitType.CARTESIAN,
        "KEPLERIAN": OrbitType.KEPLERIAN,
        "EQUINOCTIAL": OrbitType.EQUINOCTIAL,
    }
    return mapping[name]


def _orekit_position_angle(name: str) -> Any:
    from org.orekit.orbits import PositionAngleType

    mapping = {
        "MEAN": PositionAngleType.MEAN,
        "TRUE": PositionAngleType.TRUE,
        "ECCENTRIC": PositionAngleType.ECCENTRIC,
    }
    return mapping[name]


def _configure_cartesian_propagation_basis(propagator: Any) -> None:
    """Force integrated orbit parameters to Cartesian for STM consistency."""
    from org.orekit.orbits import OrbitType, PositionAngleType

    if not hasattr(propagator, "setOrbitType"):
        raise TypeError(
            "STM covariance propagation requires a propagator supporting setOrbitType()."
        )
    propagator.setOrbitType(OrbitType.CARTESIAN)
    if hasattr(propagator, "setPositionAngleType"):
        propagator.setPositionAngleType(PositionAngleType.TRUE)


def _orbit_jacobian(
    orbit: Any,
    *,
    from_orbit_type: Any,
    from_pa_type: Any,
    to_orbit_type: Any,
    to_pa_type: Any,
) -> np.ndarray:
    """Return Jacobian ``J = d(to_params) / d(from_params)`` for 6 orbital params."""
    cart_type = _orekit_orbit_type("CARTESIAN")
    if (
        from_orbit_type == to_orbit_type
        and (from_orbit_type == cart_type or from_pa_type == to_pa_type)
    ):
        return np.eye(6, dtype=np.float64)

    # Generic chain through Cartesian:
    #   d(to)/d(from) = d(to)/d(cart) @ d(cart)/d(from)
    jac_to_wrt_cart = np.eye(6, dtype=np.float64)
    if to_orbit_type != cart_type:
        orbit_to = to_orbit_type.convertType(orbit)
        j = _new_java_double_2d(6, 6)
        orbit_to.getJacobianWrtCartesian(to_pa_type, j)
        jac_to_wrt_cart = _java_double_2d_to_numpy(j, 6)

    jac_cart_wrt_from = np.eye(6, dtype=np.float64)
    if from_orbit_type != cart_type:
        orbit_from = from_orbit_type.convertType(orbit)
        j = _new_java_double_2d(6, 6)
        orbit_from.getJacobianWrtParameters(from_pa_type, j)
        jac_cart_wrt_from = _java_double_2d_to_numpy(j, 6)

    return jac_to_wrt_cart @ jac_cart_wrt_from


def _frame_jacobian(from_frame: Any, to_frame: Any, epoch: Any) -> np.ndarray:
    """Return Cartesian PV Jacobian ``d(PV_to) / d(PV_from)`` for a frame transform."""
    if from_frame == to_frame:
        return np.eye(6, dtype=np.float64)

    from org.orekit.utils import CartesianDerivativesFilter

    transform = from_frame.getTransformTo(to_frame, epoch)
    j = _new_java_double_2d(6, 6)
    transform.getJacobian(CartesianDerivativesFilter.USE_PV, j)
    return _java_double_2d_to_numpy(j, 6)


def _transform_covariance_with_jacobian(cov: np.ndarray, jac6: np.ndarray) -> np.ndarray:
    """Apply a 6D Jacobian to 6x6 or 7x7 covariance (mass is unchanged for 7x7)."""
    n = cov.shape[0]
    if n == 6:
        out6 = jac6 @ cov @ jac6.T
        return 0.5 * (out6 + out6.T)
    if n != 7:
        raise ValueError(f"Unsupported covariance shape for Jacobian transform: {cov.shape}.")

    out = np.zeros((7, 7), dtype=np.float64)
    out[:6, :6] = jac6 @ cov[:6, :6] @ jac6.T
    out[:6, 6] = jac6 @ cov[:6, 6]
    out[6, :6] = cov[6, :6] @ jac6.T
    out[6, 6] = float(cov[6, 6])
    return 0.5 * (out + out.T)


def _numpy_to_nested_tuple(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in arr)


def _state_to_orbit_record(
    state: Any,
    *,
    frame: str,
    orbit_type: str,
    mu_m3_s2: float | str,
    default_mass_kg: float,
    output_frame: Any | None = None,
) -> OrbitStateRecord:
    """Convert an Orekit SpacecraftState to an OrbitStateRecord."""
    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    epoch = from_orekit_date(state.getDate())
    orbit = state.getOrbit()
    if output_frame is not None and orbit.getFrame() != output_frame:
        pv_out = state.getPVCoordinates(output_frame)
        orbit = CartesianOrbit(pv_out, output_frame, state.getDate(), float(orbit.getMu()))
    mu = float(orbit.getMu())

    ot = orbit_type.upper()
    if ot == "CARTESIAN":
        pv = state.getPVCoordinates(orbit.getFrame())
        pos = pv.getPosition()
        vel = pv.getVelocity()
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame,
            representation="cartesian",
            position_m=(float(pos.getX()), float(pos.getY()), float(pos.getZ())),
            velocity_mps=(float(vel.getX()), float(vel.getY()), float(vel.getZ())),
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    if ot == "KEPLERIAN":
        kep = KeplerianOrbit(orbit)
        return OrbitStateRecord(
            epoch=epoch,
            frame=frame,
            representation="keplerian",
            elements={
                "a_m": float(kep.getA()),
                "e": float(kep.getE()),
                "i_deg": math.degrees(float(kep.getI())),
                "argp_deg": math.degrees(float(kep.getPerigeeArgument())),
                "raan_deg": math.degrees(float(kep.getRightAscensionOfAscendingNode())),
                "anomaly_deg": math.degrees(float(kep.getMeanAnomaly())),
                "anomaly_type": "MEAN",
            },
            mu_m3_s2=mu_m3_s2,
            mass_kg=mass,
        )

    equi = EquinoctialOrbit(orbit)
    return OrbitStateRecord(
        epoch=epoch,
        frame=frame,
        representation="equinoctial",
        elements={
            "a_m": float(equi.getA()),
            "ex": float(equi.getEquinoctialEx()),
            "ey": float(equi.getEquinoctialEy()),
            "hx": float(equi.getHx()),
            "hy": float(equi.getHy()),
            "l_deg": math.degrees(float(equi.getLM())),
            "anomaly_type": "MEAN",
        },
        mu_m3_s2=mu_m3_s2,
        mass_kg=mass,
    )


# ---------------------------------------------------------------------------
# STM-based covariance propagator
# ---------------------------------------------------------------------------

class STMCovariancePropagator:
    """Propagates a state and, optionally, its covariance matrix via the STM.

    Wraps Orekit's ``setupMatricesComputation`` / ``MatricesHarvester`` API.
    Requires a :class:`~org.orekit.propagation.numerical.NumericalPropagator`
    (or DSST propagator) **before** the first ``propagate()`` call.

    Two independent use-modes
    -------------------------
    **STM-only** (``initial_covariance=None``, the default):
        Call :meth:`propagate_with_stm` to obtain the raw Φ(t, t₀) matrix
        alongside the propagated state.  No initial covariance is required.

    **Covariance propagation** (``initial_covariance`` provided):
        Call :meth:`propagate_with_covariance` or :meth:`propagate_series` to
        obtain the linearly-propagated covariance

        .. math::

            P(t) = \\Phi(t, t_0)\\, P_0\\, \\Phi(t, t_0)^\\top

        as :class:`~astrodyn_core.uncertainty.models.CovarianceRecord` objects
        that can be serialised to YAML/HDF5.

    Parameters
    ----------
    propagator:
        An Orekit numerical propagator instance (not a builder).
    initial_covariance:
        Initial covariance matrix, shape ``(n, n)`` where ``n = spec.state_dimension``
        (6 by default, 7 when ``spec.include_mass=True``).
        Pass ``None`` (the default) for STM-only mode.
    spec:
        :class:`~astrodyn_core.uncertainty.spec.UncertaintySpec` controlling
        the STM name, orbit type and position-angle convention.
    frame:
        Frame name used for output state records (default ``"GCRF"``).
    mu_m3_s2:
        Gravitational parameter for output state records.
    default_mass_kg:
        Fallback spacecraft mass if not stored in the propagated state.
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
        _configure_cartesian_propagation_basis(self._propagator)
        # Snapshot initial state for Jacobian evaluation at t₀
        self._initial_state_orekit: Any = self._propagator.getInitialState()

        if initial_covariance is not None:
            cov_arr = np.asarray(initial_covariance, dtype=np.float64)
            n = spec.state_dimension
            if cov_arr.shape != (n, n):
                raise ValueError(
                    f"initial_covariance must have shape ({n}, {n}) for spec.include_mass="
                    f"{spec.include_mass}, got {cov_arr.shape}."
                )
            # Non-Cartesian input: convert to Cartesian for internal storage so
            # the STM formula P(t) = Φ · P₀ · Φᵀ is applied in a consistent space.
            # StateCovariance.changeCovarianceType handles all Jacobian computation.
            if spec.orbit_type != "CARTESIAN":
                from_ot = _orekit_orbit_type(spec.orbit_type)
                from_pa = _orekit_position_angle(spec.position_angle)
                cov6 = _change_covariance_type(
                    cov_arr[:6, :6],
                    self._initial_state_orekit.getOrbit(),
                    self._initial_state_orekit.getDate(),
                    self._initial_state_orekit.getFrame(),
                    from_ot,
                    from_pa,
                    _orekit_orbit_type("CARTESIAN"),
                    _orekit_position_angle("TRUE"),
                )
                if n == 6:
                    cov_arr = cov6
                else:
                    jac = _orbit_jacobian(
                        self._initial_state_orekit.getOrbit(),
                        from_orbit_type=from_ot,
                        from_pa_type=from_pa,
                        to_orbit_type=_orekit_orbit_type("CARTESIAN"),
                        to_pa_type=_orekit_position_angle("TRUE"),
                    )
                    cov_arr = _transform_covariance_with_jacobian(cov_arr, jac)
                    cov_arr[:6, :6] = cov6
            self._initial_cov: np.ndarray | None = cov_arr
        else:
            self._initial_cov = None

        self._harvester = self._setup_stm()

    def _setup_stm(self) -> Any:
        """Call ``setupMatricesComputation`` on the propagator and return the harvester.

        The STM is always computed internally in Cartesian coordinates by Orekit's
        ``NumericalPropagator``.  Any orbit-type conversion is handled in
        :meth:`propagate_with_covariance` via explicit Jacobian transformations.
        """
        from org.hipparchus.linear import MatrixUtils

        n = self._spec.state_dimension
        identity = MatrixUtils.createRealIdentityMatrix(n)

        harvester = self._propagator.setupMatricesComputation(
            self._spec.stm_name,
            identity,
            None,  # no parameter Jacobians
        )
        harvester.setReferenceState(self._initial_state_orekit)
        return harvester

    def propagate_with_stm(
        self,
        target_date_or_epoch: Any,
    ) -> tuple[Any, np.ndarray]:
        """Propagate to the target epoch and return the state + raw Cartesian STM.

        This method does **not** require an initial covariance and does not
        produce any :class:`~astrodyn_core.uncertainty.models.CovarianceRecord`
        objects — it simply exposes the underlying Φ(t, t₀) matrix for callers
        that want to work with it directly.

        .. note::

            The returned STM is **always in Cartesian coordinates**, regardless
            of ``spec.orbit_type``.  Orekit's ``NumericalPropagator`` computes
            the STM internally in Cartesian space.  To obtain the STM in a
            non-Cartesian parametrisation, apply the Jacobian transformation
            yourself::

                J_t  = ∂(orbit_params)/∂(Cartesian) at the target state
                J₀   = ∂(orbit_params)/∂(Cartesian) at the initial state
                Φ_orbit(t, t₀) = J_t · Φ_cart · J₀⁻¹

        Parameters
        ----------
        target_date_or_epoch:
            Either an Orekit ``AbsoluteDate`` or an ISO-8601 UTC string.

        Returns
        -------
        tuple[SpacecraftState, np.ndarray]
            The propagated Orekit ``SpacecraftState`` and the Cartesian STM Φ
            as a numpy array of shape ``(n, n)`` where
            ``n = spec.state_dimension``.
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

        phi = _realmatrix_to_numpy(stm_matrix)
        n = self._spec.state_dimension
        return state, phi[:n, :n]

    def propagate_with_covariance(
        self,
        target_date_or_epoch: Any,
    ) -> tuple[Any, CovarianceRecord]:
        """Propagate to the target epoch and return the state + propagated covariance.

        Requires that an ``initial_covariance`` was provided at construction time.

        Parameters
        ----------
        target_date_or_epoch:
            Either an Orekit ``AbsoluteDate`` or an ISO-8601 UTC string.

        Returns
        -------
        tuple[SpacecraftState, CovarianceRecord]

        Raises
        ------
        ValueError
            If no ``initial_covariance`` was provided at construction time.
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

        phi = _realmatrix_to_numpy(stm_matrix)
        n = self._spec.state_dimension
        phi_n = phi[:n, :n]  # guard against larger harvested matrices

        # Propagate in Cartesian (self._initial_cov is always stored in Cartesian)
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
            jac_frame = _frame_jacobian(
                state_orbit.getFrame(),
                self._output_frame_orekit,
                target_date,
            )
            cov_cart = _transform_covariance_with_jacobian(cov_cart, jac_frame)
            cov_cart = 0.5 * (cov_cart + cov_cart.T)

        # If non-Cartesian output requested, convert the propagated Cartesian
        # covariance to the target parametrisation via StateCovariance.
        if self._spec.orbit_type != "CARTESIAN":
            to_ot = _orekit_orbit_type(self._spec.orbit_type)
            to_pa = _orekit_position_angle(self._spec.position_angle)
            cov_6x6 = _change_covariance_type(
                cov_cart[:6, :6],
                output_orbit,
                target_date,
                output_orbit.getFrame(),
                _orekit_orbit_type("CARTESIAN"),
                _orekit_position_angle("TRUE"),
                to_ot,
                to_pa,
            )
            if n == 7:
                jac_orbit = _orbit_jacobian(
                    output_orbit,
                    from_orbit_type=_orekit_orbit_type("CARTESIAN"),
                    from_pa_type=_orekit_position_angle("TRUE"),
                    to_orbit_type=to_ot,
                    to_pa_type=to_pa,
                )
                cov_propagated = _transform_covariance_with_jacobian(cov_cart, jac_orbit)
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
        """Propagate over a set of epochs, collecting states + covariances.

        Parameters
        ----------
        epoch_spec:
            Defines the output epochs.
        series_name:
            Name for the returned :class:`~astrodyn_core.states.models.StateSeries`.
        covariance_name:
            Name for the returned :class:`~astrodyn_core.uncertainty.models.CovarianceSeries`.

        Returns
        -------
        tuple[StateSeries, CovarianceSeries]
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
            orbit_record = _state_to_orbit_record(
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


# ---------------------------------------------------------------------------
# Unscented Transform propagator (stub for future implementation)
# ---------------------------------------------------------------------------

class UnscentedCovariancePropagator:
    """Unscented Transform covariance propagator (not yet implemented).

    Future implementation will:

    1. Generate 2n+1 sigma points from the initial covariance using a
       Merwe (or Julier) scaling strategy.
    2. Propagate each sigma point independently through the full nonlinear
       dynamics (separate propagator instances or a single propagator
       re-initialized per point).
    3. Reconstruct the propagated mean and covariance via weighted
       recombination.

    This method is more accurate than the linear STM approach for highly
    nonlinear dynamics (e.g., long arcs, high-eccentricity orbits).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "UnscentedCovariancePropagator is not yet implemented. "
            "Use UncertaintySpec(method='stm') for the current STM-based approach."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def setup_stm_propagator(
    propagator: Any,
    spec: UncertaintySpec | None = None,
    *,
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    default_mass_kg: float = 1000.0,
) -> STMCovariancePropagator:
    """Configure a propagator for STM-only extraction (no covariance required).

    This is the lightweight entry point when you only need the State Transition
    Matrix Φ(t, t₀) at one or more target epochs — no initial covariance matrix
    is needed and no :class:`~astrodyn_core.uncertainty.models.CovarianceRecord`
    objects are produced.

    Parameters
    ----------
    propagator:
        Orekit numerical propagator instance (must support
        ``setupMatricesComputation``).
    spec:
        :class:`~astrodyn_core.uncertainty.spec.UncertaintySpec` controlling
        the STM name and orbit/angle convention.  Defaults to a Cartesian STM
        with ``UncertaintySpec()``.
    frame:
        Frame name used for output state records (default ``"GCRF"``).
    mu_m3_s2:
        Gravitational parameter for output state records.
    default_mass_kg:
        Fallback spacecraft mass.

    Returns
    -------
    STMCovariancePropagator
        An instance in STM-only mode.  Call :meth:`~STMCovariancePropagator.propagate_with_stm`
        to obtain ``(SpacecraftState, Φ_ndarray)`` tuples.

    Example
    -------
    ::

        stm_prop = setup_stm_propagator(propagator)
        state, phi = stm_prop.propagate_with_stm("2026-02-19T01:00:00Z")
        # phi is a (6, 6) numpy array
    """
    return STMCovariancePropagator(
        propagator,
        None,  # STM-only: no initial covariance
        spec,
        frame=frame,
        mu_m3_s2=mu_m3_s2,
        default_mass_kg=default_mass_kg,
    )


def create_covariance_propagator(
    propagator: Any,
    initial_covariance: np.ndarray | Sequence[Sequence[float]],
    spec: UncertaintySpec,
    *,
    frame: str = "GCRF",
    mu_m3_s2: float | str = "WGS84",
    default_mass_kg: float = 1000.0,
) -> STMCovariancePropagator:
    """Factory that creates the appropriate covariance propagator from a spec.

    Parameters
    ----------
    propagator:
        Orekit propagator instance (must support ``setupMatricesComputation``
        for ``method="stm"``).
    initial_covariance:
        Initial covariance matrix, shape (6, 6) or (7, 7).
    spec:
        :class:`~astrodyn_core.uncertainty.spec.UncertaintySpec`.
    frame:
        Output frame name.
    mu_m3_s2:
        Gravitational parameter for output state records.
    default_mass_kg:
        Fallback mass if not stored in propagated state.

    Returns
    -------
    STMCovariancePropagator
        (or raises ``NotImplementedError`` for unsupported methods)
    """
    if spec.method == "stm":
        return STMCovariancePropagator(
            propagator,
            initial_covariance,
            spec,
            frame=frame,
            mu_m3_s2=mu_m3_s2,
            default_mass_kg=default_mass_kg,
        )
    if spec.method == "unscented":
        raise NotImplementedError(
            "Unscented Transform covariance propagation is planned but not yet implemented. "
            "Use UncertaintySpec(method='stm') for now."
        )
    raise ValueError(
        f"Unknown uncertainty method: {spec.method!r}. "
        "Supported: {'stm', 'unscented' (future)}."
    )
