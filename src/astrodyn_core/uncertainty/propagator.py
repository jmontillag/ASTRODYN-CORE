"""Covariance propagation via State Transition Matrix (STM) and future Unscented Transform."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from astrodyn_core.states.models import OrbitStateRecord, OutputEpochSpec, StateSeries
from astrodyn_core.states.orekit import from_orekit_date, to_orekit_date
from astrodyn_core.uncertainty.models import CovarianceRecord, CovarianceSeries
from astrodyn_core.uncertainty.spec import UncertaintySpec


# ---------------------------------------------------------------------------
# Helpers for RealMatrix ↔ numpy conversion
# ---------------------------------------------------------------------------

def _realmatrix_to_numpy(mat: Any) -> np.ndarray:
    """Convert an Orekit ``RealMatrix`` to a numpy array."""
    rows = int(mat.getRowDimension())
    cols = int(mat.getColumnDimension())
    return np.array(
        [[float(mat.getEntry(i, j)) for j in range(cols)] for i in range(rows)],
        dtype=np.float64,
    )


def _numpy_to_nested_tuple(arr: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in arr)


def _state_to_orbit_record(
    state: Any,
    *,
    frame: str,
    orbit_type: str,
    mu_m3_s2: float | str,
    default_mass_kg: float,
) -> OrbitStateRecord:
    """Convert an Orekit SpacecraftState to an OrbitStateRecord."""
    from org.orekit.orbits import CartesianOrbit, EquinoctialOrbit, KeplerianOrbit

    mass = float(state.getMass()) if hasattr(state, "getMass") else float(default_mass_kg)
    epoch = from_orekit_date(state.getDate())
    orbit = state.getOrbit()
    mu = float(orbit.getMu())

    ot = orbit_type.upper()
    if ot == "CARTESIAN":
        pv = state.getPVCoordinates()
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
    """Propagates a covariance matrix using Orekit's State Transition Matrix.

    Requires a :class:`~org.orekit.propagation.numerical.NumericalPropagator`
    (or DSST propagator). The propagator is configured via
    ``setupMatricesComputation`` before any propagation call.

    The propagated covariance at time ``t`` is computed as:

    .. math::

        P(t) = \\Phi(t, t_0) \\, P_0 \\, \\Phi(t, t_0)^\\top

    Parameters
    ----------
    propagator:
        An Orekit numerical propagator instance (not builder).
    initial_covariance:
        Initial covariance matrix as a numpy array of shape (6, 6) or (7, 7)
        if ``spec.include_mass`` is True.
    spec:
        :class:`~astrodyn_core.uncertainty.spec.UncertaintySpec` instance
        controlling STM name, orbit type, etc.
    frame:
        Frame name for output state records (default ``"GCRF"``).
    mu_m3_s2:
        Gravitational parameter for output state records.
    default_mass_kg:
        Fallback mass if not stored in propagated state.
    """

    def __init__(
        self,
        propagator: Any,
        initial_covariance: np.ndarray | Sequence[Sequence[float]],
        spec: UncertaintySpec,
        *,
        frame: str = "GCRF",
        mu_m3_s2: float | str = "WGS84",
        default_mass_kg: float = 1000.0,
    ) -> None:
        if spec.method != "stm":
            raise ValueError(
                f"STMCovariancePropagator requires spec.method='stm', got {spec.method!r}."
            )
        self._propagator = propagator
        self._spec = spec
        self._frame = frame
        self._mu = mu_m3_s2
        self._default_mass = default_mass_kg

        cov_arr = np.asarray(initial_covariance, dtype=np.float64)
        n = spec.state_dimension
        if cov_arr.shape != (n, n):
            raise ValueError(
                f"initial_covariance must have shape ({n}, {n}) for spec.include_mass="
                f"{spec.include_mass}, got {cov_arr.shape}."
            )
        self._initial_cov = cov_arr
        self._harvester = self._setup_stm()

    def _setup_stm(self) -> Any:
        """Call ``setupMatricesComputation`` on the propagator and return the harvester."""
        from org.hipparchus.linear import MatrixUtils
        from org.orekit.orbits import OrbitType, PositionAngleType

        n = self._spec.state_dimension
        identity = MatrixUtils.createRealIdentityMatrix(n)

        # setupMatricesComputation(stmName, initialStm, initialJacobianColumns)
        harvester = self._propagator.setupMatricesComputation(
            self._spec.stm_name,
            identity,
            None,  # no parameter Jacobians
        )

        # Tell the harvester which orbit/angle type to use for the STM columns
        orbit_type_map = {
            "CARTESIAN": OrbitType.CARTESIAN,
            "KEPLERIAN": OrbitType.KEPLERIAN,
            "EQUINOCTIAL": OrbitType.EQUINOCTIAL,
        }
        pa_map = {
            "MEAN": PositionAngleType.MEAN,
            "TRUE": PositionAngleType.TRUE,
            "ECCENTRIC": PositionAngleType.ECCENTRIC,
        }
        orbit_type = orbit_type_map[self._spec.orbit_type]
        pa_type = pa_map[self._spec.position_angle]
        harvester.setReferenceState(
            self._propagator.getInitialState()
        )
        return harvester

    def propagate_with_covariance(
        self,
        target_date_or_epoch: Any,
    ) -> tuple[Any, CovarianceRecord]:
        """Propagate to the target epoch and return the state + propagated covariance.

        Parameters
        ----------
        target_date_or_epoch:
            Either an Orekit ``AbsoluteDate`` or an ISO-8601 UTC string.

        Returns
        -------
        tuple[SpacecraftState, CovarianceRecord]
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
        phi_n = phi[:n, :n]  # guard against larger harvested matrices
        cov_propagated = phi_n @ self._initial_cov @ phi_n.T

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
