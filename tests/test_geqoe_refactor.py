import numpy as np
import pytest

from astrodyn_core.propagation.geqoe import core as refactor_core
from astrodyn_core.propagation.geqoe._legacy_loader import import_legacy_module


legacy_propagator = import_legacy_module("temp_mosaic_modules.geqoe_utils.propagator")
legacy_conversion = import_legacy_module("temp_mosaic_modules.geqoe_utils.conversion")


def _earth_constants() -> tuple[float, float, float]:
    return (1.08262668e-3, 6378137.0, 3.986004418e14)


def _reference_state() -> np.ndarray:
    return np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 1000.0], dtype=float)


def test_geqoe_refactor_taylor_cart_parity() -> None:
    p = _earth_constants()
    y0 = _reference_state()
    tspan = np.array([0.0, 30.0, 120.0, 600.0], dtype=float)

    for order in (1, 2, 3, 4):
        y_new, stm_new = refactor_core.taylor_cart_propagator(tspan=tspan, y0=y0, p=p, order=order)
        y_old, stm_old = legacy_propagator.taylor_cart_propagator(tspan=tspan, Y0=y0, p=p, order=order)

        np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)


def test_geqoe_refactor_j2_core_parity() -> None:
    p = _earth_constants()
    y0_cart = _reference_state()
    dt = np.array([0.0, 15.0, 75.0], dtype=float)

    eq0_tuple = legacy_conversion.rv2geqoe(t=0.0, y=y0_cart, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    for order in (1, 2, 3, 4):
        y_new, stm_new, map_new = refactor_core.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=order)
        y_old, stm_old, map_old = legacy_propagator.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=order)

        np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(map_new, map_old, rtol=1e-13, atol=1e-13)


def test_geqoe_context_contract_and_fixed_history_sizes() -> None:
    p = _earth_constants()
    y0 = np.array([0.001, 0.0, 0.0, 0.001, 0.001, 0.1], dtype=float)
    dt = np.array([0.0, 10.0], dtype=float)

    context = refactor_core.build_context(dt=dt, y0=y0, p=p, order=4)

    assert context.order == 4
    assert context.dt_seconds.shape == (2,)
    assert context.dt_norm.shape == (2,)
    assert context.initial_state.as_array().shape == (6,)
    assert context.histories.beta.shape == (4,)
    assert context.histories.c.shape == (4,)
    assert context.histories.r.shape == (4,)
    assert context.histories.r2.shape == (4,)
    assert context.histories.r3.shape == (4,)
    assert context.histories.h.shape == (4,)
    assert context.histories.alpha.shape == (4,)
    assert context.histories.beta_plus_one.shape == (4,)
    assert context.histories.hr3.shape == (4,)
    assert context.histories.delta.shape == (4,)
    assert context.histories.delta_denominator.shape == (4,)


def test_geqoe_backend_and_order_validation() -> None:
    p = _earth_constants()
    y0 = np.array([0.001, 0.0, 0.0, 0.001, 0.001, 0.1], dtype=float)
    dt = np.array([0.0], dtype=float)

    with pytest.raises(ValueError):
        refactor_core.build_context(dt=dt, y0=y0, p=p, order=0)

    with pytest.raises(ValueError):
        refactor_core.j2_taylor_propagator(dt=dt, y0=y0, p=p, order=1, backend="invalid")


def test_geqoe_staged_order1_parity_j2() -> None:
    p = _earth_constants()
    y0_cart = _reference_state()
    dt = np.array([0.0, 15.0, 75.0], dtype=float)

    eq0_tuple = legacy_conversion.rv2geqoe(t=0.0, y=y0_cart, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    y_new, stm_new, map_new = refactor_core.j2_taylor_propagator(
        dt=dt, y0=eq0, p=p, order=1, backend="staged"
    )
    y_old, stm_old, map_old = legacy_propagator.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=1)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(map_new, map_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order1_parity_cartesian() -> None:
    p = _earth_constants()
    y0 = _reference_state()
    tspan = np.array([0.0, 30.0, 120.0], dtype=float)

    y_new, stm_new = refactor_core.taylor_cart_propagator(
        tspan=tspan, y0=y0, p=p, order=1, backend="staged"
    )
    y_old, stm_old = legacy_propagator.taylor_cart_propagator(tspan=tspan, Y0=y0, p=p, order=1)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order2_parity_j2() -> None:
    p = _earth_constants()
    y0_cart = _reference_state()
    dt = np.array([0.0, 15.0, 75.0], dtype=float)

    eq0_tuple = legacy_conversion.rv2geqoe(t=0.0, y=y0_cart, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    y_new, stm_new, map_new = refactor_core.j2_taylor_propagator(
        dt=dt, y0=eq0, p=p, order=2, backend="staged"
    )
    y_old, stm_old, map_old = legacy_propagator.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=2)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(map_new, map_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order2_parity_cartesian() -> None:
    p = _earth_constants()
    y0 = _reference_state()
    tspan = np.array([0.0, 30.0, 120.0], dtype=float)

    y_new, stm_new = refactor_core.taylor_cart_propagator(
        tspan=tspan, y0=y0, p=p, order=2, backend="staged"
    )
    y_old, stm_old = legacy_propagator.taylor_cart_propagator(tspan=tspan, Y0=y0, p=p, order=2)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order3_parity_j2() -> None:
    p = _earth_constants()
    y0_cart = _reference_state()
    dt = np.array([0.0, 15.0, 75.0], dtype=float)

    eq0_tuple = legacy_conversion.rv2geqoe(t=0.0, y=y0_cart, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    y_new, stm_new, map_new = refactor_core.j2_taylor_propagator(
        dt=dt, y0=eq0, p=p, order=3, backend="staged"
    )
    y_old, stm_old, map_old = legacy_propagator.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=3)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(map_new, map_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order3_parity_cartesian() -> None:
    p = _earth_constants()
    y0 = _reference_state()
    tspan = np.array([0.0, 30.0, 120.0], dtype=float)

    y_new, stm_new = refactor_core.taylor_cart_propagator(
        tspan=tspan, y0=y0, p=p, order=3, backend="staged"
    )
    y_old, stm_old = legacy_propagator.taylor_cart_propagator(tspan=tspan, Y0=y0, p=p, order=3)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order4_parity_j2() -> None:
    p = _earth_constants()
    y0_cart = _reference_state()
    dt = np.array([0.0, 15.0, 75.0], dtype=float)

    eq0_tuple = legacy_conversion.rv2geqoe(t=0.0, y=y0_cart, p=p)
    eq0 = np.hstack([elem.flatten() for elem in eq0_tuple])

    y_new, stm_new, map_new = refactor_core.j2_taylor_propagator(
        dt=dt, y0=eq0, p=p, order=4, backend="staged"
    )
    y_old, stm_old, map_old = legacy_propagator.j2_taylor_propagator(dt=dt, y0=eq0, p=p, order=4)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(map_new, map_old, rtol=1e-13, atol=1e-13)


def test_geqoe_staged_order4_parity_cartesian() -> None:
    p = _earth_constants()
    y0 = _reference_state()
    tspan = np.array([0.0, 30.0, 120.0], dtype=float)

    y_new, stm_new = refactor_core.taylor_cart_propagator(
        tspan=tspan, y0=y0, p=p, order=4, backend="staged"
    )
    y_old, stm_old = legacy_propagator.taylor_cart_propagator(tspan=tspan, Y0=y0, p=p, order=4)

    np.testing.assert_allclose(y_new, y_old, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(stm_new, stm_old, rtol=1e-13, atol=1e-13)
