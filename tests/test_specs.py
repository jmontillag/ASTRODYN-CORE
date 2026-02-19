from astrodyn_core.propagation.specs import IntegratorSpec, PropagatorKind, PropagatorSpec, TLESpec


def test_numerical_requires_integrator() -> None:
    try:
        PropagatorSpec(kind=PropagatorKind.NUMERICAL)
    except ValueError as exc:
        assert "integrator is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_tle_requires_tle_spec() -> None:
    try:
        PropagatorSpec(kind=PropagatorKind.TLE)
    except ValueError as exc:
        assert "tle is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_tle_spec_validates_lines() -> None:
    try:
        TLESpec(line1="X 00000", line2="2 00000")
    except ValueError as exc:
        assert "line1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_integrator_spec_normalizes_kind() -> None:
    spec = IntegratorSpec(kind=" DP853 ")
    assert spec.kind == "dp853"


def test_dsst_types_normalize_to_uppercase() -> None:
    spec = PropagatorSpec(
        kind=PropagatorKind.DSST,
        integrator=IntegratorSpec(kind="dp853", min_step=0.1, max_step=30.0, position_tolerance=10.0),
        dsst_propagation_type=" mean ",
        dsst_state_type=" osculating ",
    )
    assert spec.dsst_propagation_type == "MEAN"
    assert spec.dsst_state_type == "OSCULATING"


def test_dsst_rejects_invalid_state_or_propagation_type() -> None:
    try:
        PropagatorSpec(
            kind=PropagatorKind.DSST,
            integrator=IntegratorSpec(
                kind="dp853",
                min_step=0.1,
                max_step=30.0,
                position_tolerance=10.0,
            ),
            dsst_propagation_type="bad",
        )
    except ValueError as exc:
        assert "dsst_propagation_type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
