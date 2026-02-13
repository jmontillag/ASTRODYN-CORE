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
