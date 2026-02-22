from astrodyn_core.propagation.universe import get_universe_config, load_universe_from_dict


def test_universe_config_defaults_from_empty_dict() -> None:
    cfg = load_universe_from_dict({})
    assert cfg["iers_conventions"] == "IERS_2010"
    assert cfg["itrf_version"] == "ITRF_2020"
    assert cfg["use_simple_eop"] is True
    assert cfg["earth_shape_model"] == "WGS84"
    assert cfg["gravitational_parameter"] == "WGS84"


def test_universe_config_accepts_nested_universe_section() -> None:
    cfg = load_universe_from_dict(
        {
            "universe": {
                "iers_conventions": "IERS_2003",
                "itrf_version": "ITRF_2014",
                "use_simple_eop": False,
                "earth_shape_model": "GRS80",
                "gravitational_parameter": 3.986004415e14,
            }
        }
    )

    assert cfg["iers_conventions"] == "IERS_2003"
    assert cfg["itrf_version"] == "ITRF_2014"
    assert cfg["use_simple_eop"] is False
    assert cfg["earth_shape_model"] == "GRS80"
    assert cfg["gravitational_parameter"] == 3.986004415e14


def test_universe_config_accepts_custom_earth_shape_mapping() -> None:
    cfg = load_universe_from_dict(
        {
            "earth_shape_model": {
                "type": "custom",
                "equatorial_radius": 6378137.0,
                "flattening": 1.0 / 298.257223563,
            }
        }
    )
    shape = cfg["earth_shape_model"]

    assert isinstance(shape, dict)
    assert shape["type"] == "custom"
    assert shape["equatorial_radius"] == 6378137.0


def test_universe_config_rejects_invalid_iers_conventions() -> None:
    try:
        load_universe_from_dict({"iers_conventions": "IERS_1988"})
    except ValueError as exc:
        assert "iers_conventions" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_get_universe_config_returns_mapping() -> None:
    cfg = get_universe_config()
    assert isinstance(cfg, dict)
    assert "earth_shape_model" in cfg


def test_universe_config_coerces_use_simple_eop_string() -> None:
    cfg = load_universe_from_dict({"use_simple_eop": "false"})
    assert cfg["use_simple_eop"] is False
