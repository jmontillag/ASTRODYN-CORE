from astrodyn_core import get_spacecraft_model
from astrodyn_core.data import list_spacecraft_models
from astrodyn_core.propagation.parsers.spacecraft import (
    load_spacecraft_config,
    load_spacecraft_from_dict,
)


def test_list_spacecraft_models_contains_examples() -> None:
    names = list_spacecraft_models()
    assert "leo_smallsat.yaml" in names
    assert "cubesat_6u.yaml" in names
    assert "box_wing_bus.yaml" in names


def test_load_structured_spacecraft_isotropic() -> None:
    sc = load_spacecraft_config(get_spacecraft_model("leo_smallsat"))
    assert sc.mass == 450.0
    assert sc.use_box_wing is False
    assert sc.drag_area == 4.0
    assert sc.srp_area == 4.0


def test_load_structured_spacecraft_box_wing() -> None:
    sc = load_spacecraft_config(get_spacecraft_model("box_wing_bus"))
    assert sc.use_box_wing is True
    assert sc.x_length == 2.2
    assert sc.solar_array_area == 12.0


def test_legacy_flat_spacecraft_mapping_still_supported() -> None:
    sc = load_spacecraft_from_dict({"mass": 300.0, "srp_area": 2.0})
    assert sc.mass == 300.0
    assert sc.srp_area == 2.0
