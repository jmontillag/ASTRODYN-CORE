from astrodyn_core.orekit_env import (
    get_universe_config as get_universe_config_new,
    load_universe_from_dict as load_universe_from_dict_new,
)
from astrodyn_core.propagation.universe import (
    get_universe_config as get_universe_config_shim,
    load_universe_from_dict as load_universe_from_dict_shim,
)


def test_universe_shim_load_equivalent_to_canonical() -> None:
    payload = {
        "universe": {
            "iers_conventions": "IERS_2003",
            "itrf_version": "ITRF_2014",
            "use_simple_eop": "false",
            "earth_shape_model": "GRS80",
            "gravitational_parameter": "WGS84",
        }
    }

    cfg_shim = load_universe_from_dict_shim(payload)
    cfg_new = load_universe_from_dict_new(payload)

    assert cfg_shim == cfg_new


def test_universe_shim_get_config_equivalent_to_canonical() -> None:
    cfg_shim = get_universe_config_shim()
    cfg_new = get_universe_config_new()

    assert cfg_shim == cfg_new
