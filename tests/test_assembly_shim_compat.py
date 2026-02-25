from astrodyn_core.propagation.assembly import (
    assemble_attitude_provider as assemble_attitude_provider_shim,
    assemble_force_models as assemble_force_models_shim,
    build_atmosphere as build_atmosphere_shim,
    build_spacecraft_drag_shape as build_spacecraft_drag_shape_shim,
    get_celestial_body as get_celestial_body_shim,
)
from astrodyn_core.propagation.assembly_parts import (
    assemble_attitude_provider as assemble_attitude_provider_new,
    assemble_force_models as assemble_force_models_new,
    build_atmosphere as build_atmosphere_new,
    build_spacecraft_drag_shape as build_spacecraft_drag_shape_new,
    get_celestial_body as get_celestial_body_new,
)
from astrodyn_core.propagation.dsst_assembly import (
    assemble_dsst_force_models as assemble_dsst_force_models_shim,
)
from astrodyn_core.propagation.dsst_parts import (
    assemble_dsst_force_models as assemble_dsst_force_models_new,
)


def test_assembly_shim_symbols_alias_canonical() -> None:
    assert assemble_attitude_provider_shim is assemble_attitude_provider_new
    assert assemble_force_models_shim is assemble_force_models_new
    assert build_atmosphere_shim is build_atmosphere_new
    assert build_spacecraft_drag_shape_shim is build_spacecraft_drag_shape_new
    assert get_celestial_body_shim is get_celestial_body_new


def test_dsst_assembly_shim_symbol_aliases_canonical() -> None:
    assert assemble_dsst_force_models_shim is assemble_dsst_force_models_new
