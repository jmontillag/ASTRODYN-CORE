"""Frame and gravitational-parameter resolvers for Orekit state conversions."""

from __future__ import annotations

from typing import Any, Mapping

from astrodyn_core.propagation.universe import get_itrf_frame, get_mu


def resolve_frame(frame_name: str, universe: Mapping[str, Any] | None = None):
    """Resolve a frame name from state files into an Orekit frame."""
    try:
        from org.orekit.frames import FramesFactory
    except Exception as exc:
        raise RuntimeError(
            "Orekit classes are unavailable. Install package dependencies (orekit>=13.1)."
        ) from exc

    normalized = frame_name.strip().upper()
    if normalized == "GCRF":
        return FramesFactory.getGCRF()
    if normalized == "EME2000":
        return FramesFactory.getEME2000()
    if normalized == "TEME":
        return FramesFactory.getTEME()

    if normalized in {"ITRF", "ITRF_2020", "ITRF_2014", "ITRF_2008"}:
        cfg = dict(universe or {})
        if normalized != "ITRF":
            cfg["itrf_version"] = normalized
        return get_itrf_frame(cfg if cfg else None)

    raise ValueError(f"Unsupported frame '{frame_name}'.")


def resolve_mu(mu_m3_s2: float | str, universe: Mapping[str, Any] | None = None) -> float:
    """Resolve a numeric gravitational parameter in m^3/s^2."""
    if isinstance(mu_m3_s2, (int, float)):
        return float(mu_m3_s2)

    if isinstance(mu_m3_s2, str):
        cfg = dict(universe or {})
        cfg["gravitational_parameter"] = mu_m3_s2.strip().upper()
        return get_mu(cfg)

    raise TypeError("mu_m3_s2 must be a float or predefined model string.")
