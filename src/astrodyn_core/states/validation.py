"""Validation helpers for file-defined orbital states."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

SUPPORTED_REPRESENTATIONS = frozenset({"cartesian", "keplerian", "equinoctial"})
SUPPORTED_FRAMES = frozenset(
    {
        "GCRF",
        "EME2000",
        "TEME",
        "ITRF",
        "ITRF_2020",
        "ITRF_2014",
        "ITRF_2008",
    }
)
SUPPORTED_ANOMALY_TYPES = frozenset({"MEAN", "ECCENTRIC", "TRUE"})


def parse_epoch_utc(epoch: str) -> datetime:
    """Parse an ISO-8601 epoch string and return a UTC-aware datetime."""
    if not isinstance(epoch, str) or not epoch.strip():
        raise ValueError("epoch must be a non-empty ISO-8601 string.")

    normalized = epoch.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid epoch '{epoch}'. Expected ISO-8601 format.") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_orbit_state(
    *,
    epoch: str,
    frame: str,
    representation: str,
    position_m: Sequence[Any] | None,
    velocity_mps: Sequence[Any] | None,
    elements: Mapping[str, Any] | None,
    mu_m3_s2: float | str,
    mass_kg: float | None,
) -> dict[str, Any]:
    """Validate and normalize OrbitStateRecord fields."""
    parse_epoch_utc(epoch)

    frame_norm = str(frame).strip().upper()
    if frame_norm not in SUPPORTED_FRAMES:
        raise ValueError(f"Unsupported frame '{frame}'. Supported frames: {sorted(SUPPORTED_FRAMES)}")

    rep_norm = str(representation).strip().lower()
    if rep_norm not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported representation '{representation}'. "
            f"Supported: {sorted(SUPPORTED_REPRESENTATIONS)}"
        )

    mu_norm = _normalize_mu(mu_m3_s2)
    mass_norm = _normalize_mass(mass_kg)
    elements_norm: dict[str, Any] | None = None
    pos_norm: tuple[float, float, float] | None = None
    vel_norm: tuple[float, float, float] | None = None

    if rep_norm == "cartesian":
        pos_norm = _normalize_vector3("position_m", position_m)
        vel_norm = _normalize_vector3("velocity_mps", velocity_mps)
    elif rep_norm == "keplerian":
        elements_norm = _normalize_keplerian_elements(elements)
    elif rep_norm == "equinoctial":
        elements_norm = _normalize_equinoctial_elements(elements)

    return {
        "frame": frame_norm,
        "representation": rep_norm,
        "position_m": pos_norm,
        "velocity_mps": vel_norm,
        "elements": elements_norm,
        "mu_m3_s2": mu_norm,
        "mass_kg": mass_norm,
    }


def _normalize_mu(value: float | str) -> float | str:
    if isinstance(value, (int, float)):
        mu = float(value)
        if mu <= 0:
            raise ValueError("mu_m3_s2 must be positive.")
        return mu

    if isinstance(value, str):
        mu_name = value.strip().upper()
        if not mu_name:
            raise ValueError("mu_m3_s2 cannot be an empty string.")
        return mu_name

    raise TypeError("mu_m3_s2 must be a float or predefined model string.")


def _normalize_mass(value: float | None) -> float | None:
    if value is None:
        return None
    mass = float(value)
    if mass <= 0:
        raise ValueError("mass_kg must be positive when provided.")
    return mass


def _normalize_vector3(name: str, value: Sequence[Any] | None) -> tuple[float, float, float]:
    if value is None:
        raise ValueError(f"{name} is required for cartesian representation.")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a 3-element sequence of numbers.")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 values.")

    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


def _normalize_keplerian_elements(elements: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(elements, Mapping):
        raise ValueError("elements mapping is required for keplerian representation.")

    required = ("a_m", "e", "i_deg", "argp_deg", "raan_deg", "anomaly_deg")
    missing = [key for key in required if key not in elements]
    if missing:
        raise ValueError(f"keplerian elements missing required fields: {missing}")

    anomaly_type = str(elements.get("anomaly_type", "MEAN")).strip().upper()
    if anomaly_type not in SUPPORTED_ANOMALY_TYPES:
        raise ValueError(f"Unsupported anomaly_type '{anomaly_type}'.")

    normalized = {
        "a_m": float(elements["a_m"]),
        "e": float(elements["e"]),
        "i_deg": float(elements["i_deg"]),
        "argp_deg": float(elements["argp_deg"]),
        "raan_deg": float(elements["raan_deg"]),
        "anomaly_deg": float(elements["anomaly_deg"]),
        "anomaly_type": anomaly_type,
    }
    return normalized


def _normalize_equinoctial_elements(elements: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(elements, Mapping):
        raise ValueError("elements mapping is required for equinoctial representation.")

    required = ("a_m", "ex", "ey", "hx", "hy", "l_deg")
    missing = [key for key in required if key not in elements]
    if missing:
        raise ValueError(f"equinoctial elements missing required fields: {missing}")

    anomaly_type = str(elements.get("anomaly_type", "MEAN")).strip().upper()
    if anomaly_type not in SUPPORTED_ANOMALY_TYPES:
        raise ValueError(f"Unsupported anomaly_type '{anomaly_type}'.")

    normalized = {
        "a_m": float(elements["a_m"]),
        "ex": float(elements["ex"]),
        "ey": float(elements["ey"]),
        "hx": float(elements["hx"]),
        "hy": float(elements["hy"]),
        "l_deg": float(elements["l_deg"]),
        "anomaly_type": anomaly_type,
    }
    return normalized
