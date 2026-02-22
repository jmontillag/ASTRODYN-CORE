"""Lightweight mission-analysis plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from astrodyn_core.states.models import OrbitStateRecord, StateSeries
from astrodyn_core.states.orekit_convert import to_orekit_orbit
from astrodyn_core.states.validation import parse_epoch_utc


def plot_orbital_elements_series(
    series: StateSeries,
    output_png: str | Path,
    *,
    universe: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """Plot time evolution of Keplerian elements and save to PNG."""
    if not isinstance(series, StateSeries):
        raise TypeError("series must be a StateSeries.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for orbital element plotting.") from exc

    ordered = sorted(series.states, key=lambda item: parse_epoch_utc(item.epoch))
    t0 = parse_epoch_utc(ordered[0].epoch)

    t_min: list[float] = []
    a_km: list[float] = []
    ecc: list[float] = []
    inc_deg: list[float] = []
    argp_deg: list[float] = []
    raan_deg: list[float] = []
    mean_anom_deg: list[float] = []

    for record in ordered:
        dt = parse_epoch_utc(record.epoch) - t0
        t_min.append(dt.total_seconds() / 60.0)
        elements = _record_to_keplerian_elements(record, universe=universe)
        a_km.append(elements["a_m"] / 1000.0)
        ecc.append(elements["e"])
        inc_deg.append(elements["i_deg"])
        argp_deg.append(elements["argp_deg"])
        raan_deg.append(elements["raan_deg"])
        mean_anom_deg.append(elements["anomaly_deg"])

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    ax = axes.ravel()
    ax[0].plot(t_min, a_km, lw=1.5)
    ax[0].set_ylabel("a [km]")
    ax[1].plot(t_min, ecc, lw=1.5)
    ax[1].set_ylabel("e [-]")
    ax[2].plot(t_min, inc_deg, lw=1.5)
    ax[2].set_ylabel("i [deg]")
    ax[3].plot(t_min, argp_deg, lw=1.5)
    ax[3].set_ylabel("argp [deg]")
    ax[4].plot(t_min, raan_deg, lw=1.5)
    ax[4].set_ylabel("RAAN [deg]")
    ax[4].set_xlabel("Time since first sample [min]")
    ax[5].plot(t_min, mean_anom_deg, lw=1.5)
    ax[5].set_ylabel("M [deg]")
    ax[5].set_xlabel("Time since first sample [min]")

    final_title = title or f"Orbital Elements Evolution: {series.name}"
    fig.suptitle(final_title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _record_to_keplerian_elements(
    record: OrbitStateRecord,
    *,
    universe: Mapping[str, Any] | None,
) -> dict[str, float]:
    if record.representation == "keplerian":
        el = record.elements or {}
        return {
            "a_m": float(el["a_m"]),
            "e": float(el["e"]),
            "i_deg": float(el["i_deg"]),
            "argp_deg": float(el["argp_deg"]),
            "raan_deg": float(el["raan_deg"]),
            "anomaly_deg": float(el["anomaly_deg"]),
        }

    orbit = to_orekit_orbit(record, universe=universe)
    from org.orekit.orbits import KeplerianOrbit

    kep = KeplerianOrbit(orbit)
    return {
        "a_m": float(kep.getA()),
        "e": float(kep.getE()),
        "i_deg": math.degrees(float(kep.getI())),
        "argp_deg": math.degrees(float(kep.getPerigeeArgument())),
        "raan_deg": math.degrees(float(kep.getRightAscensionOfAscendingNode())),
        "anomaly_deg": math.degrees(float(kep.getMeanAnomaly())),
    }
