"""Ephemeris-to-BoundedPropagator factory.

Creates Orekit ``BoundedPropagator`` objects from parsed ephemeris files.
Handles multi-file fusion via ``AggregateBoundedPropagator``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

from astrodyn_core.ephemeris.models import EphemerisFormat, EphemerisSource, EphemerisSpec
from astrodyn_core.ephemeris.parser import parse_cpf, parse_ocm, parse_oem, parse_sp3


def create_propagator_from_spec(
    spec: EphemerisSpec,
    *,
    edc_api_client: Any | None = None,
    edc_ftp_client: Any | None = None,
    cache_dir: str | Path = "data/cache",
) -> Any:
    """Create an Orekit bounded propagator from an :class:`EphemerisSpec`.

    This is the lower-level factory used by :class:`astrodyn_core.ephemeris.EphemerisClient`.
    It dispatches between local parsing (OEM/OCM) and remote acquisition flows
    (SP3/CPF via EDC clients).

    Args:
        spec: Immutable ephemeris request describing source and format.
        edc_api_client: EDC REST client required for remote sources (CPF/SP3
            satellite metadata lookups and CPF download).
        edc_ftp_client: Optional EDC FTP client for SP3 file downloads.
        cache_dir: Local cache directory used by remote file acquisition.

    Returns:
        An Orekit ``BoundedPropagator`` (or aggregate bounded propagator when
        multiple segments are fused).

    Raises:
        ValueError: If required clients are missing, the spec is unsupported, or
            no usable propagator can be created from the resolved data.
    """
    if spec.source == EphemerisSource.LOCAL:
        if spec.format == EphemerisFormat.OEM:
            return _create_oem_propagator(spec.file_path)
        elif spec.format == EphemerisFormat.OCM:
            return _create_ocm_propagator(spec.file_paths)

    elif spec.source == EphemerisSource.REMOTE:
        if edc_api_client is None:
            raise ValueError(
                "edc_api_client is required for remote ephemeris sources. "
                "Provide it via EphemerisClient or pass it directly."
            )
        return _create_remote_propagator(
            spec,
            edc_api_client=edc_api_client,
            edc_ftp_client=edc_ftp_client,
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unsupported ephemeris spec: {spec}")


# ---------------------------------------------------------------------------
# Local file propagator creation
# ---------------------------------------------------------------------------

def _create_oem_propagator(file_path: Path | None) -> Any:
    """Create a BoundedPropagator from a single OEM file."""
    from org.orekit.files.general import EphemerisFile

    if file_path is None:
        raise ValueError("file_path is required for OEM propagator.")

    oem = parse_oem(file_path)
    satellites = oem.getSatellites()
    if not satellites or satellites.isEmpty():
        raise ValueError(f"No satellites found in OEM file: {file_path}")

    sat_id = satellites.keySet().iterator().next()
    oem_sat = satellites.get(sat_id)
    if oem_sat.getSegments().isEmpty():
        raise ValueError(f"No segments found for satellite in OEM file: {file_path}")

    return EphemerisFile.SatelliteEphemeris.cast_(oem_sat).getPropagator()


def _create_ocm_propagator(file_paths: tuple[Path, ...] | None) -> Any:
    """Create a (possibly fused) BoundedPropagator from OCM files."""
    if not file_paths:
        raise ValueError("file_paths is required for OCM propagator.")

    propagators = []
    for path in file_paths:
        ocm = parse_ocm(path)
        for sat_id in ocm.getSatellites().keySet():
            propagators.append(ocm.getSatellites().get(sat_id).getPropagator())

    if not propagators:
        raise ValueError("No usable propagators from OCM files.")

    if len(propagators) == 1:
        return propagators[0]

    return _fuse_bounded_propagators(propagators)


# ---------------------------------------------------------------------------
# Remote propagator creation
# ---------------------------------------------------------------------------

def _create_remote_propagator(
    spec: EphemerisSpec,
    *,
    edc_api_client: Any,
    edc_ftp_client: Any | None,
    cache_dir: str | Path,
) -> Any:
    """Create a BoundedPropagator by fetching data from EDC."""
    from astrodyn_core.ephemeris.downloader import (
        EDCFtpClient,
        EphemerisFileProcessor,
    )

    ftp_client = edc_ftp_client or EDCFtpClient()
    processor = EphemerisFileProcessor(edc_api_client, ftp_client, cache_dir=cache_dir)

    # Resolve satellite info
    sat_info = edc_api_client.get_satellite_info(spec.identifier_type, spec.satellite_name)
    if not sat_info:
        raise ValueError(
            f"No satellite info returned for {spec.identifier_type}={spec.satellite_name}"
        )

    if spec.format == EphemerisFormat.SP3:
        return _create_sp3_propagator(
            sat_name=sat_info.get("satellite_name", spec.satellite_name),
            start_date=spec.start_date,
            end_date=spec.end_date,
            provider_preference=spec.provider_preference,
            ftp_client=ftp_client,
            processor=processor,
        )

    elif spec.format == EphemerisFormat.CPF:
        return _create_cpf_propagator(
            cospar_id=sat_info.get("satellite_id", ""),
            start_date=spec.start_date,
            end_date=spec.end_date,
            local_root=spec.local_root,
            sat_name=sat_info.get("satellite_name", spec.satellite_name),
            api_client=edc_api_client,
            processor=processor,
        )

    raise ValueError(f"Unsupported remote format: {spec.format}")


def _create_sp3_propagator(
    *,
    sat_name: str,
    start_date: str,
    end_date: str,
    provider_preference: tuple[str, ...] | None,
    ftp_client: Any,
    processor: Any,
) -> Any:
    """Create a BoundedPropagator from SP3 files downloaded via FTP."""
    from datetime import datetime

    from org.orekit.files.sp3 import SP3

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    file_records = ftp_client.list_sp3_files(
        sat_name, start_dt, end_dt,
        provider_preference=provider_preference,
    )

    if not file_records:
        raise ValueError(f"No SP3 files found for {sat_name} between {start_date} and {end_date}.")

    parsed_sp3s = []
    for record in file_records:
        local_path = processor.get_sp3_file(record)
        if local_path:
            try:
                parsed_sp3s.append(parse_sp3(local_path))
            except Exception as exc:
                warnings.warn(f"Failed to parse SP3 file {local_path}: {exc}", stacklevel=2)

    if not parsed_sp3s:
        raise ValueError("No SP3 files could be parsed successfully.")

    if len(parsed_sp3s) > 1:
        from java.util import ArrayList

        java_list = ArrayList()
        for sp3 in parsed_sp3s:
            java_list.add(sp3)
        final_sp3 = SP3.splice(java_list)
    else:
        final_sp3 = parsed_sp3s[0]

    sat_map = final_sp3.getSatellites()
    sp3_ephem = list(sat_map.values())[0]
    return sp3_ephem.getPropagator()


def _create_cpf_propagator(
    *,
    cospar_id: str,
    start_date: str,
    end_date: str,
    local_root: str | None,
    sat_name: str,
    api_client: Any,
    processor: Any,
) -> Any:
    """Create a BoundedPropagator from CPF files (local or API)."""
    propagators = []

    # Try local files first if a local_root is given
    if local_root:
        propagators = _load_local_cpf_propagators(
            local_root, sat_name, start_date, end_date,
        )

    # Fall back to API download
    if not propagators:
        file_records = api_client.query_cpf_files(cospar_id, start_date, end_date)
        for record in file_records:
            local_path = processor.get_cpf_file(record)
            if local_path:
                try:
                    cpf = parse_cpf(local_path)
                    for sat_id in cpf.getSatellites().keySet():
                        propagators.append(cpf.getSatellites().get(sat_id).getPropagator())
                except Exception as exc:
                    warnings.warn(f"Failed to parse CPF {local_path}: {exc}", stacklevel=2)

    if not propagators:
        raise ValueError(f"No CPF data found for {cospar_id} between {start_date} and {end_date}.")

    if len(propagators) == 1:
        return propagators[0]

    return _fuse_bounded_propagators(propagators)


def _load_local_cpf_propagators(
    local_root: str,
    sat_name: str,
    start_date: str,
    end_date: str,
) -> list[Any]:
    """Scan a local directory for CPF files matching the satellite name."""
    import os
    from collections import defaultdict
    from datetime import datetime, timedelta

    propagators = []
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return []

    sat_prefix_lower = sat_name.lower()
    candidates: list[tuple[str, str]] = []  # (day_str, filepath)

    current = start_dt
    while current <= end_dt:
        day_folder = current.strftime("%y%m%d")
        cpf_dir = os.path.join(local_root, day_folder, "_CPF")
        if os.path.isdir(cpf_dir):
            for fname in os.listdir(cpf_dir):
                fpath = os.path.join(cpf_dir, fname)
                if os.path.isfile(fpath) and fname.lower().startswith(sat_prefix_lower):
                    candidates.append((day_folder, fpath))
        current += timedelta(days=1)

    if not candidates:
        return []

    # Pick latest version per day
    by_day: dict[str, list[str]] = defaultdict(list)
    for day, fpath in candidates:
        by_day[day].append(fpath)

    selected = []
    for day_key in sorted(by_day):
        files = sorted(by_day[day_key], reverse=True)
        selected.append(files[0])

    for path in selected:
        try:
            cpf = parse_cpf(path)
            for sat_id in cpf.getSatellites().keySet():
                propagators.append(cpf.getSatellites().get(sat_id).getPropagator())
        except Exception as exc:
            warnings.warn(f"Failed to parse local CPF {path}: {exc}", stacklevel=2)

    return propagators


# ---------------------------------------------------------------------------
# Multi-propagator fusion
# ---------------------------------------------------------------------------

def _fuse_bounded_propagators(propagators: list[Any]) -> Any:
    """Fuse multiple BoundedPropagators into a single AggregateBoundedPropagator."""
    from java.util import ArrayList
    from orekit.pyhelpers import absolutedate_to_datetime
    from org.orekit.propagation.analytical import AggregateBoundedPropagator

    propagators.sort(key=lambda p: absolutedate_to_datetime(p.getMinDate()))
    java_list = ArrayList()
    for p in propagators:
        java_list.add(p)
    return AggregateBoundedPropagator(java_list)
