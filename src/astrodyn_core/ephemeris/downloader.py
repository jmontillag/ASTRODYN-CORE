"""Remote ephemeris data acquisition from EDC (DGFI-TUM).

Provides clients for:
- EDC FTP: anonymous FTP for SP3 precise orbit files
- EDC REST API: authenticated API for CPF prediction files and satellite metadata

These are ported from the ASTROR repository's ``extra_utils.query_utils``
module and adapted to the ASTRODYN-CORE architecture (no pandas dependency,
pure-Python data structures).
"""

from __future__ import annotations

import configparser
import ftplib
import gzip
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SP3FileRecord:
    """Metadata describing one SP3 file candidate on the EDC FTP server.

    Attributes:
        date: File date in ``YYYY-MM-DD`` format.
        filename: Remote filename (usually ``*.sp3.gz``).
        ftp_url: Full FTP URL to download the file.
    """

    date: str  # YYYY-MM-DD
    filename: str
    ftp_url: str


@dataclass(frozen=True, slots=True)
class CPFFileRecord:
    """Metadata describing one CPF file candidate returned by the EDC API.

    Attributes:
        file_id: EDC internal file identifier used for download requests.
        start_date: File start date in ``YYYY-MM-DD`` format.
        filename: Suggested/remote filename for the CPF record.
    """

    file_id: int
    start_date: str
    filename: str


# ---------------------------------------------------------------------------
# EDC FTP Client (anonymous, for SP3)
# ---------------------------------------------------------------------------

DEFAULT_SP3_PROVIDERS: tuple[str, ...] = (
    "ilrsa", "ilrsb", "esa", "gfz", "asi", "bkg", "dgfi", "jcet", "nsgf",
)


class EDCFtpClient:
    """Anonymous FTP client for EDC SP3 orbit files.

    Args:
        ftp_server: FTP host name for the EDC service.
    """

    def __init__(self, ftp_server: str = "edc.dgfi.tum.de") -> None:
        self.ftp_server = ftp_server

    def list_sp3_files(
        self,
        sat_name: str,
        start_date: datetime,
        end_date: datetime,
        provider_preference: Sequence[str] | None = None,
    ) -> list[SP3FileRecord]:
        """List and select the best SP3 file per day for a satellite.

        Selection policy:

        - scan date-stamped EDC folders overlapping the requested range
        - group discovered files by day
        - choose the first available provider from ``provider_preference``
        - break ties by descending filename (highest version wins)

        Args:
            sat_name: Satellite directory name used by the EDC FTP archive.
            start_date: Inclusive start datetime used to choose folders/files.
            end_date: Inclusive end datetime used to choose folders/files.
            provider_preference: Optional ordered provider list. If omitted, the
                built-in default provider priority is used.

        Returns:
            A list containing at most one :class:`SP3FileRecord` per day.
            Returns an empty list on failure or when no files are found.
        """
        preference = list(provider_preference) if provider_preference else list(DEFAULT_SP3_PROVIDERS)
        all_records: list[dict[str, Any]] = []

        try:
            with ftplib.FTP(self.ftp_server) as ftp:
                ftp.login()

                base_dir = f"/pub/slr/products/orbits/{sat_name}"
                all_folders = ftp.nlst(base_dir)

                # Parse date-stamped folders
                dated_folders: list[tuple[datetime, str]] = []
                for folder in all_folders:
                    folder_name = folder.rsplit("/", 1)[-1]
                    try:
                        folder_date = datetime.strptime(folder_name, "%y%m%d")
                        if folder_date.year <= datetime.now().year + 1:
                            dated_folders.append((folder_date, folder))
                    except ValueError:
                        continue

                dated_folders.sort()

                # Find range of folders to scan
                start_idx = 0
                for i, (fdate, _) in enumerate(dated_folders):
                    if fdate >= start_date:
                        start_idx = max(0, i - 1)
                        break
                else:
                    start_idx = max(0, len(dated_folders) - 1)

                end_idx = len(dated_folders) - 1
                for i, (fdate, _) in enumerate(dated_folders):
                    if fdate > end_date:
                        end_idx = i
                        break

                for date, remote_dir in dated_folders[start_idx : end_idx + 1]:
                    try:
                        files_on_server = ftp.nlst(remote_dir)
                        for fname_path in files_on_server:
                            if fname_path.lower().endswith(".sp3.gz"):
                                all_records.append({
                                    "date": date.strftime("%Y-%m-%d"),
                                    "filename": os.path.basename(fname_path),
                                    "ftp_url": f"ftp://{self.ftp_server}{fname_path}",
                                })
                    except ftplib.error_perm:
                        pass

        except ftplib.all_errors as exc:
            warnings.warn(f"FTP operation failed: {exc}", stacklevel=2)
            return []

        if not all_records:
            return []

        # Group by date and pick best per day
        by_date: dict[str, list[dict[str, Any]]] = {}
        for rec in all_records:
            by_date.setdefault(rec["date"], []).append(rec)

        selected: list[SP3FileRecord] = []
        for date_str in sorted(by_date):
            day_files = by_date[date_str]
            best = None
            for provider in preference:
                provider_files = [
                    f for f in day_files
                    if f["filename"].startswith(provider + ".")
                ]
                if provider_files:
                    provider_files.sort(key=lambda f: f["filename"], reverse=True)
                    best = provider_files[0]
                    break
            if best:
                selected.append(SP3FileRecord(
                    date=best["date"],
                    filename=best["filename"],
                    ftp_url=best["ftp_url"],
                ))

        return selected

    def download_file(self, ftp_url: str, local_path: str | Path) -> bool:
        """Download a single file from a full FTP URL.

        Args:
            ftp_url: Full FTP URL to download.
            local_path: Destination path for the downloaded bytes.

        Returns:
            ``True`` on success, ``False`` if the FTP operation fails.
        """
        try:
            parsed = urlparse(ftp_url)
            with ftplib.FTP(parsed.hostname) as ftp:
                ftp.login()
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {parsed.path}", f.write)
            return True
        except ftplib.all_errors as exc:
            warnings.warn(f"FTP download failed for {ftp_url}: {exc}", stacklevel=2)
            return False


# ---------------------------------------------------------------------------
# EDC REST API Client (authenticated, for CPF + satellite info)
# ---------------------------------------------------------------------------

class EDCApiClient:
    """REST API client for the EDC (DGFI-TUM) service.

    Args:
        username: EDC account username.
        password: EDC account password.
        base_url: Base URL for the EDC API endpoint.
    """

    def __init__(
        self,
        username: str,
        password: str,
        base_url: str = "https://edc.dgfi.tum.de/api/v1/",
    ) -> None:
        self._auth = {"username": username, "password": password}
        self._base_url = base_url

    def get_satellite_info(
        self,
        identifier_type: str,
        identifier_value: str,
    ) -> dict[str, Any] | None:
        """Look up satellite metadata (COSPAR ID, name, aliases, etc.).

        Args:
            identifier_type: Query field name understood by the EDC API (for
                example ``"satellite_name"`` or ``"cospar_id"``).
            identifier_value: Identifier value to search for.

        Returns:
            A metadata mapping when found, or ``None`` on not-found/request
            failure.
        """
        payload = {
            **self._auth,
            "action": "satellite-info",
            identifier_type: identifier_value,
        }
        try:
            resp = requests.post(self._base_url, data=payload, timeout=30)
            if resp.status_code == 603:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            warnings.warn(f"EDC satellite-info request failed: {exc}", stacklevel=2)
            return None

    def query_cpf_files(
        self,
        cospar_id: str,
        start_date: str,
        end_date: str,
    ) -> list[CPFFileRecord]:
        """Query for the best available CPF file per day in a date range.

        Args:
            cospar_id: COSPAR identifier used by the EDC CPF catalog.
            start_date: Inclusive start date in ``YYYY-MM-DD`` format.
            end_date: Inclusive end date in ``YYYY-MM-DD`` format.

        Returns:
            A list containing at most one :class:`CPFFileRecord` per day.
            Returns an empty list when no data is found or when inputs are
            invalid.
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            warnings.warn("Invalid date format for CPF query, expected YYYY-MM-DD.", stacklevel=2)
            return []

        all_results: list[dict[str, Any]] = []
        current = start_dt
        while current <= end_dt:
            daily_query_date = current.strftime("%Y-%m-%d") + "%"
            payload = {
                **self._auth,
                "action": "data-query",
                "data_type": "CPF_v2",
                "satellite": cospar_id,
                "start_data_date": daily_query_date,
            }
            try:
                resp = requests.post(self._base_url, data=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json() if resp.text else []
                if data:
                    all_results.extend(data)
            except Exception:
                pass
            current += timedelta(days=1)

        if not all_results:
            return []

        # Pick best per day (highest eph_seq, highest eph_seq_daily)
        by_day: dict[str, list[dict[str, Any]]] = {}
        for rec in all_results:
            sdate = rec.get("start_data_date", "")[:10]
            by_day.setdefault(sdate, []).append(rec)

        selected: list[CPFFileRecord] = []
        for day_key in sorted(by_day):
            day_recs = by_day[day_key]
            day_recs.sort(
                key=lambda r: (
                    int(r.get("eph_seq", 0) or 0),
                    int(r.get("eph_seq_daily", 0) or 0),
                ),
                reverse=True,
            )
            best = day_recs[0]
            file_id = best.get("id")
            if file_id is not None:
                selected.append(CPFFileRecord(
                    file_id=int(file_id),
                    start_date=day_key,
                    filename=best.get("incoming_filename", f"cpf_{file_id}"),
                ))

        return selected

    def download_cpf_content(self, file_id: int) -> list[str] | None:
        """Download raw CPF file content by EDC file ID.

        Args:
            file_id: EDC file identifier returned by ``query_cpf_files``.

        Returns:
            File content as a list of text lines, or ``None`` on failure.
        """
        payload = {
            **self._auth,
            "action": "data-download",
            "data_type": "CPF_v2",
            "id": str(file_id),
        }
        try:
            resp = requests.post(self._base_url, data=payload, timeout=30)
            resp.raise_for_status()
            return resp.json() if resp.text else None
        except Exception as exc:
            warnings.warn(f"CPF download failed for id={file_id}: {exc}", stacklevel=2)
            return None


# ---------------------------------------------------------------------------
# File processor (download + cache + patch)
# ---------------------------------------------------------------------------

class EphemerisFileProcessor:
    """Download, cache, and pre-process remote ephemeris files.

    This helper centralizes:

    - SP3 FTP download + decompression + header patching
    - CPF API download + local caching

    Args:
        api_client: EDC API client used for CPF downloads.
        ftp_client: Optional EDC FTP client used for SP3 downloads.
        cache_dir: Cache directory for downloaded and processed files.
    """

    def __init__(
        self,
        api_client: EDCApiClient,
        ftp_client: EDCFtpClient | None = None,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.api = api_client
        self.ftp = ftp_client or EDCFtpClient()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_sp3_file(self, record: SP3FileRecord) -> Path | None:
        """Download, cache, and prepare an SP3 file.

        Behavior:

        - returns cached decompressed file if already present
        - downloads ``.sp3.gz`` via FTP otherwise
        - decompresses to ``.sp3``
        - patches known ``V00`` header variants for Orekit compatibility

        Args:
            record: SP3 file metadata record.

        Returns:
            Local path to the prepared SP3 file, or ``None`` if download fails.
        """
        local_filename = record.filename.replace(".sp3.gz", ".sp3")
        local_path = self.cache_dir / local_filename

        if local_path.exists():
            return local_path

        gz_path = self.cache_dir / record.filename
        if not self.ftp.download_file(record.ftp_url, gz_path):
            return None

        # Decompress
        with gzip.open(gz_path, "rt") as f_in, open(local_path, "w") as f_out:
            f_out.write(f_in.read())
        gz_path.unlink(missing_ok=True)

        # Patch SP3 v00 headers (#c -> #d) for Orekit compatibility
        if "V00" in local_filename.upper():
            self._patch_sp3_header(local_path)

        return local_path

    def get_cpf_file(self, record: CPFFileRecord) -> Path | None:
        """Download a CPF file via EDC API and cache it locally.

        Args:
            record: CPF file metadata record.

        Returns:
            Local path to the cached CPF file, or ``None`` if download fails.
        """
        local_path = self.cache_dir / f"cpf_{record.file_id}.cpf"

        if local_path.exists():
            return local_path

        content = self.api.download_cpf_content(record.file_id)
        if not content:
            return None

        local_path.write_text("\n".join(content))
        return local_path

    @staticmethod
    def _patch_sp3_header(file_path: Path) -> None:
        """Patch SP3 v00 header markers in-place for Orekit compatibility.

        Args:
            file_path: Local SP3 file path to patch in-place.
        """
        with open(file_path, "r+") as f:
            lines = f.readlines()
            if lines and lines[0].startswith("#c"):
                lines[0] = "#d" + lines[0][2:]
                f.seek(0)
                f.writelines(lines)
                f.truncate()


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def load_edc_credentials(secrets_path: str | Path) -> tuple[str, str]:
    """Load EDC credentials from a ``secrets.ini`` file.

    Expected file structure:

    - section: ``[credentials]``
    - keys: ``edc_username``, ``edc_password``

    Args:
        secrets_path: Path to a ``secrets.ini`` file.

    Returns:
        A ``(username, password)`` tuple.

    Raises:
        FileNotFoundError: If the secrets file does not exist.
        KeyError: If the credentials section or required keys are missing.
    """
    config = configparser.ConfigParser()
    path = Path(secrets_path)
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")

    config.read(path)
    if "credentials" not in config:
        raise KeyError("Missing [credentials] section in secrets file.")

    username = config["credentials"].get("edc_username", "").strip()
    password = config["credentials"].get("edc_password", "").strip()
    if not username or not password:
        raise KeyError("Missing edc_username or edc_password in secrets file.")

    return username, password
