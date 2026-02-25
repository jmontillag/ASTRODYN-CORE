"""Orekit-based ephemeris file parsers.

Thin wrappers around Orekit's native file parsers.  Each function takes a
file path and returns the parsed Orekit object (SP3, CPF, Oem, Ocm).
"""

from __future__ import annotations

from pathlib import Path


def parse_oem(file_path: str | Path):
    """Parse a CCSDS OEM (Orbit Ephemeris Message) file.

    Args:
        file_path: Path to a local OEM file.

    Returns:
        The parsed Orekit ``Oem`` object.
    """
    from org.orekit.data import DataSource
    from org.orekit.files.ccsds.ndm import ParserBuilder

    return ParserBuilder().buildOemParser().parse(DataSource(str(file_path)))


def parse_ocm(file_path: str | Path):
    """Parse a CCSDS OCM (Orbit Comprehensive Message) file.

    Args:
        file_path: Path to a local OCM file.

    Returns:
        The parsed Orekit ``Ocm`` object.
    """
    from org.orekit.data import DataSource
    from org.orekit.files.ccsds.ndm import ParserBuilder

    return ParserBuilder().buildOcmParser().parse(DataSource(str(file_path)))


def parse_sp3(file_path: str | Path):
    """Parse an IGS SP3 (Standard Product 3) file.

    Args:
        file_path: Path to a local SP3 file.

    Returns:
        The parsed Orekit ``SP3`` object.
    """
    from org.orekit.data import DataSource
    from org.orekit.files.sp3 import SP3Parser

    return SP3Parser().parse(DataSource(str(file_path)))


def parse_cpf(file_path: str | Path):
    """Parse an ILRS CPF (Consolidated Prediction Format) file.

    The CPF parser requires explicit Earth constants, frames, and time scales,
    which are provided using WGS84 Earth ``mu``, IERS 2010 conventions, and UTC.

    Args:
        file_path: Path to a local CPF file.

    Returns:
        The parsed Orekit ``CPF`` object.
    """
    from org.orekit.data import DataSource
    from org.orekit.files.ilrs import CPFParser
    from org.orekit.frames import FramesFactory
    from org.orekit.time import TimeScalesFactory
    from org.orekit.utils import Constants, IERSConventions

    utc = TimeScalesFactory.getUTC()
    mu = Constants.WGS84_EARTH_MU
    iers = IERSConventions.IERS_2010
    frames = FramesFactory.getFrames()

    parser = CPFParser(mu, 4, iers, utc, frames)
    return parser.parse(DataSource(str(file_path)))
