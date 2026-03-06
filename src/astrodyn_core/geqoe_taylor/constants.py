"""Physical constants for GEqOE propagation.

Values from Baù et al. (2021), Eq. 59.  Units: km, s.
"""

MU = 398600.4354360959        # km^3/s^2
J2 = 1.08262617385222e-3      # dimensionless
RE = 6378.1366                # km
A_J2 = MU * J2 * RE**2 / 2   # km^5/s^2  — convenience constant for J2 potential

# Standard zonal harmonic coefficients (EGM2008, unnormalized)
J3 = -2.53265648533224e-6
J4 = -1.61989759991697e-6
J5 = -2.27296082868698e-7
J6 = 5.40681239107085e-7

# Third-body gravitational parameters (km^3/s^2)
GM_SUN = 1.32712440041279419e11
GM_MOON = 4902.800066

# Astronomical unit (km)
AU_KM = 149597870.7

# J2000 obliquity of the ecliptic (radians)
import math as _math
OBLIQUITY_J2000 = _math.radians(23.439291111)
COS_OBLIQUITY = _math.cos(OBLIQUITY_J2000)
SIN_OBLIQUITY = _math.sin(OBLIQUITY_J2000)

# J2000 epoch Julian date
JD_J2000 = 2451545.0

# Time unit conversions
SECONDS_PER_DAY = 86400.0
DAYS_PER_JULIAN_CENTURY = 36525.0
DAYS_PER_JULIAN_MILLENNIUM = 365250.0

# Propulsion / unit conversion constants
G0_MPS2 = 9.80665
METERS_PER_KILOMETER = 1000.0
