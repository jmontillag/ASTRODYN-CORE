"""Physical constants for GEqOE propagation.

Values from Baù et al. (2021), Eq. 59.  Units: km, s.
"""

MU = 398600.4354360959        # km^3/s^2
J2 = 1.08262617385222e-3      # dimensionless
RE = 6378.1366                # km
A_J2 = MU * J2 * RE**2 / 2   # km^5/s^2  — convenience constant for J2 potential
