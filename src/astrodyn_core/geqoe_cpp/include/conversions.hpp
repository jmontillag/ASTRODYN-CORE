#pragma once

#include <cstddef>

namespace astrodyn_core {
namespace geqoe {

/**
 * Convert N Cartesian states to Generalised Equinoctial Orbital Elements.
 *
 * Pure C++ core -- operates on raw double pointers, no Python knowledge.
 * All arrays are row-major and must be allocated by the caller.
 *
 * @param y_in   (N*6) Cartesian states [rx,ry,rz,vx,vy,vz] in SI units (m, m/s)
 * @param eq_out (N*6) GEqOE states [nu,q1,q2,p1,p2,Lr]
 *               nu in rad/s; q1,q2,p1,p2 dimensionless; Lr in rad
 * @param N      number of states
 * @param J2     J2 gravitational coefficient (positive, dimensionless)
 * @param Re     equatorial radius (m)
 * @param mu     gravitational parameter GM (m^3/s^2)
 */
void rv2geqoe(
    const double* y_in,
    double* eq_out,
    size_t N,
    double J2, double Re, double mu
);

/**
 * Convert N GEqOE states to Cartesian position and velocity.
 *
 * @param eq_in   (N*6) GEqOE states [nu,q1,q2,p1,p2,Lr]
 * @param rv_out  (N*3) positions in metres
 * @param rpv_out (N*3) velocities in m/s
 * @param N       number of states
 * @param J2      J2 gravitational coefficient
 * @param Re      equatorial radius (m)
 * @param mu      gravitational parameter GM (m^3/s^2)
 */
void geqoe2rv(
    const double* eq_in,
    double* rv_out,
    double* rpv_out,
    size_t N,
    double J2, double Re, double mu
);

} // namespace geqoe
} // namespace astrodyn_core
