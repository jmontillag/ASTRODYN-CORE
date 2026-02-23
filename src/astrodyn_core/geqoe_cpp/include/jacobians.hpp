#pragma once

#include <cstddef>

namespace astrodyn_core {
namespace geqoe {

/**
 * Compute the Jacobian d(Eq)/d(Y) -- Cartesian to GEqOE.
 *
 * For N states, computes N individual 6x6 Jacobians stored in row-major
 * order in a contiguous (N*36) buffer.
 *
 * @param y_in    (N*6) Cartesian states [rx,ry,rz,vx,vy,vz] in SI
 * @param jac_out (N*36) Jacobian matrices, row-major 6x6 per state
 * @param N       number of states
 * @param J2      J2 coefficient
 * @param Re      equatorial radius (m)
 * @param mu      gravitational parameter (m^3/s^2)
 */
void get_pEqpY(
    const double* y_in,
    double* jac_out,
    size_t N,
    double J2, double Re, double mu
);

/**
 * Compute the Jacobian d(Y)/d(Eq) -- GEqOE to Cartesian.
 *
 * @param eq_in   (N*6) GEqOE states [nu,q1,q2,p1,p2,Lr]
 * @param jac_out (N*36) Jacobian matrices, row-major 6x6 per state
 * @param N       number of states
 * @param J2      J2 coefficient
 * @param Re      equatorial radius (m)
 * @param mu      gravitational parameter (m^3/s^2)
 */
void get_pYpEq(
    const double* eq_in,
    double* jac_out,
    size_t N,
    double J2, double Re, double mu
);

} // namespace geqoe
} // namespace astrodyn_core
