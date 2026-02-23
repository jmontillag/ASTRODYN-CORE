#pragma once

#include <cstddef>

namespace astrodyn_core {
namespace geqoe {

/**
 * Solve the generalised Kepler equation via Newton-Raphson.
 *
 * Finds K such that  Lr - K - p1*cos(K) + p2*sin(K) = 0  for each of
 * the N elements.  All input/output arrays must be of length N and
 * allocated by the caller.
 *
 * The iteration updates ALL elements on every pass and converges when
 * every |delta_K| < tol (matching the vectorised Python behaviour).
 *
 * @param Lr       (N,) true longitude values
 * @param p1       (N,) eccentricity-like parameter
 * @param p2       (N,) eccentricity-like parameter
 * @param K_out    (N,) output eccentric longitude (caller-allocated)
 * @param N        number of elements
 * @param tol      convergence tolerance (default 1e-14)
 * @param max_iter maximum Newton-Raphson iterations (default 1000)
 */
void solve_kep_gen(
    const double* Lr,
    const double* p1,
    const double* p2,
    double* K_out,
    size_t N,
    double tol = 1e-14,
    int max_iter = 1000
);

/**
 * Single-element Kepler solve (convenience for inlining in geqoe2rv).
 *
 * @param Lr  true longitude
 * @param p1  eccentricity-like parameter
 * @param p2  eccentricity-like parameter
 * @param tol convergence tolerance
 * @param max_iter maximum iterations
 * @return K  solved eccentric longitude
 */
double solve_kep_gen_scalar(
    double Lr, double p1, double p2,
    double tol = 1e-14, int max_iter = 1000
);

} // namespace geqoe
} // namespace astrodyn_core
