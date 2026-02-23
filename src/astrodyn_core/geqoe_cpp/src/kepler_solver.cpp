#include "kepler_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace astrodyn_core {
namespace geqoe {

// ---------------------------------------------------------------------------
// Vectorised solver -- matches the Python np.all() convergence semantics
// ---------------------------------------------------------------------------
void solve_kep_gen(
    const double* Lr,
    const double* p1,
    const double* p2,
    double* K_out,
    size_t N,
    double tol,
    int max_iter
) {
    // Initialise K = Lr
    for (size_t i = 0; i < N; ++i) {
        K_out[i] = Lr[i];
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        bool all_converged = true;

        for (size_t i = 0; i < N; ++i) {
            double sinK = std::sin(K_out[i]);
            double cosK = std::cos(K_out[i]);

            // Residual:  f = Lr - K - p1*cos(K) + p2*sin(K)
            double f  = Lr[i] - K_out[i] - p1[i] * cosK + p2[i] * sinK;
            // Derivative: f' = -1 + p1*sin(K) + p2*cos(K)
            double fp = -1.0 + p1[i] * sinK + p2[i] * cosK;

            // Guard against near-zero denominator (matches Python clip)
            if (std::abs(fp) < 1e-15) fp = 1e-15;

            double delta = -f / fp;
            K_out[i] += delta;

            if (std::abs(delta) >= tol) all_converged = false;
        }

        if (all_converged) return;
    }

    throw std::runtime_error(
        "Newton-Raphson solver failed to converge within "
        + std::to_string(max_iter) + " iterations in solve_kep_gen."
    );
}

// ---------------------------------------------------------------------------
// Scalar convenience for use inside geqoe2rv (one state at a time)
// ---------------------------------------------------------------------------
double solve_kep_gen_scalar(
    double Lr, double p1, double p2,
    double tol, int max_iter
) {
    double K = Lr;

    for (int iter = 0; iter < max_iter; ++iter) {
        double sinK = std::sin(K);
        double cosK = std::cos(K);

        double f  = Lr - K - p1 * cosK + p2 * sinK;
        double fp = -1.0 + p1 * sinK + p2 * cosK;

        if (std::abs(fp) < 1e-15) fp = 1e-15;

        double delta = -f / fp;
        K += delta;

        if (std::abs(delta) < tol) return K;
    }

    throw std::runtime_error(
        "Newton-Raphson solver failed to converge within "
        + std::to_string(max_iter) + " iterations in solve_kep_gen_scalar."
    );
}

} // namespace geqoe
} // namespace astrodyn_core
