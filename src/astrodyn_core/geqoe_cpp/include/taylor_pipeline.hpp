#pragma once

#include <cstddef>
#include <memory>

#include "propagator_core.hpp"
#include "taylor_order_1.hpp"
#include "taylor_order_2.hpp"
#include "taylor_order_3.hpp"
#include "taylor_order_4.hpp"

namespace astrodyn_core {
namespace geqoe {

struct PreparedTaylorCoefficients {
    PropagationConstants constants;
    int order;
    double initial_geqoe[6];
    double map_components[24];
    Order1Coefficients order1;
    Order1Intermediates inter1;
    Order2Coefficients order2;
    Order2Intermediates inter2;
    Order3Coefficients order3;
    Order3Intermediates inter3;
    Order4Coefficients order4;

    PreparedTaylorCoefficients();
};

std::shared_ptr<PreparedTaylorCoefficients> prepare_taylor_coefficients_cpp(
    const double* y0,
    double j2,
    double re,
    double mu,
    int order
);

void evaluate_taylor_cpp(
    const PreparedTaylorCoefficients& coeffs,
    const double* dt_seconds,
    std::size_t M,
    double* y_prop,
    double* y_y0,
    double* map_components
);

std::shared_ptr<PreparedTaylorCoefficients> prepare_cart_coefficients_cpp(
    const double* y0_cart,
    double j2,
    double re,
    double mu,
    int order,
    double* peq_py_0
);

void evaluate_cart_taylor_cpp(
    const PreparedTaylorCoefficients& coeffs,
    const double* peq_py_0,
    const double* tspan,
    std::size_t M,
    double* y_out,
    double* dy_dy0
);

} // namespace geqoe
} // namespace astrodyn_core
