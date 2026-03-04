#pragma once

#include <cstddef>
#include <vector>

#include "taylor_order_3.hpp"

namespace astrodyn_core {
namespace geqoe {

struct Order4Intermediates {
    double q1p4_0;
    double q2p4_0;
    double p1p4_0;
    double p2p4_0;
    double Lrp4_0;
    double q1p4_nu;
    double q1p4_Lr;
    double q1p4_q1;
    double q1p4_q2;
    double q1p4_p1;
    double q1p4_p2;
    double q2p4_nu;
    double q2p4_Lr;
    double q2p4_q1;
    double q2p4_q2;
    double q2p4_p1;
    double q2p4_p2;
    double p1p4_nu;
    double p1p4_Lr;
    double p1p4_q1;
    double p1p4_q2;
    double p1p4_p1;
    double p1p4_p2;
    double p2p4_nu;
    double p2p4_Lr;
    double p2p4_q1;
    double p2p4_q2;
    double p2p4_p1;
    double p2p4_p2;
    double Lrp4_nu;
    double Lrp4_Lr;
    double Lrp4_q1;
    double Lrp4_q2;
    double Lrp4_p1;
    double Lrp4_p2;
};

struct Order4Coefficients {
    double q1p4_0;
    double q2p4_0;
    double p1p4_0;
    double p2p4_0;
    double Lrp4_0;
    double q1p4_nu;
    double q1p4_Lr;
    double q1p4_q1;
    double q1p4_q2;
    double q1p4_p1;
    double q1p4_p2;
    double q2p4_nu;
    double q2p4_Lr;
    double q2p4_q1;
    double q2p4_q2;
    double q2p4_p1;
    double q2p4_p2;
    double p1p4_nu;
    double p1p4_Lr;
    double p1p4_q1;
    double p1p4_q2;
    double p1p4_p1;
    double p1p4_p2;
    double p2p4_nu;
    double p2p4_Lr;
    double p2p4_q1;
    double p2p4_q2;
    double p2p4_p1;
    double p2p4_p2;
    double Lrp4_nu;
    double Lrp4_Lr;
    double Lrp4_q1;
    double Lrp4_q2;
    double Lrp4_p1;
    double Lrp4_p2;
    double map_components_col3[6];
};

// Order 4 reuses Order1EvaluationScratch

void compute_coefficients_4(
    const double* y0,
    const PropagationConstants& constants,
    Order1Coefficients& out1,
    Order1Intermediates& inter1,
    Order2Coefficients& out2,
    Order2Intermediates& inter2,
    Order3Coefficients& out3,
    Order3Intermediates& inter3,
    Order4Coefficients& out,
    Order4Intermediates& inter
);

void evaluate_order_4(
    const Order4Coefficients& coeffs,
    const double* dt_norm,
    std::size_t M,
    double* y_prop,
    Order1EvaluationScratch& scratch
);

} // namespace geqoe
} // namespace astrodyn_core
