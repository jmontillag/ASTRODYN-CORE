#pragma once

#include <cstddef>
#include <vector>

#include "propagator_core.hpp"
#include "taylor_order_1.hpp"

namespace astrodyn_core {
namespace geqoe {

struct Order2Coefficients {
    // Second-order EOM values
    double q1pp_0;
    double q2pp_0;
    double p1pp_0;
    double p2pp_0;
    double Lrpp_0;

    // Second-order EOM partials wrt initial conditions
    double q1p2_nu;
    double q1p2_Lr;
    double q1p2_q1;
    double q1p2_q2;
    double q1p2_p1;
    double q1p2_p2;

    double q2p2_nu;
    double q2p2_Lr;
    double q2p2_q1;
    double q2p2_q2;
    double q2p2_p1;
    double q2p2_p2;

    double p1p2_nu;
    double p1p2_Lr;
    double p1p2_q1;
    double p1p2_q2;
    double p1p2_p1;
    double p1p2_p2;

    double p2p2_nu;
    double p2p2_Lr;
    double p2p2_q1;
    double p2p2_q2;
    double p2p2_p1;
    double p2p2_p2;

    double Lrp2_nu;
    double Lrp2_Lr;
    double Lrp2_q1;
    double Lrp2_q2;
    double Lrp2_p1;
    double Lrp2_p2;

    // Updated fic (overwritten by derivatives_of_inverse(c_vector))
    double fic_updated;

    double map_components_col1[6];
};

struct Order2EvaluationScratch {
    std::vector<double> nu_nu;

    std::vector<double> q1_nu;
    std::vector<double> q1_q1;
    std::vector<double> q1_q2;
    std::vector<double> q1_p1;
    std::vector<double> q1_p2;
    std::vector<double> q1_Lr;

    std::vector<double> q2_nu;
    std::vector<double> q2_q1;
    std::vector<double> q2_q2;
    std::vector<double> q2_p1;
    std::vector<double> q2_p2;
    std::vector<double> q2_Lr;

    std::vector<double> p1_nu;
    std::vector<double> p1_q1;
    std::vector<double> p1_q2;
    std::vector<double> p1_p1;
    std::vector<double> p1_p2;
    std::vector<double> p1_Lr;

    std::vector<double> p2_nu;
    std::vector<double> p2_q1;
    std::vector<double> p2_q2;
    std::vector<double> p2_p1;
    std::vector<double> p2_p2;
    std::vector<double> p2_Lr;

    std::vector<double> Lr_nu;
    std::vector<double> Lr_q1;
    std::vector<double> Lr_q2;
    std::vector<double> Lr_p1;
    std::vector<double> Lr_p2;
    std::vector<double> Lr_Lr;

    void resize(std::size_t M);
    StmAccumulatorView view() const;
};

void compute_coefficients_2(
    const double* y0,
    const PropagationConstants& constants,
    Order2Coefficients& out,
    Order1Coefficients& out1
);

void evaluate_order_2(
    const Order1Coefficients& coeffs1,
    const Order2Coefficients& coeffs2,
    const double* dt_norm,
    std::size_t M,
    double* y_prop,
    Order2EvaluationScratch& scratch
);

} // namespace geqoe
} // namespace astrodyn_core
