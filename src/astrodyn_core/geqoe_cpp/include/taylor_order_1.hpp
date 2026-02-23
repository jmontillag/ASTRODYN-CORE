#pragma once

#include <cstddef>
#include <vector>

#include "propagator_core.hpp"

namespace astrodyn_core {
namespace geqoe {

struct Order1Coefficients {
    double nu_0;
    double q1_0;
    double q2_0;
    double p1_0;
    double p2_0;
    double Lr_0;

    double q1p_0;
    double q2p_0;
    double p1p_0;
    double p2p_0;
    double Lrp_0;

    double q1p_nu;
    double q1p_Lr;
    double q1p_q1;
    double q1p_q2;
    double q1p_p1;
    double q1p_p2;

    double q2p_nu;
    double q2p_Lr;
    double q2p_q1;
    double q2p_q2;
    double q2p_p1;
    double q2p_p2;

    double p1p_nu;
    double p1p_Lr;
    double p1p_q1;
    double p1p_q2;
    double p1p_p1;
    double p1p_p2;

    double p2p_nu;
    double p2p_Lr;
    double p2p_q1;
    double p2p_q2;
    double p2p_p1;
    double p2p_p2;

    double Lrp_nu;
    double Lrp_Lr;
    double Lrp_q1;
    double Lrp_q2;
    double Lrp_p1;
    double Lrp_p2;

    double map_components_col0[6];
};

struct Order1EvaluationScratch {
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

void compute_coefficients_1(
    const double* y0,
    const PropagationConstants& constants,
    Order1Coefficients& out
);

void evaluate_order_1(
    const Order1Coefficients& coeffs,
    const double* dt_norm,
    std::size_t M,
    double* y_prop,
    Order1EvaluationScratch& scratch
);

} // namespace geqoe
} // namespace astrodyn_core
