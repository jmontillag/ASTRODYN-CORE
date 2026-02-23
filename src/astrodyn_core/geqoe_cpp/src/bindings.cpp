#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>

#include "kepler_solver.hpp"
#include "conversions.hpp"
#include "jacobians.hpp"
#include "taylor_pipeline.hpp"

namespace py = pybind11;
using namespace astrodyn_core::geqoe;

// =============================================================================
// WRAPPERS: Python <-> C++ translation layer
// =============================================================================

// ---------------------------------------------------------------------------
// solve_kep_gen
// ---------------------------------------------------------------------------
py::array_t<double> py_solve_kep_gen(
    py::array_t<double, py::array::c_style | py::array::forcecast> Lr_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> p1_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> p2_arr,
    double tol,
    int max_iter
) {
    auto buf_Lr = Lr_arr.request();
    auto buf_p1 = p1_arr.request();
    auto buf_p2 = p2_arr.request();

    size_t N = static_cast<size_t>(buf_Lr.size);
    if (static_cast<size_t>(buf_p1.size) != N || static_cast<size_t>(buf_p2.size) != N) {
        throw std::runtime_error("Lr, p1, p2 must have the same length");
    }

    py::array_t<double> K_out(N);
    solve_kep_gen(
        static_cast<const double*>(buf_Lr.ptr),
        static_cast<const double*>(buf_p1.ptr),
        static_cast<const double*>(buf_p2.ptr),
        static_cast<double*>(K_out.request().ptr),
        N, tol, max_iter
    );
    return K_out;
}

// ---------------------------------------------------------------------------
// rv2geqoe  --  returns tuple of 6 arrays (nu, q1, q2, p1, p2, Lr)
// ---------------------------------------------------------------------------
py::tuple py_rv2geqoe(
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr,
    double J2, double Re, double mu
) {
    auto buf = y_arr.request();

    // Accept (6,) or (N, 6)
    size_t N;
    const double* ptr = static_cast<const double*>(buf.ptr);

    if (buf.ndim == 1) {
        if (buf.shape[0] != 6)
            throw std::runtime_error("1-D input must have exactly 6 elements");
        N = 1;
    } else if (buf.ndim == 2) {
        if (buf.shape[1] != 6)
            throw std::runtime_error("2-D input must have shape (N, 6)");
        N = static_cast<size_t>(buf.shape[0]);
    } else {
        throw std::runtime_error("Input must be 1-D (6,) or 2-D (N, 6)");
    }

    // Allocate contiguous (N, 6) buffer for the C++ core
    py::array_t<double> eq_buf({static_cast<py::ssize_t>(N), py::ssize_t(6)});
    double* eq_ptr = static_cast<double*>(eq_buf.request().ptr);

    rv2geqoe(ptr, eq_ptr, N, J2, Re, mu);

    // Split into 6 separate 1-D arrays to match the Python API
    py::array_t<double> nu_out(N), q1_out(N), q2_out(N),
                        p1_out(N), p2_out(N), Lr_out(N);
    double* nu_p  = static_cast<double*>(nu_out.request().ptr);
    double* q1_p  = static_cast<double*>(q1_out.request().ptr);
    double* q2_p  = static_cast<double*>(q2_out.request().ptr);
    double* p1_p  = static_cast<double*>(p1_out.request().ptr);
    double* p2_p  = static_cast<double*>(p2_out.request().ptr);
    double* Lr_p  = static_cast<double*>(Lr_out.request().ptr);

    for (size_t i = 0; i < N; ++i) {
        const double* row = eq_ptr + i * 6;
        nu_p[i] = row[0];
        q1_p[i] = row[1];
        q2_p[i] = row[2];
        p1_p[i] = row[3];
        p2_p[i] = row[4];
        Lr_p[i] = row[5];
    }

    return py::make_tuple(nu_out, q1_out, q2_out, p1_out, p2_out, Lr_out);
}

// ---------------------------------------------------------------------------
// geqoe2rv  --  returns tuple of two (N, 3) arrays (rv, rpv)
// ---------------------------------------------------------------------------
py::tuple py_geqoe2rv(
    py::array_t<double, py::array::c_style | py::array::forcecast> eq_arr,
    double J2, double Re, double mu
) {
    auto buf = eq_arr.request();

    size_t N;
    const double* ptr = static_cast<const double*>(buf.ptr);

    if (buf.ndim == 1) {
        if (buf.shape[0] != 6)
            throw std::runtime_error("1-D input must have exactly 6 elements");
        N = 1;
    } else if (buf.ndim == 2) {
        if (buf.shape[1] != 6)
            throw std::runtime_error("2-D input must have shape (N, 6)");
        N = static_cast<size_t>(buf.shape[0]);
    } else {
        throw std::runtime_error("Input must be 1-D (6,) or 2-D (N, 6)");
    }

    // Allocate output arrays
    py::array_t<double> rv_out({static_cast<py::ssize_t>(N), py::ssize_t(3)});
    py::array_t<double> rpv_out({static_cast<py::ssize_t>(N), py::ssize_t(3)});

    double* rv_ptr  = static_cast<double*>(rv_out.request().ptr);
    double* rpv_ptr = static_cast<double*>(rpv_out.request().ptr);

    geqoe2rv(ptr, rv_ptr, rpv_ptr, N, J2, Re, mu);

    return py::make_tuple(rv_out, rpv_out);
}

// ---------------------------------------------------------------------------
// Helper to parse (6,) or (N, 6) input and return N + pointer
// ---------------------------------------------------------------------------
static std::pair<size_t, const double*> parse_Nx6(
    const py::buffer_info& buf, const char* name
) {
    size_t N;
    if (buf.ndim == 1) {
        if (buf.shape[0] != 6)
            throw std::runtime_error(std::string(name) + ": 1-D input must have 6 elements");
        N = 1;
    } else if (buf.ndim == 2) {
        if (buf.shape[1] != 6)
            throw std::runtime_error(std::string(name) + ": 2-D input must be (N, 6)");
        N = static_cast<size_t>(buf.shape[0]);
    } else {
        throw std::runtime_error(std::string(name) + ": input must be 1-D or 2-D");
    }
    return {N, static_cast<const double*>(buf.ptr)};
}

static std::shared_ptr<PreparedTaylorCoefficients> parse_coeff_capsule(py::object coeffs_obj) {
    py::capsule cap = py::reinterpret_borrow<py::capsule>(coeffs_obj);
    if (std::string(cap.name()) != "PreparedTaylorCoefficients") {
        throw std::runtime_error("Invalid coefficient object capsule.");
    }
    auto* holder = static_cast<std::shared_ptr<PreparedTaylorCoefficients>*>(cap.get_pointer());
    if (holder == nullptr || !(*holder)) {
        throw std::runtime_error("Coefficient capsule is null.");
    }
    return *holder;
}

static py::capsule make_coeff_capsule(std::shared_ptr<PreparedTaylorCoefficients> coeffs) {
    auto* holder = new std::shared_ptr<PreparedTaylorCoefficients>(std::move(coeffs));
    return py::capsule(holder, "PreparedTaylorCoefficients", [](void* p) {
        delete static_cast<std::shared_ptr<PreparedTaylorCoefficients>*>(p);
    });
}

// ---------------------------------------------------------------------------
// get_pEqpY  --  returns (N, 6, 6)
// ---------------------------------------------------------------------------
py::array_t<double> py_get_pEqpY(
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr,
    double J2, double Re, double mu
) {
    auto buf = y_arr.request();
    auto [N, ptr] = parse_Nx6(buf, "get_pEqpY");

    py::array_t<double> jac_out(
        {static_cast<py::ssize_t>(N), py::ssize_t(6), py::ssize_t(6)});
    double* jac_ptr = static_cast<double*>(jac_out.request().ptr);

    get_pEqpY(ptr, jac_ptr, N, J2, Re, mu);
    return jac_out;
}

// ---------------------------------------------------------------------------
// get_pYpEq  --  returns (N, 6, 6)
// ---------------------------------------------------------------------------
py::array_t<double> py_get_pYpEq(
    py::array_t<double, py::array::c_style | py::array::forcecast> eq_arr,
    double J2, double Re, double mu
) {
    auto buf = eq_arr.request();
    auto [N, ptr] = parse_Nx6(buf, "get_pYpEq");

    py::array_t<double> jac_out(
        {static_cast<py::ssize_t>(N), py::ssize_t(6), py::ssize_t(6)});
    double* jac_ptr = static_cast<double*>(jac_out.request().ptr);

    get_pYpEq(ptr, jac_ptr, N, J2, Re, mu);
    return jac_out;
}

py::capsule py_prepare_taylor_coefficients_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> y0_arr,
    double J2, double Re, double mu,
    int order
) {
    auto buf = y0_arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 6) {
        throw std::runtime_error("y0 must be a 1-D 6-element GEqOE state vector.");
    }

    const double* y0 = static_cast<const double*>(buf.ptr);
    auto coeffs = prepare_taylor_coefficients_cpp(y0, J2, Re, mu, order);
    return make_coeff_capsule(std::move(coeffs));
}

py::tuple py_evaluate_taylor_cpp(
    py::object coeffs_obj,
    py::array_t<double, py::array::c_style | py::array::forcecast> dt_arr
) {
    auto coeffs = parse_coeff_capsule(coeffs_obj);

    auto dt_buf = dt_arr.request();
    if (dt_buf.ndim != 1) {
        throw std::runtime_error("dt must be a 1-D array.");
    }
    size_t M = static_cast<size_t>(dt_buf.shape[0]);
    const double* dt_ptr = static_cast<const double*>(dt_buf.ptr);

    py::array_t<double> y_prop({static_cast<py::ssize_t>(M), py::ssize_t(6)});
    py::array_t<double> y_y0({py::ssize_t(6), py::ssize_t(6), static_cast<py::ssize_t>(M)});
    py::array_t<double> map_components({py::ssize_t(6), py::ssize_t(coeffs->order)});

    evaluate_taylor_cpp(
        *coeffs,
        dt_ptr,
        M,
        static_cast<double*>(y_prop.request().ptr),
        static_cast<double*>(y_y0.request().ptr),
        static_cast<double*>(map_components.request().ptr)
    );

    return py::make_tuple(y_prop, y_y0, map_components);
}

py::tuple py_prepare_cart_coefficients_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> y0_cart_arr,
    double J2, double Re, double mu,
    int order
) {
    auto buf = y0_cart_arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 6) {
        throw std::runtime_error("y0_cart must be a 1-D 6-element Cartesian state vector.");
    }

    py::array_t<double> peq_py_0({py::ssize_t(6), py::ssize_t(6)});
    auto coeffs = prepare_cart_coefficients_cpp(
        static_cast<const double*>(buf.ptr), J2, Re, mu, order,
        static_cast<double*>(peq_py_0.request().ptr)
    );

    return py::make_tuple(make_coeff_capsule(std::move(coeffs)), peq_py_0);
}

py::tuple py_evaluate_cart_taylor_cpp(
    py::object coeffs_obj,
    py::array_t<double, py::array::c_style | py::array::forcecast> peq_py_0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> tspan_arr
) {
    auto coeffs = parse_coeff_capsule(coeffs_obj);

    auto peq_buf = peq_py_0_arr.request();
    if (peq_buf.ndim != 2 || peq_buf.shape[0] != 6 || peq_buf.shape[1] != 6) {
        throw std::runtime_error("peq_py_0 must be shape (6, 6).");
    }

    auto t_buf = tspan_arr.request();
    if (t_buf.ndim != 1) {
        throw std::runtime_error("tspan must be a 1-D array.");
    }
    size_t M = static_cast<size_t>(t_buf.shape[0]);

    py::array_t<double> y_out({static_cast<py::ssize_t>(M), py::ssize_t(6)});
    py::array_t<double> dy_dy0({py::ssize_t(6), py::ssize_t(6), static_cast<py::ssize_t>(M)});

    evaluate_cart_taylor_cpp(
        *coeffs,
        static_cast<const double*>(peq_buf.ptr),
        static_cast<const double*>(t_buf.ptr),
        M,
        static_cast<double*>(y_out.request().ptr),
        static_cast<double*>(dy_dy0.request().ptr)
    );

    return py::make_tuple(y_out, dy_dy0);
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================
PYBIND11_MODULE(geqoe_utils_cpp, m) {
    m.doc() = "C++ accelerated GEqOE conversion and utility functions";

    m.def("solve_kep_gen", &py_solve_kep_gen,
          "Solve the generalised Kepler equation (vectorised Newton-Raphson)",
          py::arg("Lr"), py::arg("p1"), py::arg("p2"),
          py::arg("tol") = 1e-14, py::arg("max_iter") = 1000);

    m.def("rv2geqoe", &py_rv2geqoe,
          "Convert Cartesian states to GEqOE (returns tuple of 6 arrays)",
          py::arg("y"), py::arg("J2"), py::arg("Re"), py::arg("mu"));

    m.def("geqoe2rv", &py_geqoe2rv,
          "Convert GEqOE states to Cartesian (returns tuple of (N,3) rv, rpv)",
          py::arg("y"), py::arg("J2"), py::arg("Re"), py::arg("mu"));

    m.def("get_pEqpY", &py_get_pEqpY,
          "Jacobian d(Eq)/d(Y): Cartesian -> GEqOE (returns (N,6,6))",
          py::arg("y"), py::arg("J2"), py::arg("Re"), py::arg("mu"));

    m.def("get_pYpEq", &py_get_pYpEq,
          "Jacobian d(Y)/d(Eq): GEqOE -> Cartesian (returns (N,6,6))",
          py::arg("y"), py::arg("J2"), py::arg("Re"), py::arg("mu"));

        m.def("prepare_taylor_coefficients_cpp", &py_prepare_taylor_coefficients_cpp,
            "Prepare staged GEqOE Taylor coefficients (C++, order-1 currently)",
            py::arg("y0"), py::arg("J2"), py::arg("Re"), py::arg("mu"), py::arg("order") = 1);

        m.def("evaluate_taylor_cpp", &py_evaluate_taylor_cpp,
            "Evaluate staged GEqOE Taylor polynomial (C++, order-1 currently)",
            py::arg("coeffs"), py::arg("dt"));

        m.def("prepare_cart_coefficients_cpp", &py_prepare_cart_coefficients_cpp,
            "Prepare staged Cartesian Taylor coefficients (C++, order-1 currently)",
            py::arg("y0_cart"), py::arg("J2"), py::arg("Re"), py::arg("mu"), py::arg("order") = 1);

        m.def("evaluate_cart_taylor_cpp", &py_evaluate_cart_taylor_cpp,
            "Evaluate staged Cartesian Taylor propagation (C++, order-1 currently)",
            py::arg("coeffs"), py::arg("peq_py_0"), py::arg("tspan"));
}
