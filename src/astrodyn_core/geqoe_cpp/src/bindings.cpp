#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "kepler_solver.hpp"
#include "conversions.hpp"
#include "jacobians.hpp"

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
}
