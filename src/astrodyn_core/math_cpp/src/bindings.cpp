#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "math_utils.hpp"

namespace py = pybind11;
using namespace astrodyn_core::math;

// -----------------------------------------------------------------------------
// WRAPPERS: These functions handle Python <-> C++ translation.
// They unpack numpy arrays, allocate numpy results, and call the pure core.
// -----------------------------------------------------------------------------

py::object py_derivatives_of_inverse(py::array_t<double> a_vector, bool do_one) {
    py::buffer_info buf = a_vector.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    size_t n = buf.size;

    if (n == 0) return py::cast(0.0);

    if (do_one) {
        double result;
        compute_derivatives_of_inverse(ptr, &result, n, do_one);
        return py::cast(result);
    }

    py::array_t<double> result(n);
    double* res_ptr = static_cast<double*>(result.request().ptr);
    compute_derivatives_of_inverse(ptr, res_ptr, n, do_one);
    return result;
}

py::object py_derivatives_of_inverse_wrt_param(py::array_t<double> a_vector, py::array_t<double> a_d_vector, bool do_one) {
    py::buffer_info buf_a = a_vector.request();
    py::buffer_info buf_ad = a_d_vector.request();
    
    if (buf_a.size != buf_ad.size) throw std::runtime_error("Array sizes must match");
    
    const double* ptr = static_cast<const double*>(buf_a.ptr);
    const double* ptr_d = static_cast<const double*>(buf_ad.ptr);
    size_t n = buf_a.size;

    if (n == 0) return py::cast(0.0);

    if (do_one) {
        double result;
        compute_derivatives_of_inverse_wrt_param(ptr, ptr_d, &result, n, do_one);
        return py::cast(result);
    }

    py::array_t<double> result(n);
    double* res_ptr = static_cast<double*>(result.request().ptr);
    compute_derivatives_of_inverse_wrt_param(ptr, ptr_d, res_ptr, n, do_one);
    return result;
}

py::object py_derivatives_of_product(py::array_t<double> a_vector, bool do_one) {
    py::buffer_info buf = a_vector.request();
    const double* ptr = static_cast<const double*>(buf.ptr);
    size_t m = buf.size;

    if (m <= 1) {
        if (do_one) return py::cast(0.0);
        else return py::array_t<double>(0);
    }

    size_t n = m - 1;

    if (do_one) {
        double result;
        compute_derivatives_of_product(ptr, &result, m, do_one);
        return py::cast(result);
    }

    py::array_t<double> result(n);
    double* res_ptr = static_cast<double*>(result.request().ptr);
    compute_derivatives_of_product(ptr, res_ptr, m, do_one);
    return result;
}

py::object py_derivatives_of_product_wrt_param(py::array_t<double> a_vector, py::array_t<double> a_d_vector, bool do_one) {
    py::buffer_info buf_a = a_vector.request();
    py::buffer_info buf_ad = a_d_vector.request();
    
    if (buf_a.size != buf_ad.size) throw std::runtime_error("Array sizes must match");

    const double* ptr = static_cast<const double*>(buf_a.ptr);
    const double* ptr_d = static_cast<const double*>(buf_ad.ptr);
    size_t m = buf_a.size;

    if (m <= 1) {
        if (do_one) return py::cast(0.0);
        else return py::array_t<double>(0);
    }
    
    size_t n = m - 1;

    if (do_one) {
        double result;
        compute_derivatives_of_product_wrt_param(ptr, ptr_d, &result, m, do_one);
        return py::cast(result);
    }

    py::array_t<double> result(n);
    double* res_ptr = static_cast<double*>(result.request().ptr);
    compute_derivatives_of_product_wrt_param(ptr, ptr_d, res_ptr, m, do_one);
    return result;
}

// -----------------------------------------------------------------------------
// MODULE DEFINITION
// -----------------------------------------------------------------------------
PYBIND11_MODULE(math_utils_cpp, m) {
    m.doc() = "C++ accelerated mathematical utility functions";
    m.def("derivatives_of_inverse", &py_derivatives_of_inverse, "Computes the derivatives of the inverse of a scalar function", py::arg("a_vector"), py::arg("do_one") = false);
    m.def("derivatives_of_inverse_wrt_param", &py_derivatives_of_inverse_wrt_param, "Computes the partial derivatives of the derivatives of f(t) = 1/a(t)", py::arg("a_vector"), py::arg("a_d_vector"), py::arg("do_one") = false);
    m.def("derivatives_of_product", &py_derivatives_of_product, "Computes derivatives of the product f(t) = a(t) * a'(t)", py::arg("a_vector"), py::arg("do_one") = false);
    m.def("derivatives_of_product_wrt_param", &py_derivatives_of_product_wrt_param, "Computes the partial derivatives of the derivatives of f(t) = a(t) * a'(t)", py::arg("a_vector"), py::arg("a_d_vector"), py::arg("do_one") = false);
}