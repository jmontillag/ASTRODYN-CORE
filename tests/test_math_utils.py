import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
from pathlib import Path

# Add the root directory to path so we can import temp_mosaic_modules
sys.path.append(str(Path(__file__).parent.parent))

from temp_mosaic_modules.extra_utils.math_utils import (
    derivatives_of_inverse as py_inv,
    derivatives_of_inverse_wrt_param as py_inv_param,
    derivatives_of_product as py_prod,
    derivatives_of_product_wrt_param as py_prod_param
)

from astrodyn_core.math_cpp import (
    derivatives_of_inverse as cpp_inv,
    derivatives_of_inverse_wrt_param as cpp_inv_param,
    derivatives_of_product as cpp_prod,
    derivatives_of_product_wrt_param as cpp_prod_param
)

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("do_one", [True, False])
def test_derivatives_of_inverse_parity(n, do_one):
    np.random.seed(42 + n)
    a_vector = np.random.uniform(0.5, 5.0, size=n)
    
    py_result = py_inv(a_vector, do_one=do_one)
    cpp_result = cpp_inv(a_vector, do_one=do_one)
    
    assert_allclose(py_result, cpp_result, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("do_one", [True, False])
def test_derivatives_of_inverse_wrt_param_parity(n, do_one):
    np.random.seed(142 + n)
    a_vector = np.random.uniform(0.5, 5.0, size=n)
    a_d_vector = np.random.uniform(-2.0, 2.0, size=n)
    
    py_result = py_inv_param(a_vector, a_d_vector, do_one=do_one)
    cpp_result = cpp_inv_param(a_vector, a_d_vector, do_one=do_one)
    
    assert_allclose(py_result, cpp_result, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("do_one", [True, False])
def test_derivatives_of_product_parity(n, do_one):
    np.random.seed(242 + n)
    a_vector = np.random.uniform(0.5, 5.0, size=n)
    
    py_result = py_prod(a_vector, do_one=do_one)
    cpp_result = cpp_prod(a_vector, do_one=do_one)
    
    assert_allclose(py_result, cpp_result, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("do_one", [True, False])
def test_derivatives_of_product_wrt_param_parity(n, do_one):
    np.random.seed(342 + n)
    a_vector = np.random.uniform(0.5, 5.0, size=n)
    a_d_vector = np.random.uniform(-2.0, 2.0, size=n)
    
    py_result = py_prod_param(a_vector, a_d_vector, do_one=do_one)
    cpp_result = cpp_prod_param(a_vector, a_d_vector, do_one=do_one)
    
    assert_allclose(py_result, cpp_result, rtol=1e-14, atol=1e-14)
