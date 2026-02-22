#!/usr/bin/env python3
"""
ASTRODYN-CORE Basic Math Comparison Example

This script compares the outputs of the original pure-Python math utilities 
with the new pybind11 C++ backend port to demonstrate output parity and performance differences.

Usage:
    python examples/math_cpp_comparison.py
"""

import sys
import timeit
from pathlib import Path

import numpy as np

# Add the project root to sys.path so we can import the temp_mosaic_modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 1. Import original Python implementations
from temp_mosaic_modules.extra_utils.math_utils import (
    derivatives_of_inverse as py_inv,
    derivatives_of_product as py_prod
)

# 2. Import the new C++ compiled implementations
from astrodyn_core.math_cpp import (
    derivatives_of_inverse as cpp_inv,
    derivatives_of_product as cpp_prod
)

def main():
    print("=== ASTRODYN-CORE C++ vs Python Math Utilities ===")
    
    # We create a dummy vector `[a, a', a'', a''', a'''']` 
    # representing a function and its 4 derivatives evaluated at some time t.
    a_vector = np.array([2.0, 0.5, -0.1, 0.05, -0.01])
    
    print(f"\n[Input Vector (a, a', a'', a''', a''''):]\n{a_vector}\n")

    # --- 1. Output Parity Checks ---
    print("--- 1. Output Parity (derivatives_of_inverse) ---")
    py_res_inv = py_inv(a_vector)
    cpp_res_inv = cpp_inv(a_vector)
    
    print(f"Python Result: {py_res_inv}")
    print(f"C++ Result:    {cpp_res_inv}")
    print(f"Match exactly: {np.allclose(py_res_inv, cpp_res_inv, rtol=1e-14, atol=1e-14)}\n")


    print("--- 2. Output Parity (derivatives_of_product) ---")
    py_res_prod = py_prod(a_vector)
    cpp_res_prod = cpp_prod(a_vector)
    
    print(f"Python Result: {py_res_prod}")
    print(f"C++ Result:    {cpp_res_prod}")
    print(f"Match exactly: {np.allclose(py_res_prod, cpp_res_prod, rtol=1e-14, atol=1e-14)}\n")


    # --- 2. Performance Benchmark ---
    print("--- 3. Performance Benchmark (100,000 evaluations) ---")
    n_iters = 100_000
    
    # Benchmark derivatives_of_inverse
    py_time = timeit.timeit(lambda: py_inv(a_vector), number=n_iters)
    cpp_time = timeit.timeit(lambda: cpp_inv(a_vector), number=n_iters)
    
    print(f"derivatives_of_inverse():")
    print(f"  Python time: {py_time:.4f} seconds")
    print(f"  C++ time:    {cpp_time:.4f} seconds")
    print(f"  Speedup:     {py_time / cpp_time:.2f}x faster in C++\n")

    # Benchmark derivatives_of_product
    py_time_prod = timeit.timeit(lambda: py_prod(a_vector), number=n_iters)
    cpp_time_prod = timeit.timeit(lambda: cpp_prod(a_vector), number=n_iters)
    
    print(f"derivatives_of_product():")
    print(f"  Python time: {py_time_prod:.4f} seconds")
    print(f"  C++ time:    {cpp_time_prod:.4f} seconds")
    print(f"  Speedup:     {py_time_prod / cpp_time_prod:.2f}x faster in C++\n")


if __name__ == "__main__":
    main()
