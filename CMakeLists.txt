cmake_minimum_required(VERSION 3.15...3.27)
project(astrodyn_core_cpp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# Math utils module
pybind11_add_module(math_utils_cpp 
    src/astrodyn_core/math_cpp/src/bindings.cpp
    src/astrodyn_core/math_cpp/src/math_utils.cpp
)

target_include_directories(math_utils_cpp PRIVATE src/astrodyn_core/math_cpp/include)

install(TARGETS math_utils_cpp DESTINATION astrodyn_core/math_cpp)

# GEqOE utils module (Kepler solver + conversions + jacobians)
pybind11_add_module(geqoe_utils_cpp
    src/astrodyn_core/geqoe_cpp/src/bindings.cpp
    src/astrodyn_core/geqoe_cpp/src/kepler_solver.cpp
    src/astrodyn_core/geqoe_cpp/src/conversions.cpp
    src/astrodyn_core/geqoe_cpp/src/jacobians.cpp
    src/astrodyn_core/geqoe_cpp/src/propagator_core.cpp
)

target_include_directories(geqoe_utils_cpp PRIVATE
    src/astrodyn_core/geqoe_cpp/include
    src/astrodyn_core/math_cpp/include      # for future Phase 4 math_utils reuse
)

install(TARGETS geqoe_utils_cpp DESTINATION astrodyn_core/geqoe_cpp)
