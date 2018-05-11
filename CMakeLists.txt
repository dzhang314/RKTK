cmake_minimum_required(VERSION 3.10)
project(RKTK)

set(CMAKE_CXX_STANDARD 11)

include_directories(inc)

add_compile_options(-O2 -march=native -mtune=native)

add_executable(RKTK
        bfgs_subroutines.hpp
        LinearAlgebraSubroutines.hpp
        nonlinear_optimizers.hpp
        ObjectiveFunction.hpp
        rksearch_main.cpp OrderConditionHelpers.hpp OrderConditionHelpers.cpp ObjectiveFunction.cpp)

target_link_libraries(RKTK mpfr gmp)